const MAX_I64: i128 = i64::MAX as i128;

#[derive(Clone, Copy, Debug)]
enum Scalar {
    Int(i128),
    Bool,
}

fn ints(values: &[i128]) -> Vec<Scalar> {
    values.iter().copied().map(Scalar::Int).collect()
}

fn numel(shape: &[Scalar]) -> Result<i128, &'static str> {
    if shape.len() > 32 {
        return Err("invalid rank");
    }
    let mut dims = Vec::with_capacity(shape.len());
    for value in shape {
        let Scalar::Int(dim) = value else {
            return Err("invalid extent");
        };
        if *dim < 0 || *dim > MAX_I64 {
            return Err("invalid extent");
        }
        dims.push(*dim);
    }

    if dims.contains(&0) {
        return Ok(0);
    }

    let mut count = 1_i128;
    for dim in dims {
        if dim != 0 && count > MAX_I64 / dim {
            return Err("numel overflow");
        }
        count *= dim;
    }
    Ok(count)
}

fn reachable_span(
    shape: &[Scalar],
    strides: &[Scalar],
    offset: i128,
    storage: i128,
) -> Result<Option<(i128, i128)>, &'static str> {
    if shape.len() != strides.len() || offset < 0 {
        return Err("invalid descriptor");
    }
    if storage < 0 || storage > MAX_I64 {
        return Err("invalid storage size");
    }

    // Empty tensors have no reachable elements, but their descriptors must
    // still have well-typed, representable strides before admission.
    for stride in strides {
        let Scalar::Int(value) = stride else {
            return Err("invalid stride");
        };
        if value.unsigned_abs() > MAX_I64 as u128 {
            return Err("invalid stride");
        }
    }

    let n = numel(shape)?;
    if n == 0 {
        if offset > storage {
            return Err("empty view has invalid offset");
        }
        return Ok(None);
    }

    let mut lo = offset;
    let mut hi = offset;
    for (dim, stride) in shape.iter().zip(strides) {
        let Scalar::Int(dim) = dim else {
            return Err("invalid extent");
        };
        let Scalar::Int(stride) = stride else {
            return Err("invalid stride");
        };
        if stride.unsigned_abs() > MAX_I64 as u128 {
            return Err("invalid stride");
        }
        let extent = (*dim - 1)
            .checked_mul(*stride)
            .ok_or("descriptor arithmetic overflow")?;
        lo = lo
            .checked_add(extent.min(0))
            .ok_or("descriptor arithmetic overflow")?;
        hi = hi
            .checked_add(extent.max(0))
            .ok_or("descriptor arithmetic overflow")?;
    }

    if lo < 0 || hi >= storage {
        return Err("view escapes storage");
    }
    Ok(Some((lo, hi)))
}

#[test]
fn scalar_and_zero_extent() {
    assert_eq!(numel(&[]).unwrap(), 1);
    assert_eq!(numel(&ints(&[7, 0, 11])).unwrap(), 0);
    assert_eq!(numel(&ints(&[MAX_I64, MAX_I64, 0])).unwrap(), 0);
    assert_eq!(
        reachable_span(&ints(&[0, 3]), &ints(&[3, 1]), 0, 0).unwrap(),
        None
    );
}

#[test]
fn checked_shape_product() {
    assert_eq!(numel(&ints(&[3, 4, 5])).unwrap(), 60);
    assert!(numel(&ints(&[1_i128 << 62, 4])).is_err());
    assert!(numel(&ints(&[-1, 3])).is_err());
    assert!(numel(&[Scalar::Bool, Scalar::Int(2)]).is_err());
    assert!(numel(&vec![Scalar::Int(1); 33]).is_err());
}

#[test]
fn strides_reverse_and_escape() {
    assert_eq!(
        reachable_span(&ints(&[3]), &ints(&[-1]), 2, 3).unwrap(),
        Some((0, 2))
    );
    assert_eq!(
        reachable_span(&ints(&[2, 3]), &ints(&[3, 1]), 0, 6).unwrap(),
        Some((0, 5))
    );

    assert!(reachable_span(&ints(&[4]), &ints(&[1]), 0, 3).is_err());
    assert!(reachable_span(&ints(&[3]), &ints(&[-1]), 1, 3).is_err());
    assert!(reachable_span(&ints(&[1]), &ints(&[1]), 4, 4).is_err());
    assert!(reachable_span(&ints(&[1]), &ints(&[1]), -1, 4).is_err());
    assert!(reachable_span(&ints(&[2]), &[Scalar::Bool], 0, 3).is_err());
}

#[test]
fn empty_views_validate_all_strides_and_offsets() {
    assert!(reachable_span(&ints(&[0]), &[Scalar::Bool], 0, 0).is_err());
    assert!(reachable_span(&ints(&[0]), &ints(&[MAX_I64 + 1]), 0, 0).is_err());
    assert!(reachable_span(&ints(&[0]), &ints(&[-MAX_I64 - 1]), 0, 0).is_err());
    assert!(reachable_span(&ints(&[0]), &ints(&[0]), 1, 0).is_err());
    assert_eq!(reachable_span(&ints(&[0]), &ints(&[0]), 0, 0), Ok(None));
}

#[test]
fn bounding_span_is_not_proof_of_disjoint_mutable_views() {
    let a = reachable_span(&ints(&[2]), &ints(&[1]), 0, 4)
        .unwrap()
        .unwrap();
    let b = reachable_span(&ints(&[2]), &ints(&[1]), 1, 4)
        .unwrap()
        .unwrap();
    assert!(a.0 <= b.1 && b.0 <= a.1);
}
