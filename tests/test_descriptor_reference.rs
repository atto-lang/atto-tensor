const MAX_I64: i128 = i64::MAX as i128;

#[derive(Clone, Copy, Debug)]
enum Value {
    Int(i128),
    Bool,
}

fn as_i64_domain(v: Value, what: &str) -> Result<i128, String> {
    match v {
        Value::Int(n) if (0..=MAX_I64).contains(&n) => Ok(n),
        Value::Int(_) => Err(format!("invalid {what}")),
        Value::Bool => Err(format!("invalid {what}")),
    }
}

fn numel(shape: &[Value]) -> Result<i128, String> {
    if shape.len() > 32 {
        return Err("invalid rank".into());
    }

    let mut dims = Vec::with_capacity(shape.len());
    for &dim in shape {
        dims.push(as_i64_domain(dim, "extent")?);
    }

    // Match the prior reference contract: a zero extent annihilates the
    // product even when an earlier left-to-right multiplication would overflow.
    if dims.contains(&0) {
        return Ok(0);
    }

    let mut count = 1_i128;
    for dim in dims {
        if count > MAX_I64 / dim {
            return Err("numel overflow".into());
        }
        count *= dim;
    }
    Ok(count)
}

fn stride_value(v: Value) -> Result<i128, String> {
    match v {
        Value::Int(n) if (-MAX_I64..=MAX_I64).contains(&n) => Ok(n),
        Value::Int(_) => Err("invalid stride".into()),
        Value::Bool => Err("invalid stride".into()),
    }
}

fn reachable_span(
    shape: &[Value],
    strides: &[Value],
    offset: Value,
    storage: Value,
) -> Result<Option<(i128, i128)>, String> {
    if shape.len() != strides.len() {
        return Err("invalid descriptor".into());
    }
    let offset = as_i64_domain(offset, "descriptor")?;
    let storage = as_i64_domain(storage, "storage size")?;
    let n = numel(shape)?;
    // Validate strides before returning for zero-extent tensors. Otherwise
    // malformed descriptors can bypass type/range checks entirely.
    let checked_strides: Vec<i128> = strides
        .iter()
        .copied()
        .map(stride_value)
        .collect::<Result<_, _>>()?;

    if n == 0 {
        if offset > storage {
            return Err("empty view has invalid offset".into());
        }
        return Ok(None);
    }

    let dims: Vec<i128> = shape
        .iter()
        .copied()
        .map(|v| as_i64_domain(v, "extent"))
        .collect::<Result<_, _>>()?;

    let mut lo = offset;
    let mut hi = offset;
    for (dim, stride) in dims.into_iter().zip(checked_strides) {
        let extent = (dim - 1)
            .checked_mul(stride)
            .ok_or("descriptor arithmetic overflow")?;
        lo = lo.checked_add(extent.min(0)).ok_or("descriptor arithmetic overflow")?;
        hi = hi.checked_add(extent.max(0)).ok_or("descriptor arithmetic overflow")?;
    }

    if lo < 0 || hi >= storage {
        return Err("view escapes storage".into());
    }
    Ok(Some((lo, hi)))
}

fn ints(values: &[i128]) -> Vec<Value> {
    values.iter().copied().map(Value::Int).collect()
}

#[test]
fn scalar_and_zero_extent() {
    assert_eq!(numel(&[]).unwrap(), 1);
    assert_eq!(numel(&ints(&[7, 0, 11])).unwrap(), 0);
    assert_eq!(numel(&ints(&[MAX_I64, MAX_I64, 0])).unwrap(), 0);
    assert_eq!(
        reachable_span(&ints(&[0, 3]), &ints(&[3, 1]), Value::Int(0), Value::Int(0)).unwrap(),
        None
    );
}

#[test]
fn checked_shape_product() {
    assert_eq!(numel(&ints(&[3, 4, 5])).unwrap(), 60);
    assert!(numel(&ints(&[(1_i128 << 62), 4])).is_err());
    assert!(numel(&ints(&[-1, 3])).is_err());
    assert!(numel(&[Value::Bool, Value::Int(2)]).is_err());
    assert!(numel(&vec![Value::Int(1); 33]).is_err());
}

#[test]
fn strides_reverse_and_escape() {
    assert_eq!(
        reachable_span(&ints(&[3]), &ints(&[-1]), Value::Int(2), Value::Int(3)).unwrap(),
        Some((0, 2))
    );
    assert_eq!(
        reachable_span(&ints(&[2, 3]), &ints(&[3, 1]), Value::Int(0), Value::Int(6)).unwrap(),
        Some((0, 5))
    );

    assert!(reachable_span(&ints(&[4]), &ints(&[1]), Value::Int(0), Value::Int(3)).is_err());
    assert!(reachable_span(&ints(&[3]), &ints(&[-1]), Value::Int(1), Value::Int(3)).is_err());
    assert!(reachable_span(&ints(&[1]), &ints(&[1]), Value::Int(4), Value::Int(4)).is_err());
    assert!(reachable_span(&ints(&[1]), &ints(&[1]), Value::Int(-1), Value::Int(4)).is_err());
    assert!(reachable_span(
        &ints(&[2]),
        &[Value::Bool],
        Value::Int(0),
        Value::Int(3)
    )
    .is_err());
}

#[test]
fn empty_views_validate_all_strides_and_offsets() {
    assert!(reachable_span(&ints(&[0]), &[Value::Bool], Value::Int(0), Value::Int(0)).is_err());
    assert!(reachable_span(&ints(&[0]), &ints(&[MAX_I64 + 1]), Value::Int(0), Value::Int(0)).is_err());
    assert!(reachable_span(&ints(&[0]), &ints(&[-MAX_I64 - 1]), Value::Int(0), Value::Int(0)).is_err());
    assert!(reachable_span(&ints(&[0]), &ints(&[0]), Value::Int(1), Value::Int(0)).is_err());
    assert_eq!(
        reachable_span(&ints(&[0]), &ints(&[0]), Value::Int(0), Value::Int(0)),
        Ok(None)
    );
}

#[test]
fn huge_valid_extents_and_strides_cannot_escape_storage() {
    // Both values fit i64, but their product greatly exceeds the storage.
    // Keep descriptor arithmetic checked even as the domain evolves.
    assert!(reachable_span(
        &ints(&[MAX_I64]),
        &ints(&[MAX_I64]),
        Value::Int(0),
        Value::Int(MAX_I64)
    ).is_err());
}

#[test]
fn zero_extent_cannot_hide_invalid_later_fields() {
    assert!(reachable_span(
        &ints(&[0, 2]),
        &[Value::Int(0), Value::Bool],
        Value::Int(0),
        Value::Int(0)
    ).is_err());
    assert!(reachable_span(
        &ints(&[0, -1]), &ints(&[0, 0]), Value::Int(0), Value::Int(0)
    ).is_err());
    assert!(reachable_span(
        &ints(&[0, MAX_I64 + 1]), &ints(&[0, 0]), Value::Int(0), Value::Int(0)
    ).is_err());
    assert_eq!(
        reachable_span(
            &ints(&[0, 1]), &ints(&[MAX_I64, -MAX_I64]),
            Value::Int(MAX_I64), Value::Int(MAX_I64)
        ), Ok(None)
    );
    assert!(reachable_span(
        &ints(&[0, 1]), &ints(&[0, 0]),
        Value::Int(MAX_I64 + 1), Value::Int(MAX_I64)
    ).is_err());
}

#[test]
fn bounding_span_is_not_proof_of_disjoint_mutable_views() {
    let a = reachable_span(&ints(&[2]), &ints(&[1]), Value::Int(0), Value::Int(4))
        .unwrap()
        .unwrap();
    let b = reachable_span(&ints(&[2]), &ints(&[1]), Value::Int(1), Value::Int(4))
        .unwrap()
        .unwrap();
    assert!(a.0 <= b.1 && b.0 <= a.1);
}
