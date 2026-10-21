const MAX_I64: i128 = i64::MAX as i128;

#[derive(Clone, Copy, Debug)]
enum Atom {
    Int(i128),
    Bool,
}

fn integer(value: Atom, message: &'static str) -> Result<i128, &'static str> {
    match value {
        Atom::Int(value) => Ok(value),
        Atom::Bool => Err(message),
    }
}

fn numel(shape: &[Atom]) -> Result<i128, &'static str> {
    if shape.len() > 32 {
        return Err("invalid rank");
    }

    let mut dims = Vec::with_capacity(shape.len());
    for &value in shape {
        let dim = integer(value, "invalid extent")?;
        if !(0..=MAX_I64).contains(&dim) {
            return Err("invalid extent");
        }
        dims.push(dim);
    }

    if dims.contains(&0) {
        return Ok(0);
    }

    let mut count = 1_i128;
    for dim in dims {
        if count > MAX_I64 / dim {
            return Err("numel overflow");
        }
        count *= dim;
    }
    Ok(count)
}

fn reachable_span(
    shape: &[Atom],
    strides: &[Atom],
    offset: Atom,
    storage: Atom,
) -> Result<Option<(i128, i128)>, &'static str> {
    if shape.len() != strides.len() {
        return Err("invalid descriptor");
    }

    let offset = integer(offset, "invalid descriptor")?;
    if offset < 0 {
        return Err("invalid descriptor");
    }
    let storage = integer(storage, "invalid storage size")?;
    if !(0..=MAX_I64).contains(&storage) {
        return Err("invalid storage size");
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
    for (&dim_atom, &stride_atom) in shape.iter().zip(strides) {
        let dim = integer(dim_atom, "invalid extent")?;
        let stride = integer(stride_atom, "invalid stride")?;
        if stride.abs() > MAX_I64 {
            return Err("invalid stride");
        }
        let extent = (dim - 1)
            .checked_mul(stride)
            .ok_or("descriptor arithmetic overflow")?;
        lo = lo.checked_add(extent.min(0)).ok_or("descriptor arithmetic overflow")?;
        hi = hi.checked_add(extent.max(0)).ok_or("descriptor arithmetic overflow")?;
    }
    if lo < 0 || hi >= storage {
        return Err("view escapes storage");
    }
    Ok(Some((lo, hi)))
}

#[cfg(test)]
mod tests {
    use super::{numel, reachable_span, Atom, MAX_I64};

    fn ints(values: &[i128]) -> Vec<Atom> {
        values.iter().copied().map(Atom::Int).collect()
    }

    #[test]
    fn scalar_and_zero_extent() {
        assert_eq!(numel(&[]), Ok(1));
        assert_eq!(numel(&ints(&[7, 0, 11])), Ok(0));
        assert_eq!(numel(&ints(&[MAX_I64, MAX_I64, 0])), Ok(0));
        assert_eq!(
            reachable_span(
                &ints(&[0, 3]),
                &ints(&[3, 1]),
                Atom::Int(0),
                Atom::Int(0)
            ),
            Ok(None)
        );
    }

    #[test]
    fn checked_shape_product() {
        assert_eq!(numel(&ints(&[3, 4, 5])), Ok(60));
        assert!(numel(&ints(&[1_i128 << 62, 4])).is_err());
        assert!(numel(&ints(&[-1, 3])).is_err());
        assert!(numel(&[Atom::Bool, Atom::Int(2)]).is_err());
        assert!(numel(&vec![Atom::Int(1); 33]).is_err());
    }

    #[test]
    fn strides_reverse_and_escape() {
        assert_eq!(
            reachable_span(
                &ints(&[3]),
                &ints(&[-1]),
                Atom::Int(2),
                Atom::Int(3)
            ),
            Ok(Some((0, 2)))
        );
        assert_eq!(
            reachable_span(
                &ints(&[2, 3]),
                &ints(&[3, 1]),
                Atom::Int(0),
                Atom::Int(6)
            ),
            Ok(Some((0, 5)))
        );

        let cases = [
            (ints(&[4]), ints(&[1]), Atom::Int(0), Atom::Int(3)),
            (ints(&[3]), ints(&[-1]), Atom::Int(1), Atom::Int(3)),
            (ints(&[1]), ints(&[1]), Atom::Int(4), Atom::Int(4)),
            (ints(&[1]), ints(&[1]), Atom::Int(-1), Atom::Int(4)),
        ];
        for (shape, strides, offset, storage) in cases {
            assert!(reachable_span(&shape, &strides, offset, storage).is_err());
        }
        assert!(
            reachable_span(
                &ints(&[2]),
                &[Atom::Bool],
                Atom::Int(0),
                Atom::Int(3)
            )
            .is_err()
        );
    }

    #[test]
    fn bounding_span_is_not_mutable_disjointness_proof() {
        let a = reachable_span(
            &ints(&[2]),
            &ints(&[1]),
            Atom::Int(0),
            Atom::Int(4),
        )
        .unwrap()
        .unwrap();
        let b = reachable_span(
            &ints(&[2]),
            &ints(&[1]),
            Atom::Int(1),
            Atom::Int(4),
        )
        .unwrap()
        .unwrap();
        assert!(a.0 <= b.1 && b.0 <= a.1);
    }
}
