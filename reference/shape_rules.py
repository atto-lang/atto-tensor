"""Bounded row-major tensor shape reference rules (HOST ONLY, not Atto runtime)."""

MAX_RANK = 64
MAX_ELEMENTS = (1 << 63) - 1


class ShapeError(ValueError):
    """Invalid shape, incompatible broadcast, or unrepresentable element count."""


def _shape(dims):
    if not isinstance(dims, (tuple, list)) or len(dims) > MAX_RANK:
        raise ShapeError("shape must be a list/tuple of at most 64 dimensions")
    if any(type(d) is not int or d < 0 for d in dims):
        raise ShapeError("dimensions must be nonnegative integers (not booleans)")
    return tuple(dims)


def numel(shape):
    """Number of elements; rank-zero scalar has one element."""
    dims = _shape(shape)
    if 0 in dims:
        return 0
    count = 1
    for dim in dims:
        if dim and count > MAX_ELEMENTS // dim:
            raise ShapeError("element count exceeds signed 64-bit range")
        count *= dim
    return count


def row_major_strides(shape):
    """Strides in elements, including for a zero-sized axis."""
    dims = _shape(shape)
    strides = [0] * len(dims)
    stride = 1
    for index in range(len(dims) - 1, -1, -1):
        strides[index] = stride
        stride *= dims[index]
        if stride > MAX_ELEMENTS:
            raise ShapeError("stride product exceeds signed 64-bit range")
    return tuple(strides)


def broadcast_shapes(*shapes):
    """Right-aligned NumPy-style broadcast (1 expands, including to 0)."""
    if not shapes:
        raise ShapeError("at least one shape is required")
    dims = [_shape(shape) for shape in shapes]
    rank = max(map(len, dims))
    output = []
    for axis in range(rank):
        values = [shape[-axis - 1] if len(shape) > axis else 1 for shape in dims]
        non_one = {value for value in values if value != 1}
        if len(non_one) > 1:
            raise ShapeError("incompatible broadcast dimensions")
        output.append(next(iter(non_one)) if non_one else 1)
    result = tuple(reversed(output))
    numel(result)
    return result


def broadcast_index(output_index, source_shape, output_shape):
    """Map an output coordinate back to a broadcast source coordinate."""
    source = _shape(source_shape)
    target = _shape(output_shape)
    if len(source) > len(target) or broadcast_shapes(source, target) != target:
        raise ShapeError("source does not broadcast to the requested output")
    if not isinstance(output_index, (tuple, list)) or len(output_index) != len(target):
        raise ShapeError("output coordinate rank mismatch")
    if any(type(i) is not int or i < 0 or i >= d
           for i, d in zip(output_index, target)):
        raise ShapeError("output coordinate out of bounds")
    aligned = (1,) * (len(target) - len(source)) + source
    return tuple(0 if d == 1 else i for i, d in zip(output_index, aligned))[-len(source):] if source else ()
