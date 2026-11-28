// Host-only source/logic guard. This does NOT execute native Oreslang.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../src/row_batches.ores', import.meta.url), 'utf8');
const guest = readFileSync(new URL('../tests/row_batches_smoke.ores', import.meta.url), 'utf8');
const begin = source.indexOf('pub fnc slice_rows_2d_f64_into(');
const end = source.indexOf('pub fnc gather_rows_vjp_f64_checked_into(', begin);
const sliceBody = source.slice(begin, end);

test('zero-column row slice short-circuits only after all validations', () => {
  assert.ok(begin >= 0 && end > begin, 'expected public row-slice function');
  const markers = [
    'if output is source.values; do',
    'if TensorOps.validate_tensor(rt borrow source).is_err(); do',
    'if source.shape.size != 2; do',
    'if start < 0 || count < 0 || start > rows; do',
    'if count > rows - start; do',
    'val List<int> result_shape = arr[count, columns];',
    'if TensorOps.checked_numel(rt borrow result_shape, output.size).is_err(); do',
    'if output.size == 0; do',
    'return Ok(true);',
    'for int i = 0; i < count; i++ do',
  ];
  let previous = -1;
  for (const marker of markers) {
    const position = sliceBody.indexOf(marker, previous + 1);
    assert.ok(position > previous, `missing or reordered admission guard: ${marker}`);
    previous = position;
  }
});

test('native guest fixture includes enormous empty rows and rejection sentinels', () => {
  assert.match(guest, /huge_zero_columns/);
  assert.match(guest, /arr\[2000000000, 0\]/);
  assert.match(guest, /huge_zero_columns, 0, 2000000000, rt mut borrow empty/);
  assert.match(guest, /huge_zero_columns, 1, 2000000000, rt mut borrow empty/);
  assert.match(guest, /wrong_zero_slice_storage\[0\], 37\.0/);
});

test('independent descriptor oracle rejects invalid zero-column slices', () => {
  function validate(rows, columns, start, count, outputLength, aliased = false) {
    if (aliased) throw Error('alias');
    if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(columns) ||
        rows < 0 || columns < 0) throw Error('shape');
    if (start < 0 || count < 0 || start > rows || count > rows - start)
      throw Error('bounds');
    if (count * columns !== outputLength) throw Error('storage');
    return outputLength === 0 ? 'empty' : 'nonempty';
  }
  assert.equal(validate(2000000000, 0, 0, 2000000000, 0), 'empty');
  assert.throws(() => validate(2000000000, 0, 1, 2000000000, 0), /bounds/);
  assert.throws(() => validate(2000000000, 0, 0, 2000000000, 1), /storage/);
  assert.throws(() => validate(2000000000, 0, 0, 2000000000, 0, true), /alias/);
});

test('zero-output gather follows descriptor and every-index admission', () => {
  const start = source.indexOf('pub fnc gather_rows_2d_f64_into(');
  const end = source.indexOf('pub fnc slice_rows_2d_f64_into(', start);
  assert.ok(start >= 0 && end > start, 'gather function missing');
  const body = source.slice(start, end);
  const markers = [
    'if output is source.values; do',
    'if TensorOps.validate_tensor(rt borrow source).is_err(); do',
    'if source.shape.size != 2; do',
    'val List<int> result_shape = arr[selected, columns];',
    'if TensorOps.checked_numel(rt borrow result_shape, output.size).is_err(); do',
    'for int index_offset = 0; index_offset < indices.size; index_offset++ do',
    'if index < 0 || index >= rows; do',
    'if output.size == 0; do',
    'return Ok(true);',
    'for int i = 0; i < selected; i++ do',
  ];
  let last = -1;
  for (const marker of markers) {
    const index = body.indexOf(marker, last + 1);
    assert.ok(index > last, `missing or reordered gather preflight: ${marker}`);
    last = index;
  }
});

test('zero-output scatter VJP follows all descriptor, alias and index checks', () => {
  const start = source.indexOf('pub fnc gather_rows_vjp_f64_checked_into(');
  assert.ok(start >= 0, 'gather VJP missing');
  const body = source.slice(start);
  const markers = [
    'if scratch is output || scratch is upstream.values || output is upstream.values; do',
    'if TensorOps.validate_tensor(rt borrow upstream).is_err(); do',
    'if upstream.shape.size != 2 || source_shape.size != 2; do',
    'if rows < 0 || columns < 0; do',
    'if upstream.shape[0] != selected || upstream.shape[1] != columns; do',
    'if scratch.size != n; do',
    'if TensorOps.checked_numel(rt borrow source_shape, n).is_err(); do',
    'for int index_offset = 0; index_offset < indices.size; index_offset++ do',
    'if index < 0 || index >= rows; do',
    'if n == 0; do',
    'return Ok(true);',
    'for int k = 0; k < n; k++ do',
  ];
  let last = -1;
  for (const marker of markers) {
    const index = body.indexOf(marker, last + 1);
    assert.ok(index > last, `missing or reordered gather VJP preflight: ${marker}`);
    last = index;
  }
});

test('native zero-column gather and VJP fixtures keep negative-path assertions', () => {
  const markers = [
    'huge_zero_columns, rt borrow wide_repeated_indices, rt mut borrow empty',
    'huge_zero_columns, rt borrow past_wide_end, rt mut borrow empty',
    'huge_zero_columns, rt borrow negative_index, rt mut borrow empty',
    'huge_zero_columns, rt borrow first_index, rt mut borrow wrong_zero_slice_storage',
    'wrong_zero_slice_storage[0], 37.0',
    'dy_zero_columns, rt borrow wide_repeated_indices, rt borrow wide_zero_column_shape',
    'dy_zero_columns, rt borrow invalid_wide_indices, rt borrow wide_zero_column_shape',
    'zero_grad_scratch, rt mut borrow zero_grad_scratch',
  ];
  for (const marker of markers) {
    assert.ok(guest.includes(marker), `missing native acceptance assertion: ${marker}`);
  }
});
