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
    'if validate_tensor(source).is_err(); do',
    'if count_ints(source.shape) != 2; do',
    'if start < 0 || count < 0 || start > rows; do',
    'if count > rows - start; do',
    'if checked_numel(arr[count, columns], count_floats(output)).is_err(); do',
    'if count_floats(output) == 0; do',
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
  assert.match(guest, /huge_zero_columns, 0, 2000000000, empty/);
  assert.match(guest, /huge_zero_columns, 1, 2000000000, empty/);
  assert.match(guest, /huge_zero_columns, 0, 2000000000, huge_zero_columns\.values/);
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
