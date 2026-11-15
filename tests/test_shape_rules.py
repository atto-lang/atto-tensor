import unittest
from reference.shape_rules import (
    ShapeError, broadcast_index, broadcast_shapes, numel, row_major_strides
)


class TensorShapeRulesTest(unittest.TestCase):
    def test_elements(self):
        self.assertEqual(numel(()), 1)
        self.assertEqual(numel((2, 3, 4)), 24)
        self.assertEqual(numel((2, 0, 3)), 0)

    def test_strides(self):
        self.assertEqual(row_major_strides(()), ())
        self.assertEqual(row_major_strides((2, 3, 4)), (12, 4, 1))
        self.assertEqual(row_major_strides((2, 0, 3)), (0, 3, 1))

    def test_broadcast(self):
        self.assertEqual(broadcast_shapes((2, 1, 4), (3, 4)), (2, 3, 4))
        self.assertEqual(broadcast_shapes((), (2, 3)), (2, 3))
        self.assertEqual(broadcast_shapes((0, 3), (1, 3)), (0, 3))
        with self.assertRaises(ShapeError):
            broadcast_shapes((0, 3), (2, 3))
        with self.assertRaises(ShapeError):
            broadcast_shapes()

    def test_index(self):
        self.assertEqual(broadcast_index((1, 2, 3), (1, 4), (2, 3, 4)), (0, 3))
        self.assertEqual(broadcast_index((1, 2), (), (2, 3)), ())
        for args in [
            ((0, 0), (2,), (2, 3)),
            ((1, 3), (2, 3), (2, 3)),
            ((False,), (1,), (1,))
        ]:
            with self.assertRaises(ShapeError):
                broadcast_index(*args)

    def test_invalid_and_overflow(self):
        for bad in [(-1,), (True,), ("2",), list(range(65))]:
            with self.assertRaises(ShapeError):
                numel(bad)
        with self.assertRaises(ShapeError):
            numel((1 << 62, 4))
        with self.assertRaises(ShapeError):
            row_major_strides((1 << 62, 4))
        with self.assertRaises(ShapeError):
            broadcast_shapes((1 << 62, 4))


if __name__ == "__main__":
    unittest.main()
