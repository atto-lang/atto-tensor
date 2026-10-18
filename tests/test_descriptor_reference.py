"""Independent descriptor oracle only; not an Oreslang Array<T> implementation."""
import unittest
MAX_I64=(1<<63)-1

def numel(shape):
    if type(shape) not in (tuple,list) or len(shape)>32:
        raise ValueError("invalid rank")
    count=1
    for d in shape:
        if type(d) is not int or d<0 or d>MAX_I64:
            raise ValueError("invalid extent")
        # Empty tensors consume no elements even if other extents are huge.
        if d==0:
            count=0
        elif count and count>MAX_I64//d:
            raise ValueError("numel overflow")
        else:
            count*=d
    return count

def reachable_span(shape,strides,offset,storage):
    if len(shape)!=len(strides) or type(offset) is not int or offset<0:
        raise ValueError("invalid descriptor")
    if type(storage) is not int or storage<0 or storage>MAX_I64:
        raise ValueError("invalid storage size")
    n=numel(shape)
    if n==0:
        if offset>storage: raise ValueError("empty view has invalid offset")
        return None
    lo=hi=offset
    for dim,stride in zip(shape,strides):
        if type(stride) is not int or abs(stride)>MAX_I64:
            raise ValueError("invalid stride")
        extent=(dim-1)*stride
        lo+=min(extent,0)
        hi+=max(extent,0)
    if lo<0 or hi>=storage:
        raise ValueError("view escapes storage")
    return lo,hi

class TensorDescriptorReferenceTests(unittest.TestCase):
    def test_scalar_and_zero_extent(self):
        self.assertEqual(numel(()),1)
        self.assertEqual(numel((7,0,11)),0)
        self.assertEqual(reachable_span((0,3),(3,1),0,0),None)
    def test_checked_shape_product(self):
        self.assertEqual(numel((3,4,5)),60)
        for shape in ((1<<62,4),(-1,3),(True,2),(1,)*33):
            with self.subTest(shape=shape),self.assertRaises(ValueError):
                numel(shape)
    def test_strides_reverse_and_escape(self):
        self.assertEqual(reachable_span((3,),(-1,),2,3),(0,2))
        self.assertEqual(reachable_span((2,3),(3,1),0,6),(0,5))
        for spec in [((4,),(1,),0,3),((3,),(-1,),1,3),
                     ((1,),(1,),4,4),((1,),(1,),-1,4),
                     ((2,),(True,),0,3)]:
            with self.subTest(spec=spec),self.assertRaises(ValueError):
                reachable_span(*spec)
    def test_bounding_span_is_not_proof_of_disjoint_mutable_views(self):
        # Equal backing span is allowed for read-only views; a real compiler
        # must NOT infer mutable uniqueness from these arithmetic checks.
        a=reachable_span((2,),(1,),0,4)
        b=reachable_span((2,),(1,),1,4)
        self.assertTrue(a[0]<=b[1] and b[0]<=a[1])

if __name__=="__main__":
    unittest.main()
