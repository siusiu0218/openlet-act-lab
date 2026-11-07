import pickle,tempfile,unittest
from pathlib import Path
from validation_cache import ValidationCache
class Codec:
    @staticmethod
    def save(x,p):Path(p).write_bytes(pickle.dumps(x))
    @staticmethod
    def load(p,**kw):return pickle.loads(Path(p).read_bytes())
class Data:
    def __init__(self):self.calls=0
    def __getitem__(self,i):self.calls+=1;return {'value':[i,0.25], 'task':'fixed'}
class Tests(unittest.TestCase):
    def test_repeat_read_is_exact_and_does_not_access_source(self):
        with tempfile.TemporaryDirectory() as d:
            source=Data();c=ValidationCache(source,range(350),Path(d)/'cache',Codec)
            a=c[92];a['value'][1]=99
            self.assertEqual(c[92],{'value':[92,0.25],'task':'fixed'});self.assertEqual(source.calls,1)
    def test_corruption_rejected_without_source_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            source=Data();c=ValidationCache(source,range(350),Path(d)/'cache',Codec);c[92]
            (c.directory/'92.pt').write_bytes(b'bad')
            with self.assertRaises(ValueError):c[92]
            self.assertEqual(source.calls,1)
    def test_capacity_and_plan_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            c=ValidationCache(Data(),range(350),Path(d)/'cache',Codec,cap=1)
            with self.assertRaises(ValueError):c[350]
            with self.assertRaises(ValueError):c[92]
            self.assertEqual(c.stats()['entries'],0)
if __name__=='__main__':unittest.main()
