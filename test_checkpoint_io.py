import json
from pathlib import Path
import tempfile
import unittest
from checkpoint_io import save_checkpoint

class CheckpointTests(unittest.TestCase):
    def test_success_publishes_only_after_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);latest=root/'latest.json';events=[]
            def write(path):
                self.assertFalse(latest.exists());(path/'model').write_bytes(b'weights')
            result=save_checkpoint(root/'step-3000',latest,3000,[('model',write)],lambda stage,**kw:events.append(stage))
            self.assertEqual(json.loads(latest.read_text())['total_steps'],3000)
            self.assertTrue(json.loads((result/'checkpoint-receipt.json').read_text())['complete'])
            self.assertIn('checkpoint_hash_end',events)
    def test_interrupted_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);latest=root/'latest.json';latest.write_text('previous')
            def fail(path):
                (path/'model').write_bytes(b'partial');raise RuntimeError('injected write failure')
            with self.assertRaises(RuntimeError):save_checkpoint(root/'step-4000',latest,4000,[('model',fail)],lambda *a,**kw:None)
            self.assertEqual(latest.read_text(),'previous');self.assertFalse((root/'step-4000').exists())
            self.assertTrue((root/'.step-4000.incomplete/model').exists())
    def test_hash_failure_preserves_previous(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);latest=root/'latest.json';latest.write_text('previous')
            with patch('checkpoint_io.digest',side_effect=OSError('injected read failure')):
                with self.assertRaises(OSError):
                    save_checkpoint(root/'next',latest,4000,[('model',lambda p:(p/'model').write_bytes(b'data'))],lambda *a,**kw:None)
            self.assertEqual(latest.read_text(),'previous');self.assertFalse((root/'next').exists())
    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'saved';target.mkdir();(target/'old').write_text('keep')
            with self.assertRaises(FileExistsError):save_checkpoint(target,root/'latest',3000,[],lambda *a,**kw:None)
            self.assertEqual((target/'old').read_text(),'keep')
    def test_final_root_keeps_training_logs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'steps.jsonl').write_text('history')
            save_checkpoint(root,root/'latest.json',10000,[('model',lambda p:(p/'model').write_text('weights'))],lambda *a,**kw:None,existing_root=True)
            self.assertEqual((root/'steps.jsonl').read_text(),'history');self.assertTrue((root/'checkpoint-receipt.json').exists())

if __name__=='__main__':unittest.main()
