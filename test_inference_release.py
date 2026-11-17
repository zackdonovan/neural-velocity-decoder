"""Release-interface regressions; no model fitting or scientific selection."""
from concurrent.futures import ThreadPoolExecutor
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import infer

class InferenceReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def arrays(self):return {'prediction':np.array([[1.25,-2.5]])}
    def test_reject_missing_suffix_without_touching_existing_npz(self):
        p=self.root/'prediction.npz';p.write_bytes(b'keep')
        with self.assertRaises(ValueError):infer.write_exclusive_npz(self.root/'prediction',self.arrays())
        self.assertEqual(p.read_bytes(),b'keep')
    def test_existing_output_preserved(self):
        p=self.root/'prediction.npz';p.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):infer.write_exclusive_npz(p,self.arrays())
        self.assertEqual(p.read_bytes(),b'keep')
    def test_exact_uppercase_suffix_no_implicit_extension(self):
        p=self.root/'Prediction.NPZ';infer.write_exclusive_npz(p,self.arrays())
        self.assertTrue(p.is_file());self.assertFalse((self.root/'Prediction.NPZ.npz').exists())
        with np.load(p,allow_pickle=False) as f:np.testing.assert_array_equal(f['prediction'],self.arrays()['prediction'])
    def test_parent_directory_missing(self):
        with self.assertRaises(FileNotFoundError):infer.write_exclusive_npz(self.root/'missing/out.npz',self.arrays())
    def test_symlink_destination_preserved(self):
        target=self.root/'target';target.write_bytes(b'keep');p=self.root/'out.npz'
        try:p.symlink_to(target)
        except (OSError,NotImplementedError):self.skipTest('Host cannot create test symlinks')
        with self.assertRaises(FileExistsError):infer.write_exclusive_npz(p,self.arrays())
        self.assertEqual(target.read_bytes(),b'keep')
    def test_dangling_symlink_refused(self):
        p=self.root/'out.npz'
        try:p.symlink_to(self.root/'absent')
        except (OSError,NotImplementedError):self.skipTest('Host cannot create test symlinks')
        with self.assertRaises(FileExistsError):infer.write_exclusive_npz(p,self.arrays())
        self.assertTrue(p.is_symlink())
    def test_simultaneous_writers_only_one_succeeds(self):
        p=self.root/'out.npz'
        def write(i):
            try:infer.write_exclusive_npz(p,{'prediction':np.array([[i,i]])});return True
            except FileExistsError:return False
        with ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(write,[11,22]))
        self.assertEqual(sum(out),1)
        with np.load(p,allow_pickle=False) as f:self.assertIn(int(f['prediction'][0,0]),[11,22])
    def test_serialization_error_creates_no_file(self):
        p=self.root/'out.npz'
        with patch('infer.np.savez_compressed',side_effect=OSError('mock disk error')):
            with self.assertRaises(OSError):infer.write_exclusive_npz(p,self.arrays())
        self.assertFalse(p.exists())
    def test_caught_write_error_removes_partial_output(self):
        p=self.root/'out.npz'
        with patch('infer.os.fsync',side_effect=OSError('mock flush error')):
            with self.assertRaises(OSError):infer.write_exclusive_npz(p,self.arrays())
        self.assertFalse(p.exists())
    def test_valid_count_array_unchanged(self):
        x=np.zeros((2,7,90));u=np.arange(90)
        np.testing.assert_array_equal(infer.validate_input_arrays(x,u,u),x)
    def test_invalid_dimensions(self):
        u=np.arange(90)
        for x in [np.zeros((0,7,90)),np.zeros((1,6,90)),np.zeros((1,7,89)),np.zeros((7,90))]:
            with self.subTest(shape=x.shape),self.assertRaises(ValueError):infer.validate_input_arrays(x,u,u)
    def test_invalid_counts(self):
        u=np.arange(90)
        for value in [-1,.5,np.nan,np.inf]:
            x=np.zeros((1,7,90));x[0,0,0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):infer.validate_input_arrays(x,u,u)
        with self.assertRaises(ValueError):infer.validate_input_arrays(np.full((1,7,90),'x'),u,u)
    def test_wrong_unit_order(self):
        u=np.arange(90)
        with self.assertRaises(ValueError):infer.validate_input_arrays(np.zeros((1,7,90)),u[::-1],u)
    def test_cli_suffix_error_is_clear(self):
        with patch('sys.stderr',new_callable=io.StringIO) as err,self.assertRaises(SystemExit) as ex:
            infer.main(['--input','unused.npz','--output',str(self.root/'result')])
        self.assertEqual(ex.exception.code,2);self.assertIn('must end with .npz',err.getvalue())
    def test_repeated_call_does_not_invoke_model(self):
        p=self.root/'already.npz';p.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):infer.predict_file('missing.npz',p)
        self.assertEqual(p.read_bytes(),b'keep')

if __name__=='__main__':unittest.main(verbosity=2)
