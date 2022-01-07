#Python Libraries
import os
import unittest
import numpy as np

#Custom Modules
import abcmrt
from generate_files import generate_file_paths, generate_speech_vectors, generate_csv, evaluate_csv

class TestABC_MRT16(unittest.TestCase):
    """
    Purpose
    -------
    This test uses the original, unimpaired speech files that the ABC_MRT16 templates are based off of.
    
    Speakers F1, F3, M3 and M4 are used in this test. This test compares the csv generated from the
    Python implementation of ABC_MRT16 to that generated from the MATLAB implementation.
    
    Original path to access these files: \\cfs2w.nist.gov\671\Projects\MCV\ITS Audio\source\Selected
    
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Purpose
        -------
        Instantiate instance of mrt object.
        Read in wav files.
    
        """
        
        #Can modify directories if needed
        F1_dir = os.path.join(os.curdir, 'source', 'F1')
        F3_dir = os.path.join(os.curdir, 'source', 'F3')
        M3_dir = os.path.join(os.curdir, 'source', 'M3')
        M4_dir = os.path.join(os.curdir, 'source', 'M4')
        
        #Generate a list of audio file paths 
        F1_paths = generate_file_paths(['F1'], F1_dir, condition='orig', pad_flag=False)
        F3_paths = generate_file_paths(['F3'], F3_dir, condition='orig', pad_flag=False)
        M3_paths = generate_file_paths(['M3'], M3_dir, condition='orig', pad_flag=False)
        M4_paths = generate_file_paths(['M4'], M4_dir, condition='orig', pad_flag=False)
        cls.file_paths = F1_paths + F3_paths + M3_paths + M4_paths

        #Generate a list of file numbers
        cls.file_nums = [abcmrt.file2number(file) for file in cls.file_paths]
        
        #Generate a list of speech vectors
        cls.speech = generate_speech_vectors(cls.file_paths)
        
        #Generate file paths for csv files
        cls.python_csv = os.path.join('csv', 'python', 'source', 'python_source.csv')
        cls.matlab_csv = os.path.join('csv', 'matlab', 'source', 'matlab_source.csv')
        cls.diff_csv = os.path.join('csv', 'diff', 'source', 'diff_source.csv')
           
    def test_source(self):
       phi_hat, success = abcmrt.process(self.speech, self.file_nums, verbose=True)  
       
       generate_csv(success, self.file_paths, self.python_csv)
       self.assertTrue(evaluate_csv(self.python_csv, self.matlab_csv, self.diff_csv))
                       
if __name__ == '__main__':
    unittest.main()    