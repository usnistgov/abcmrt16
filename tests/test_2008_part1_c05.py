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
    This test uses audio files played during Part 1 the PSCR 2008 MRT. These audio files are "impaired"
    by factors such as (ask Steve Voran) and have been sorted into 10 different condition directories
    in the form cX_2008 where X represents the condition number.
    
    Speakers F1, F3, M3 and M4 are used in this test. This test compares the success values in the csv files generated from the
    Python implementation of ABC_MRT16 to those generated from the MATLAB implementation.
    
    Documentation of the PSCR 2008 MRT experiment: http://www.its.bldrdoc.gov/publications/2490.aspx

    Original path to access these files: \\cfs2w.nist.gov\671\Projects\MCV\ITS Audio\part1
    Link to access these files online: https://www.its.bldrdoc.gov/outreach/audio/mrt_library/test_audio_and_results_files/
    
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Purpose
        -------
        Instantiate instance of mrt object.
        Prepare necessary audio clips.
    
        """
        
        #Can modify directory if needed
        audio_dir = os.path.join(os.curdir, '2008 study', 'audio2008_part1', 'c05_2008')
        
        #Specify talkers
        talkers = np.array(['F1', 'F3', 'M3', 'M4'])
       
        #For each condition, generate a list of audio file paths 
        cls.c05_file_paths = generate_file_paths(talkers, audio_dir, condition='c05')
        
        #Generate list of files numbers (will be used for all conditions)
        cls.file_num = [abcmrt.file2number(file) for file in cls.c05_file_paths]
         
        #For each condition, generate a list of speech vectors  
        cls.c05_speech = generate_speech_vectors(cls.c05_file_paths)

        #Generate file paths for csv files
        cls.python_csv = os.path.join('csv', 'python', '2008 part 1', 'python_2008_part1_c05.csv')
        cls.matlab_csv = os.path.join('csv', 'matlab', '2008 part 1', 'matlab_2008_part1_c05.csv')
        cls.diff_csv = os.path.join('csv','diff', '2008 part 1', 'diff_2008_part1_c05.csv')
        
    def test_impaired_c05(self): 
        c05_phi_hat, c05_success = abcmrt.process(self.c05_speech, self.file_num, verbose=True)
        
        generate_csv(c05_success, self.c05_file_paths, self.python_csv) 
        self.assertTrue(evaluate_csv(self.python_csv, self.matlab_csv, self.diff_csv))
        
if __name__ == '__main__':
    unittest.main()    