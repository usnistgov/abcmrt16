#Python Libraries
import os
import re
import unittest

#Custom Modules
import abcmrt
from generate_files import generate_speech_vectors, generate_csv, evaluate_csv

def generate_psud_file_num(file):
    m=re.search('Rx\d+_[MF]\d_.+_cp\d+_w(?P<word>\d+)\.wav', file)
    return int(m.group('word'))

class TestABC_MRT16(unittest.TestCase):
    """
    Purpose
    -------
    This test uses analog, P25d, P25p2 and P25t PSuD audio clips.
    
    Speakers F1, F3, M3 and M4 are used in this test. This test compares the csv generated from the
    Python implementation of ABC_MRT16 to that generated from the MATLAB implementation.
    
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
        analog_dir = os.path.join(os.curdir, 'ABC_MRT_clips', 'analog')
        P25d_dir = os.path.join(os.curdir, 'ABC_MRT_clips', 'P25d')
        P25p2_dir = os.path.join(os.curdir, 'ABC_MRT_clips', 'P25p2')
        P25t_dir = os.path.join(os.curdir, 'ABC_MRT_clips', 'P25t')
        
        #For each psud directory, generate a list of audio file paths 
        cls.analog_file_paths = [os.path.join(analog_dir, file) for file in os.listdir(analog_dir) if file.endswith(".wav")]
        cls.P25d_file_paths = [os.path.join(P25d_dir, file) for file in os.listdir(P25d_dir) if file.endswith(".wav")] 
        cls.P25p2_file_paths = [os.path.join(P25p2_dir, file) for file in os.listdir(P25p2_dir) if file.endswith(".wav")]
        cls.P25t_file_paths = [os.path.join(P25t_dir, file) for file in os.listdir(P25t_dir) if file.endswith(".wav")] 
        
        #For each psud directory, generate a list of file numbers
        cls.analog_file_nums = [generate_psud_file_num(file) for file in cls.analog_file_paths]
        cls.P25d_file_nums = [generate_psud_file_num(file) for file in cls.P25d_file_paths]
        cls.P25p2_file_nums = [generate_psud_file_num(file) for file in cls.P25p2_file_paths]
        cls.P25t_file_nums = [generate_psud_file_num(file) for file in cls.P25t_file_paths]
    
        #For each psud directory, generate a list of speech vectors
        cls.analog_speech = generate_speech_vectors(cls.analog_file_paths)
        cls.P25d_speech = generate_speech_vectors(cls.P25d_file_paths)
        cls.P25p2_speech = generate_speech_vectors(cls.P25p2_file_paths)
        cls.P25t_speech = generate_speech_vectors(cls.P25t_file_paths)
        
        #Generate file paths for csv files
        cls.python_path = os.path.join('csv', 'python', 'psud')
        cls.matlab_path = os.path.join('csv', 'matlab', 'psud')
        cls.diff_path = os.path.join('csv','diff', 'psud')
    
    def test_analog(self):
        analog_python_csv = os.path.join(self.python_path, 'python_analog.csv')
        analog_matlab_csv = os.path.join(self.matlab_path, 'matlab_analog.csv')
        analog_diff_csv = os.path.join(self.diff_path,'diff_analog.csv')
        
        analog_phi_hat, analog_success = abcmrt.process(self.analog_speech, self.analog_file_nums, verbose=True)
        
        generate_csv(analog_success, self.analog_file_paths, analog_python_csv)
        self.assertTrue(evaluate_csv(analog_python_csv, analog_matlab_csv, analog_diff_csv))
        
    def test_P25d(self):
        P25d_python_csv = os.path.join(self.python_path, 'python_P25d.csv')
        P25d_matlab_csv = os.path.join(self.matlab_path, 'matlab_P25d.csv')
        P25d_diff_csv = os.path.join(self.diff_path, 'diff_P25d.csv')
        
        P25d_phi_hat, P25d_success = abcmrt.process(self.P25d_speech, self.P25d_file_nums, verbose=True)
        
        generate_csv(P25d_success, self.P25d_file_paths, P25d_python_csv)
        self.assertTrue(evaluate_csv(P25d_python_csv, P25d_matlab_csv, P25d_diff_csv))
        
    def test_P25p2(self):
        P25p2_python_csv = os.path.join(self.python_path, 'python_P25p2.csv')
        P25p2_matlab_csv = os.path.join(self.matlab_path, 'matlab_P25p2.csv')
        P25p2_diff_csv = os.path.join(self.diff_path, 'diff_P25p2.csv')
        
        P25p2_phi_hat, P25p2_success = abcmrt.process(self.P25p2_speech, self.P25p2_file_nums, verbose=True)
        
        generate_csv(P25p2_success, self.P25p2_file_paths, P25p2_python_csv)
        self.assertTrue(evaluate_csv(P25p2_python_csv, P25p2_matlab_csv, P25p2_diff_csv))
        
    def test_P25t(self):
        P25t_python_csv = os.path.join(self.python_path, 'python_P25t.csv')
        P25t_matlab_csv = os.path.join(self.matlab_path, 'matlab_P25t.csv')
        P25t_diff_csv = os.path.join(self.diff_path, 'diff_P25t.csv')
        
        P25t_phi_hat, P25t_success = abcmrt.process(self.P25t_speech, self.P25t_file_nums, verbose=True)
        
        generate_csv(P25t_success, self.P25t_file_paths, P25t_python_csv)        
        self.assertTrue(evaluate_csv(P25t_python_csv, P25t_matlab_csv, P25t_diff_csv)) 
                     
if __name__ == '__main__':
    unittest.main()    