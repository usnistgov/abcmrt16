import os
import unittest

import abcmrt
import numpy as np

from generate_files import evaluate_csv, generate_csv, generate_file_paths, generate_speech_vectors


class TestABC_MRT16(unittest.TestCase):
    @staticmethod
    def _gen_test_impaired(partnum, cnum):
        def test_impaired(self):

            # Can modify directory if needed
            audio_dir = os.path.join(
                os.path.dirname(__file__), "2008 study", f"audio2008_part{partnum}", f"c{cnum:02}_2008"
            )

            # Specify talkers
            talkers = np.array(["F1", "F3", "M3", "M4"])

            # For each condition, generate a list of audio file paths
            c_file_paths = generate_file_paths(talkers, audio_dir, condition=f"c{cnum:02}")

            # Generate list of files numbers (will be used for all conditions)
            file_num = [abcmrt.file2number(file) for file in c_file_paths]

            # For each condition, generate a list of speech vectors
            c_speech = generate_speech_vectors(c_file_paths)

            # Generate file paths for csv files
            python_csv = os.path.join(
                os.path.dirname(__file__),
                "csv",
                "python",
                f"2008 part {partnum}",
                f"python_2008_part{partnum}_c{cnum:02}.csv",
            )
            matlab_csv = os.path.join(
                os.path.dirname(__file__),
                "csv",
                "matlab",
                f"2008 part {partnum}",
                f"matlab_2008_part{partnum}_c{cnum:02}.csv",
            )
            diff_csv = os.path.join(
                os.path.dirname(__file__),
                "csv",
                "diff",
                f"2008 part {partnum}",
                f"diff_2008_part{partnum}_c{cnum:02}.csv",
            )

            # Run test
            c_phi_hat, c_success = abcmrt.process(c_speech, file_num, verbose=False)

            generate_csv(c_success, c_file_paths, python_csv)
            self.assertTrue(evaluate_csv(python_csv, matlab_csv, diff_csv))

        return test_impaired


# Manually add test functions to the TestCase.
# Executes at top level because test functions must be added at import-time or possibly via metaclass
# to be recognized by unittest. This is the simpler option.
for i in range(1, 31):
    testname = f"test_2008_part{(i-1)//10 + 1}_c{i:02}"
    setattr(TestABC_MRT16, testname, TestABC_MRT16._gen_test_impaired((i - 1) // 10 + 1, i))


if __name__ == "__main__":
    # TODO: Accept command line arguments to run tests on specific conditions
    unittest.main()
