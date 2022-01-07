# ABC-MRT16 Python Package
Code for running articulation band correlation - modified rhyme tests (ABC-MRT). 

# Building and Installing the Package Locally

To build and install the package, clone this repository and run the following
from the root of the git repository :

```
pip install .
```

## Background
This software implements the ABC-MRT16 algorithm for objective estimation of
speech intelligibility.  The algorithm is discussed in detail in [1] and [2].
ABC-MRT is short for “Articulation Band Correlation Modified Rhyme Test.” The
software was  originally written by NTIA/ITS in MATLAB 
(https://github.com/NTIA/ABC-MRT16), but has now been ported to Python.

The Modified Rhyme Test (MRT) [3] is a protocol for evaluating speech 
intelligibility using human subjects. The subjects are presented with the 
task of identifying one of six different words that take the phonetic form 
CVC.  The six options differ only in the leading or trailing consonant. 
MRT results take the form of success rates (corrected for guessing) that 
range from 0 (guessing) to 1 (correct identification in every case).  
These success rates form a measure of speech intelligibility in this 
specific (MRT) context.

Articulation Band Correlation-MRT (ABC-MRT) is a signal processing 
algorithm that processes MRT audio files and produces success rates.  The 
goal of ABC-MRT is to produce success rates that agree with those produced 
by MRT. Thus ABC-MRT is an automated or objective version of MRT and no 
human subjects are required.
  
ABC-MRT performs a narrowband (nominally 4 kHz) analysis. ABC-MRT16
is applicable to narrowband, wideband, superwideband, and fullband speech.
ABC-MRT processes the first 17 AI bands while ABC-MRT16 processes all 20 AI
bands, as well as an additional "AI Band 21" that covers 7 kHz to 20 kHz.
Of equal importance is that ABC-MRT16 incorporates a model for attention
that allows it to properly operate across the different bandwidths without
any bandwidth detection or switching.

Unless backwards compatibility is required, ABC-MRT16 is the recommended
algorithm, even if only narrowband conditions are to be tested. The 
attention model makes it superior to ABC-MRT. ABC-MRT16 is the only algorithm
that has been ported to Python.

The software provided here runs using the Python interpreter.

Application of ABC-MRT(16) to a speech communication system-under-test (SUT) 
requires two steps.
1.  Pass a set of reference recordings through the SUT to produce a set 
of test recordings.
2.  Apply ABC-MRT(16) to the test recordings to produce a success rates that 
describe the intelligibility of the SUT.

All of these steps are run by the intelligibility measurement software found at
<https://github.com/usnistgov/intelligibility>. Although it is possible to use
stand alone, the GUI available at <https://github.com/usnistgov/mcvqoe> is also
recommended.


## References
[1] S. Voran "Using articulation index band correlations to objectively 
estimate speech intelligibility consistent with the modified rhyme test," 
Proc. 2013 IEEE Workshop on Applications of Signal Processing to Audio and
Acoustics, New Paltz, NY, October 20- 23, 2013.  Available at 
www.its.bldrdoc.gov/audio after October 20, 2013.

[2] S. Voran "A multiple bandwidth objective speech intelligibility 
estimator based on articulation index band correlations and attention,"
Proc. 2017 IEEE International Conference on Acoustics, Speech, and 
Signal Processing, New Orleans, March 5-9, 2017.  Available at
www.its.bldrdoc.gov/audio.

[3] ANSI S3.2, "American national standard method for measuring the 
intelligibility of speech over communication systems," 1989.


## Legal

### NTIA

THE NATIONAL TELECOMMUNICATIONS AND INFORMATION ADMINISTRATION, INSTITUTE 
FOR TELECOMMUNICATION SCIENCES ("NTIA/ITS") DOES NOT MAKE ANY WARRANTY OF 
ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, 
THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
NON-INFRINGEMENT AND DATA ACCURACY.  THIS SOFTWARE IS PROVIDED "AS IS."  
NTIA/ITS does not warrant or make any representations regarding the use of 
the software or the results thereof, including but not limited to the 
correctness, accuracy, reliability or usefulness of the software or the 
results.

You can use, copy, modify, and redistribute the NTIA/ITS developed 
software upon your acceptance of these terms and conditions and upon your 
express agreement to provide appropriate acknowledgments of NTIA's 
ownership of and development of the software by keeping this exact text 
present in any copied or derivative works.

The user of this Software ("Collaborator") agrees to hold the U.S. 
Government harmless and indemnifies the U.S. Government for all 
liabilities, demands, damages, expenses, and losses arising out of
the use by the Collaborator, or any party acting on its behalf, of 
NTIA/ITS' Software, or out of any use, sale, or other disposition by the 
Collaborator, or others acting on its behalf, of products made
by the use of NTIA/ITS' Software.

### NIST

This software was developed by employees of the National Institute of Standards 
and Technology (NIST), an agency of the Federal Government. Pursuant to title 17 
United States Code Section 105, works of NIST employees are not subject to 
copyright protection in the United States and are considered to be in the public 
domain. Permission to freely use, copy, modify, and distribute this software and 
its documentation without fee is hereby granted, provided that this notice and 
disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER 
EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY 
THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, 
AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY 
WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE 
FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR 
CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED 
WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR 
OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR 
OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE 
RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
