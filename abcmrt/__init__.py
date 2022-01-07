"""
Estimator of MRT speech intelligibility.

The Modified Rhyme Test (MRT) is a protocol for evaluating speech
intelligibility using human subjects [3]. The subjects are presented with
the task of identifying one of six different words that take the phonetic
form CVC.  The six options differ only in the leading or trailing
consonant. MRT results take the form of success rates (corrected for
guessing) that range from 0 (guessing) to 1 (correct identification in
every case).  These success rates form a measure of speech intelligibility
in this specific (MRT) context.

The 2016 version of Articulation Band Correlation-MRT, (ABC-MRT16) is a
signal processing algorithm that processes MRT audio files and produces
success rates.

The goal of ABC-MRT16 is to produce success rates that agree with those
produced by MRT. Thus ABC-MRT16 is an automated or objective version of MRT
and no human subjects are required. ABC-MRT16 uses a very simple and
specialized speech recognition algorithm to decide which word was spoken.
This version has been tested on narrowband, wideband, superwideband, and
fullband speech.

Information on preparing test files and running ABC_MRT16.m can be found in
the readme file included in the distribution.  ABC_MRTdemo16.m shows
example use.

Methods
-------
process : Estimate intelligibility for audio.
file2number : Get file number from filename.

See Also
--------
scipy.io.wavefile.read : Function to load audio.

References
----------
[1] S. Voran "Using articulation index band correlations to objectively
estimate speech intelligibility consistent with the modified rhyme test,"
Proc. 2013 IEEE Workshop on Applications of Signal Processing to Audio and
Acoustics, New Paltz, NY, October 20-23, 2013.  Available at
www.its.bldrdoc.gov/audio.

[2] S. Voran " A multiple bandwidth objective speech intelligibility
estimator based on articulation index band correlations and attention,"
Proc. 2017 IEEE International Conference on Acoustics, Speech, and Signal
Processing, New Orleans, March 5-9, 2017.  Available at
www.its.bldrdoc.gov/audio.

[3] ANSI S3.2, "American national standard method for measuring the
intelligibility of speech over communication systems," 1989.

Examples
--------

Given an audio vector, recorded with F3_b6_w6_orig.wav, as a numpy array
in word_audio, compute the ABC MRT intelligibility.
>>>abcmrt.process(word_audio.astype(float),336)
"""

from .ABC_MRT16 import *

from .version import version

