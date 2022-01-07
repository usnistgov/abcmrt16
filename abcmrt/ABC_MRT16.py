# Python Libraries
import io
import math
import os.path
import pkgutil
import re
from warnings import warn

import numpy as np
import numpy.matlib
import scipy.io as sio
import scipy.signal

"""
 --------------------------Background--------------------------
 ABC_MRT16.m implements the ABC-MRT16 algorithm for objective estimation of
 speech intelligibility.  The algorithm is discussed in detail in [1] and
 [2].

 The Modified Rhyme Test (MRT) is a protocol for evaluating speech
 intelligibility using human subjects [3]. The subjects are presented
 with the task of identifying one of six different words that take the
 phonetic form CVC.  The six options differ only in the leading or
 trailing consonant. MRT results take the form of success rates
 (corrected for guessing) that range from 0 (guessing) to 1
 (correct identification in every case).  These success rates form a
 measure of speech intelligibility in this specific (MRT) context.

 The 2016 version of Articulation Band Correlation-MRT, (ABC-MRT16) is a
 signal processing algorithm that processes MRT audio files and produces
 success rates.

 The goal of ABC-MRT16 is to produce success rates that agree with those
 produced by MRT. Thus ABC-MRT16 is an automated or objective version of
 MRT and no human subjects are required. ABC-MRT16 uses a very simple and
 specialized speech recognition algorithm to decide which word was spoken.
 This version has been tested on narrowband, wideband, superwideband,
 and fullband speech.

 Information on preparing test files and running ABC_MRT16.m can be found
 in the readme file included in the distribution.  ABC_MRTdemo16.m shows
 example use.

 --------------------------References--------------------------
 [1] S. Voran "Using articulation index band correlations to objectively
 estimate speech intelligibility consistent with the modified rhyme test,"
 Proc. 2013 IEEE Workshop on Applications of Signal Processing to Audio and
 Acoustics, New Paltz, NY, October 20-23, 2013.  Available at
 www.its.bldrdoc.gov/audio.

 [2] S. Voran " A multiple bandwidth objective speech intelligibility
 estimator based on articulation index band correlations and attention,"
 Proc. 2017 IEEE International Conference on Acoustics, Speech, and
 Signal Processing, New Orleans, March 5-9, 2017.  Available at
 www.its.bldrdoc.gov/audio.

 [3] ANSI S3.2, "American national standard method for measuring the
 intelligibility of speech over communication systems," 1989.

 --------------------------Legal--------------------------
 THE NATIONAL TELECOMMUNICATIONS AND INFORMATION ADMINISTRATION,
 INSTITUTE FOR TELECOMMUNICATION SCIENCES ("NTIA/ITS") DOES NOT MAKE
 ANY WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING,
 WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR
 A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY.  THIS SOFTWARE
 IS PROVIDED "AS IS."  NTIA/ITS does not warrant or make any
 representations regarding the use of the software or the results thereof,
 including but not limited to the correctness, accuracy, reliability or
 usefulness of the software or the results.

 You can use, copy, modify, and redistribute the NTIA/ITS developed
 software upon your acceptance of these terms and conditions and upon
 your express agreement to provide appropriate acknowledgments of
 NTIA's ownership of and development of the software by keeping this
 exact text present in any copied or derivative works.

 The user of this Software ("Collaborator") agrees to hold the U.S.
 Government harmless and indemnifies the U.S. Government for all
 liabilities, demands, damages, expenses, and losses arising out of
 the use by the Collaborator, or any party acting on its behalf, of
 NTIA/ITS' Software, or out of any use, sale, or other disposition by
 the Collaborator, or others acting on its behalf, of products made
 by the use of NTIA/ITS' Software.

"""

__all__ = ["process", "file2number", "guess_correction", "fs", "file_order", "number2file", "load_templates"]

"""
sample rate that ABC_MRT uses for audio
"""
fs = 48000


class TemplateLoader:
    """
    simple class for abcmrt template loading.
    """

    def __init__(self):
        self.loaded = False
        self.templates = None

    def load_templates(self):
        """
        Purpose
        -------
        Loads in the speech templates for the MRT words from
        ABC_MRT_FB_templates.mat which contains a 1 by 1200 cell array TFtemplatesFB.
        Each cell in TFtemplatesFB contains a fullband time-frequency template for
        one of the 1200 talker x keyword combinations.

        Parameters
        ----------
        None

        Returns
        -------
            TFtemplatesFB : numpy vector
                            fullband time-frequency templates for the 1200 talker x keyword combinations
        """

        # Read templates file and make a file like object to pass to loadmat
        stream = io.BytesIO(pkgutil.get_data(__name__, "templates/ABC_MRT_FB_templates.mat"))

        TFtemplatesFB = sio.loadmat(stream)
        TFtemplatesFB = TFtemplatesFB["TFtemplatesFB"]

        self.templates = TFtemplatesFB
        self.loaded = True


def _makeAI():
    """
    Purpose
    -------
    Generates 21 by 215 matrix that maps FFT bins 1 to 215 to 21 AI bands.
    These are the AI bands specified on page 38 of the book:
    S. Quackenbush, T. Barnwell and M. Clements, "Objective measures of
    speech quality," Prentice Hall, Englewood Cliffs, NJ, 1988.

    Parameters
    ----------
    None

    Returns
    -------
    AI : numpy array
         21 by 215 matrix that maps FFT bins 1 to 215 to 21 AI bands.
    """

    AIlims = np.array(
        [
            [4, 4],  # AI band 1
            [5, 6],
            [7, 7],
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
            [16, 17],
            [18, 19],
            [20, 21],
            [22, 23],
            [24, 26],
            [27, 28],
            [29, 31],
            [32, 35],
            [36, 40],  # AI band  16
            [41, 45],  # AI band  17
            [46, 52],  # AI band  18
            [53, 62],  # AI band  19
            [63, 76],  # AI band  20
            [77, 215],
        ]
    )  # Everything above AI band 20 and below 20 kHz makes "AI band 21"

    AI = np.zeros((21, 215))

    for k in range(21):
        firstfreq = AIlims[k, 0]
        lastfreq = AIlims[k, 1]
        AI[k, (firstfreq - 1) : lastfreq] = 1

    return AI


# FFT bins to use for time alignment
_ALIGN_BINS = np.arange(6, 9)

# Generate a 21 by 215 matrix that maps 215 FFT bins to 21 AI bands
AI = _makeAI()

# Template loader for data
_loader = TemplateLoader()

# Number of FFT bins in each AI band
binsPerBand = np.sum(AI, axis=1, keepdims=True)


def file2number(file):
    """
    Get file number from filename.

    Compute the file number for a given MRT filename. All directory components
    are stripped from the filename with os.path.basename. The filename must have
    the string '[talker]_b[batch]_w[word]' somewhere in the filename.

    Parameters
    ----------
    file : str
        File name to extract file number from.

    Returns
    -------
    int
        File number to pass to abcmrt.process for this file. If the file number
        could not be determined, None will be returned.

    See Also
    --------
    os.path.basename : Function to remove directory names from path.
    abcmrt.process : Estimate MRT speech intelligibility.

    Examples
    --------

    Get the file number for 'M3_b24_w2_orig.wav'
    >>>abcmrt.file2number('M3_b24_w2_orig.wav')
    740
    """
    # remove extraneous path components
    name = os.path.basename(file)
    # find talker, batch and word from filename
    m = re.search(r"(?P<talker>[MF]\d)_b(?P<batch>\d+)_w(?P<word>\d+)", name)
    # check if we got a match
    if not m:
        return None
    try:
        # get talker index from talker string
        talker_index = ("F1", "F3", "M3", "M4").index(m.group("talker"))
    except ValueError:
        # invalid talker number
        return None
    # return file number
    return talker_index * 300 + (int(m.group("batch")) - 1) * 6 + int(m.group("word"))


def number2file(num):
    """
    Get file name from file number.

    Parameters
    ----------
    num : int
        Integer between 1 and 1200.

    Returns
    -------
    str:
        File name associated with file number num.

    See Also
    --------
    abcmrt.file2number

    abcmrt.file_order
    """
    if num < 1 or num > 1200:
        raise ValueError(f"Invalid input {num} must be between 1-1200.")
    talkers = ["F1", "F3", "M3", "M4"]
    batches = 50
    words = 6

    N_words = batches * words

    talker_index = int(np.floor((num - 1) / N_words))
    talker_offset = (num - 1) % N_words + 1

    batch_index = int(np.floor((talker_offset - 1) / words)) + 1
    word_index = (num - 1) % words + 1

    file = f"{talkers[talker_index]}_b{batch_index}_w{word_index}.wav"
    return file


def file_order():
    """
    Get file order for approximating full MRT with a smaller test.

    Running an intelligibility test with all 1200 MRT files can take a long
    time. This function returns a list of file numbers to in order of
    minimizing the RMSE between the first N files in this list and all 1200.

    Returns
    -------
    list:
        Ordering of files for approximating a full MRT test with a smaller
        subset of files.

    See Also
    --------
    number2file()

    Examples
    --------
    Get the 16 MRT file numbers that best approximate a full MRT test of 1200
    files.

    >>> f_order = file_order()
    >>> subset_file_numbers = f_order[:16]
    """
    # fmt: off
    file_order = [
        232, 393, 1068, 729, 230, 470, 910, 831, 288, 562, 1174, 632,
        237, 452, 955, 885, 7, 515, 1119, 838, 92, 545, 1038, 658,
        126, 600, 1045, 694, 10, 505, 1063, 864, 264, 405, 1113, 870,
        104, 540, 1105, 856, 111, 431, 1086, 853, 261, 430, 1142, 670,
        223, 343, 1065, 690, 91, 570, 1173, 741, 279, 364, 1128, 787,
        239, 548, 1076, 634, 55, 372, 911, 898, 190, 315, 935, 624,
        11, 437, 1073, 899, 229, 583, 1110, 620, 58, 380, 907, 731,
        253, 527, 1047, 604, 138, 519, 1176, 692, 51, 336, 1008, 883,
        43, 322, 1094, 719, 135, 346, 1067, 605, 73, 386, 995, 852,
        116, 567, 1131, 665, 83, 318, 974, 782, 18, 304, 1035, 644,
        65, 435, 968, 747, 78, 558, 957, 797, 134, 508, 927, 684,
        79, 512, 1044, 752, 269, 523, 914, 681, 56, 598, 1170, 863,
        194, 327, 986, 699, 278, 553, 1198, 778, 87, 469, 1077, 613,
        33, 363, 1186, 619, 248, 467, 1147, 647, 93, 493, 1134, 611,
        179, 348, 1029, 637, 98, 533, 1034, 725, 256, 373, 1020, 742,
        182, 411, 904, 735, 131, 561, 912, 674, 160, 599, 1157, 892,
        42, 530, 1064, 846, 35, 537, 973, 607, 5, 459, 958, 614,
        21, 433, 1049, 862, 287, 378, 1054, 738, 169, 366, 949, 859,
        30, 588, 1175, 889, 45, 522, 1200, 783, 71, 451, 1015, 877,
        90, 310, 953, 779, 119, 547, 963, 625, 147, 463, 1096, 643,
        69, 303, 921, 734, 240, 455, 903, 745, 227, 499, 919, 786,
        296, 355, 1059, 851, 198, 531, 1071, 888, 299, 338, 1166, 645,
        53, 389, 1033, 775, 174, 555, 1006, 636, 32, 528, 1193, 649,
        110, 595, 1108, 823, 188, 388, 1167, 679, 15, 368, 1053, 816,
        241, 395, 970, 628, 60, 449, 985, 746, 123, 311, 925, 763,
        189, 482, 1120, 895, 38, 396, 1101, 689, 102, 385, 1005, 688,
        149, 413, 1090, 672, 39, 453, 1195, 821, 96, 399, 1060, 654,
        206, 438, 1050, 873, 6, 471, 1000, 798, 81, 542, 1039, 790,
        176, 546, 1155, 618, 72, 342, 1004, 715, 209, 485, 965, 696,
        66, 305, 1130, 847, 20, 302, 972, 805, 224, 502, 1125, 865,
        291, 323, 1042, 606, 254, 432, 1103, 837, 244, 481, 1055, 891,
        136, 574, 983, 768, 105, 337, 952, 834, 178, 436, 1014, 811,
        27, 448, 971, 615, 75, 325, 984, 900, 17, 365, 1152, 602,
        211, 447, 946, 663, 202, 458, 1082, 820, 108, 592, 1168, 667,
        273, 320, 908, 617, 24, 359, 1153, 718, 88, 356, 1137, 609,
        225, 424, 1019, 621, 124, 333, 1021, 784, 29, 575, 1124, 722,
        50, 503, 928, 650, 4, 367, 937, 836, 300, 406, 1080, 874,
        3, 466, 948, 673, 228, 301, 966, 867, 243, 560, 967, 802,
        270, 410, 1148, 770, 193, 489, 962, 815, 120, 312, 1017, 695,
        294, 525, 1074, 845, 62, 507, 964, 794, 74, 496, 1140, 832,
        155, 397, 933, 603, 238, 423, 1159, 677, 250, 416, 1066, 732,
        140, 468, 1189, 709, 213, 351, 1143, 793, 284, 500, 1129, 796,
        231, 591, 943, 861, 137, 513, 924, 635, 222, 554, 1112, 601,
        205, 349, 1139, 622, 59, 381, 1056, 693, 185, 426, 1133, 739,
        121, 326, 1185, 855, 268, 417, 1081, 702, 25, 510, 1144, 764,
        196, 371, 1156, 894, 207, 403, 956, 785, 272, 335, 994, 707,
        181, 573, 1115, 814, 118, 314, 1095, 659, 247, 543, 1085, 766,
        67, 306, 1037, 743, 106, 441, 989, 765, 129, 509, 990, 887,
        9, 572, 1197, 882, 28, 439, 1031, 881, 285, 324, 1199, 843,
        41, 375, 1183, 698, 113, 421, 1190, 753, 258, 587, 1135, 827,
        221, 370, 1180, 686, 165, 552, 1093, 701, 19, 520, 960, 705,
        1, 332, 930, 849, 46, 404, 1111, 676, 86, 425, 1003, 675,
        167, 358, 981, 803, 82, 345, 939, 767, 141, 532, 1002, 750,
        54, 487, 1132, 835, 122, 420, 961, 840, 12, 490, 901, 869,
        40, 462, 906, 876, 22, 565, 916, 733, 34, 446, 1016, 875,
        130, 422, 920, 809, 133, 491, 1122, 706, 95, 392, 1181, 700,
        262, 394, 1041, 669, 13, 414, 918, 662, 297, 504, 1136, 612,
        293, 486, 1024, 854, 283, 564, 1179, 780, 49, 461, 1123, 668,
        186, 450, 1163, 710, 94, 475, 950, 751, 215, 328, 1010, 817,
        281, 353, 1026, 776, 180, 580, 932, 757, 26, 494, 1048, 848,
        245, 473, 945, 826, 36, 480, 1187, 703, 208, 347, 922, 655,
        85, 506, 1165, 858, 84, 495, 1164, 657, 242, 511, 1107, 866,
        101, 369, 977, 812, 267, 402, 1011, 781, 277, 549, 1092, 829,
        107, 465, 1087, 850, 226, 443, 1098, 656, 195, 568, 1036, 726,
        233, 377, 1072, 748, 263, 445, 942, 841, 212, 362, 917, 661,
        197, 418, 1062, 756, 217, 400, 1178, 680, 151, 581, 1075, 744,
        158, 390, 1091, 758, 117, 484, 1114, 691, 168, 488, 1069, 685,
        164, 354, 1106, 652, 109, 586, 941, 678, 187, 419, 1127, 740,
        114, 407, 1145, 804, 172, 563, 1089, 736, 77, 309, 1028, 789,
        163, 412, 1079, 806, 139, 516, 905, 721, 286, 517, 1100, 819,
        89, 440, 1078, 697, 153, 539, 1154, 626, 31, 360, 1102, 727,
        57, 341, 1109, 687, 234, 529, 1177, 818, 112, 340, 1184, 683,
        298, 329, 999, 860, 218, 429, 1032, 801, 44, 571, 1057, 724,
        246, 387, 926, 642, 97, 374, 1083, 671, 16, 566, 951, 760,
        152, 556, 1023, 714, 132, 313, 1097, 641, 184, 357, 1138, 711,
        143, 376, 1158, 791, 70, 307, 1118, 761, 260, 534, 982, 871,
        68, 589, 1104, 651, 157, 478, 1196, 795, 252, 541, 940, 828,
        154, 409, 923, 716, 183, 476, 1099, 886, 249, 474, 969, 897,
        251, 401, 1009, 830, 290, 352, 938, 728, 216, 316, 1161, 629,
        292, 308, 1116, 666, 236, 524, 1027, 755, 204, 536, 959, 842,
        144, 501, 979, 825, 37, 331, 1058, 712, 48, 384, 1149, 627,
        14, 569, 1001, 762, 173, 582, 1188, 844, 148, 557, 997, 810,
        23, 319, 1169, 723, 265, 428, 1040, 788, 170, 492, 929, 769,
        259, 514, 1182, 759, 274, 350, 993, 737, 257, 334, 975, 833,
        201, 464, 1084, 772, 255, 434, 947, 749, 125, 577, 936, 730,
        266, 383, 1146, 708, 156, 579, 980, 717, 99, 330, 944, 660,
        171, 361, 1051, 610, 47, 596, 915, 648, 127, 454, 1061, 890,
        219, 584, 1030, 896, 235, 518, 998, 839, 52, 460, 1162, 773,
        271, 521, 1126, 878, 275, 408, 1192, 631, 100, 578, 978, 893,
        282, 483, 1025, 704, 289, 576, 1141, 608, 146, 457, 1172, 774,
        103, 498, 987, 777, 276, 550, 1013, 884, 177, 479, 934, 813,
        142, 544, 1088, 638, 145, 593, 1043, 808, 150, 339, 931, 868,
        166, 477, 996, 664, 199, 551, 1012, 807, 220, 597, 1117, 880,
        115, 398, 1171, 879, 61, 317, 1160, 799, 63, 497, 976, 623,
        159, 472, 1046, 824, 128, 344, 1121, 653, 191, 535, 1150, 771,
        295, 594, 902, 633, 200, 590, 1070, 754, 161, 321, 1007, 800,
        175, 379, 1191, 720, 64, 526, 1052, 630, 80, 415, 1018, 872,
        203, 427, 954, 640, 162, 444, 1022, 639, 76, 559, 991, 857,
        214, 391, 992, 792, 210, 442, 1151, 646, 192, 382, 909, 822,
        2, 538, 913, 616, 280, 456, 1194, 682, 8, 585, 988, 713]
    # fmt: on
    return file_order


def guess_correction(intell):
    """
    Correct intelligibility estimates for guessing.

    In the case that a MRT test subject has no information about which clip the
    original audio was, they must make a guess. Given that there are 6 words to
    chose from one would expect them to be right about 1/6th of the time,
    however in this case the intelligibility of the audio is essentially zero.
    To correct for this the scores are transformed so that a score of 1/6th
    becomes a zero and a score of 1 remains a 1.

    Parameters
    ----------
    intell : float
        Intelligibility value to be corrected.

    Returns
    -------
    float
        intelligibility corrected for guessing.

    See Also
    --------
    abcmrt.process : function to estimate intelligibility.

    Examples
    --------
    Correct for guessing in the case of no intelligibility.
    >>> guess_correction(1/6)
    0

    """
    return (6 / 5) * (intell - (1 / 6))


def load_templates():
    """
    Load abcmrt templates if not loaded.

    Checks if templates have been loaded and loads them if they are not loaded.

    See Also
    --------
    abcmrt.process : Use templates to process audio

    Examples
    --------

    Load templates, this should take a bit.
    >>> abcmrt.load_templates()

    """

    if not _loader.loaded:
        _loader.load_templates()


def process(speech, file_num, verbose=False):
    """
    Estimate MRT speech intelligibility.

    Processes audio, using the ABC_MRT16 algorithm, to get an estimated speech
    intelligibility.

    Parameters
    ----------
    speech : list of numpy vectors or single numpy vector
        Speech from audio clips.

    file_num : list of ints or single int
        Original speech file number. Gives the number of the original speech
        file used to record `speech` Has the same number of elements as `speech`.
        Given a talker ordering of ('F1','F3','M3','M4') `file_num is`
        determined as `talker_index*300 + (batch_index - 1)*6 + word_num`.

    verbose : bool, default=False
        If True, causes the status of the trials to be displayed on the console.

    Returns
    -------
    phi_hat : int
        Average intelligibility over all words corrected for guessing.

    success : numpy vector
        Intelligibility of each individual word not corrected for guessing.

    See Also
    --------
    scipy.io.wavefile.read : Function to load audio.
    abcmrt.file2number : Get file number from filename.

    Examples
    --------

    load in the source audio clip 'M3_b24_w2_orig.wav' and run it through abcmrt.
    >>>import abcmrt
    >>>import scipy.io.wavfile
    >>>file_num=abcmrt.file2number(file_name)
    >>>(fs,orig_audio)=scipy.io.wavfile.read(file_name)
    >>>orig_audio_float=orig_audio.astype(float)
    >>>abcmrt.process(orig_audio_float,file_num)
    This should return (1.0, array([1.])), which makes sense because
    'M3_b24_w2_orig.wav' is the original clip.
    Now add some noise.
    >>>import numpy as np
    >>>noise = np.random.normal(0,0.15*max(abs(orig_audio_float), orig_audio_float.shape)
    >>>noisy_audio=orig_audio_float+noise
    >>>abcmrt.process(noisy_audio,file_num)
    This should give values less than 1
    """
    # make sure templates are loaded
    load_templates()
    # Handle single audio file case
    # Wrap single speech vector in a list
    if not isinstance(speech, list):
        speech = [speech]
        success = np.zeros(len(speech))

    # Wrap single file_num scalar in a list
    if not isinstance(file_num, list):
        file_num = [file_num]

    success = np.zeros(len(speech))

    # Pad speech to minimum length
    speech = [_padSpeech(s) for s in speech]

    for k in range(len(speech)):
        # Check for empty speech vector
        if np.size(speech[k]) == 0 or math.isnan(file_num[k]):
            success[k] = np.nan

        else:
            # Check for speech using autocorrelation
            # If the signals are periodic (speech), there will be anticorrelation
            # If the signals are noise, there will be no anticorrelation
            # NaN is returned from xcorr if the autocorrelation at lag zero is 0 due to normalization

            xcm = np.min(scipy.signal.correlate(speech[k], speech[k], mode="full") / np.inner(speech[k], speech[k]))

            if xcm > -0.1 or math.isnan(xcm):
                # Speech not detected, skip the algorithm
                success[k] = 0

                if verbose == True:
                    msg = f"In clip #{k}, speech not detected"
                    warn(msg)
            else:
                if verbose == True:
                    msg = f"Working on clip {k} of {len(speech)}"
                    print(msg, "\n")

                C = np.zeros((215, 6))

                # Create a time-frequency (TF) representation and apply Stevens' Law
                X = np.abs(_T_to_TF(speech[k])) ** 0.6

                # correct_word is a pointer that indicates which of the 6 words in the list was spoken in the .wav file
                # This is known in advance from file_num
                # As file_num runs from 1 to 1200, correct word runs from 1 to 6, 200 times
                correct_word = (file_num[k] - 1) % 6

                # first_word is a pointer to first of the six words in the list associated with the present speech file
                # As file_num runs from 1 to 1200, first_word is 1 1 1 1 1 1 7 7 7 7 7 7 ...1195 1195 1195 1195 1195 1195
                first_word = 6 * (math.floor((file_num[k] - 1) / 6) + 1) - 5

                # Compare the computed TF representation for the input .wav file with the TF templates for the 6 candidate words
                for word in range(6):
                    # Find number of columns (time samples) in template
                    ncols = _loader.templates[0, (first_word - 1 + word)].shape[1]

                    # Perform a correlation using a group of rows to find best time alignment between X and template
                    shift = _group_corr(
                        X[_ALIGN_BINS, :], _loader.templates[0, (first_word - 1 + word)][_ALIGN_BINS, :]
                    )

                    # Extract and normalize the best aligned portion of X
                    temp = X[:, (shift + 1) : (shift + ncols + 1)]
                    XX = _TFnorm(temp)

                    # Find the correlation between XX and template, one result per FFT bin
                    C[:, word] = np.sum(np.multiply(XX, _loader.templates[0, (first_word - 1 + word)]), axis=1)

                binsPerBand_tiled = binsPerBand
                binsPerBand_tiled = np.matlib.repmat(binsPerBand_tiled, 1, 6)

                # Aggregate correlation values across each AI band
                C = np.true_divide((AI @ C), binsPerBand_tiled)
                C = np.maximum(C, 0)  # clamp
                C = np.sort(C, axis=0)

                SAC = np.flip(C, axis=0)

                # For each of the 6 word options, sort the 21 AI band correlations from largest to smallest
                SAC = SAC[0:16, :]

                # Consider only the 16 largest correlations for each word
                loc = np.nanargmax(SAC, axis=1)

                # Find which word has largest correlation in each of these 16 cases
                success[k] = np.mean(loc == correct_word)

    # Average over files and correct for guessing
    phi_hat = guess_correction(np.mean(success))

    return phi_hat, success


def _padSpeech(s):
    """
    Purpose
    -------
    Pads speech vector to a minimum length allowable length of 42000

    Parameters
    ----------
        s : TYPE
            Original speech vector

    Returns
    -------
        s : TYPE
            Padded speech vector
    """

    # Minimum speech vector length
    minLen = 42000

    # Get length of speech vector
    l = s.size

    if l < minLen:
        # Fill in zeros at the end
        size_of_S = minLen - l

        S = np.zeros(size_of_S)
        s = np.concatenate((s, S), axis=None)

    else:
        pass

    return s


def _T_to_TF(x):
    """
    Purpose
    -------
    Generates a time-frequency representation for x using
    a length 512 periodic Hanning window with 75% window overlap and FFT.
    Zero padding is used if necessary to create samples for final full window.
    Window length must be evenly divisible by 4.

    Parameters
    ----------
        x : numpy vector
            Speech vector. Must be a column vector.
    Returns
    -------
        X : numpy array
            Time-frequency representation. First 215 values.
    """
    m = x.size
    n = 512
    nframes = math.ceil((m - n) / (n / 4)) + 1
    newm = int((nframes - 1) * (n / 4) + n)

    x = np.concatenate((x, np.zeros((newm - m))))
    X = np.zeros((n, nframes))

    # TODO: Denominator does not follow the definition of Hanning window. Ask Steve Voran about this (should be 511).
    # Generate periodic Hanning window
    win = 0.5 - 0.5 * np.cos((np.pi * 2) * (np.arange(0, 512).T / 512))
    # win_old = np.multiply(0.5, np.subtract(1, np.cos(np.multiply((np.conjugate(np.arange(0, 512)).T / 512), (math.pi * 2)))))

    for i in range(nframes):
        start = int(((i) * (n / 4)))
        X[:, i] = np.multiply(x[start : (start + n)], win)

    X = np.fft.fft(X, axis=0)
    X = X[0:215, :]

    return X


def _TFnorm(X):
    """
    Purpose
    -------
    Removes the mean of every row of the time-frequency representation
    and scales each row so the sum of squares is equal to 1

    Parameters
    ----------
        X : numpy array
            Original time-frequency representation.

    Returns
    -------
        Y : numpy array
            Normalized time-frequency representation.
    """

    n = X.shape[1]

    div = np.true_divide(np.sum(X, axis=1, keepdims=True), n)
    div = np.reshape(div, (div.shape[0], 1))

    X = np.subtract(X, (div @ np.ones((1, n))))

    temp = np.sqrt(np.sum((X ** 2), axis=1, keepdims=True))
    temp = np.reshape(temp, (temp.shape[0], 1))
    temp = np.matlib.repmat(temp, 1, n)

    Y = np.true_divide(X, temp)

    return Y


def _group_corr(X, R):
    """
    Purpose
    -------
    Uses all rows of X and R together in a cross-correlation.
    Evaluates all possible alignments of R with X.

    Parameters
    ----------
        X : numpy array
            Time-frequency representation.
            Has no fewer columns than R.
            Has same number of rows as R.

        R : numpy array
            Has same number of rows as X.
            Can assume R is already normalized for
            zero mean in each row and each row
            has a sum of squares equal to 1.


    Returns
    -------
        shift : int
                Shift that maximizes the correlation value.
                If R has q columns, then a shift value s
                means that R is best aligned with X(:,s+1:s+q).
    """

    n = X.shape[1]
    q = R.shape[1]

    nshifts = n - q + 1

    C = np.zeros((nshifts, 1))

    for i in range(nshifts):
        T = X[:, i : (i + q)]
        if i == 0:
            T_sum = np.sum(T, axis=1)
        else:
            T_sum = T_sum - X[:, i - 1] + X[:, i + q - 1]

        temp = np.true_divide(T_sum, q)

        T = T - temp[:, None]

        kk = np.sqrt(np.sum(np.power(T, 2), axis=1, keepdims=False))

        T = np.true_divide(T, kk[:, None])

        C[i] = np.sum(np.multiply(T, R), keepdims=True)

    shift = np.nanargmax(C)
    shift = shift - 1

    return shift
