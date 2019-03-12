
from time import time
from subprocess import Popen, PIPE
from pathlib import Path

import joblib


def time2seed():
    """
    Function to seed a random number generator using the current
    time in milliseconds.
    :return:
    """
    seed = int(round(time.time() * 1000))
    return seed


def save_obj(obj, filepath, **kwargs):
    settings = {
        "compress": ("gzip", 6)
    }
    settings.update(kwargs)
    joblib.dump(obj, filepath, **kwargs)


def xyz2sdf(xyz_file):
    filepath = Path(xyz_file)
    fileout = Path.with_suffix(".sdf")
    convert = [
        "obabel",
        "-ixyz",
        xyz_file,
        "-osdf"
    ]
    # Call obabel to convert the file and spit out into a
    # sdf file
    with Popen(convert, stdout=fileout) as process:
        process.wait()


def babel_convert(infile, outform):
    """
    Convert between file formats
    :param infile:
    :param outform:
    :return:
    """
    filepath = Path(infile)
    fileout = Path.with_suffix(".{}".format(outform))
    convert = [
        "obabel",
        "-i{}".format(filepath.suffix)
        infile,
        "-o{}".format(outform)
    ]
    with Popen(convert, stdout=fileout) as process:
        process.wait()

def smi2xyz()
