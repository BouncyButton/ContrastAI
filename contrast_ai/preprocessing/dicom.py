import pydicom


def read_dicom(path):
    """Read a dicom file and return the image as a numpy array."""
    return pydicom.dcmread(path).pixel_array
