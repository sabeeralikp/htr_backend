"""
Module Name: models.py

This module defines the database models for the project.

Models:
- UploadHTR: Represents an uploaded file with associated metadata.
- ThresholdValue: Represents the threshold value and dilate values for image processing.
- AutoSegmentValue: Represents the auto-segmentation values for image processing.
- ImageCordinate: Represents the coordinates and dimensions of an image region.
- SaveData: Represents annotated text and its associated metadata.
- ExportPDF: Represents an exported PDF file.
- HTR: Represents a file with extracted text and associated metadata.

Note: This module assumes the presence of the following packages:
- `datetime`
- `django`
- `django.contrib.auth.models.User`
- `django.db.models`
- `django.utils.translation.gettext_lazy`
"""
import datetime
from django.contrib.auth.models import User
from django.db import models
from django.utils.translation import gettext_lazy as _


def upload_to(instance: dict[any, any], filename: str) -> str:
    """
    Generate the upload path for a file.

    This function takes an instance and a filename and returns the upload path
    for the file. The generated path follows the format: "documents/{uploaded_by}/{filename}",
    where {uploaded_by} is the username associated with the instance and {filename} is
    the original filename.

    Parameters:
    - instance: An instance representing the object associated with the file upload.
    - filename: The original filename of the uploaded file.

    Returns:
    - A string representing the upload path for the file.

    Example:
    If the instance has an uploaded_by attribute with value "john" and the filename is "example.txt",
    the return value would be "documents/john/example.txt".
    """
    return "documents/{uploaded_by}/{filename}".format(
        uploaded_by=instance.uploaded_by, filename=filename
    )


def upload_to_export_pdf(instance: dict[any, any], filename: str) -> str:
    """
    Generate the upload path for exporting a PDF.

    This function takes an instance and a filename and returns the upload path
    for exporting a PDF. The generated path follows the format: "convertedPDF/{filename}",
    where {filename} is the original filename.

    Parameters:
    - instance: An instance representing the object associated with the PDF export.
    - filename: The original filename of the exported PDF.

    Returns:
    - A string representing the upload path for the exported PDF.

    Example:
    If the filename is "document.pdf", the return value would be "convertedPDF/document.pdf".
    """
    return "convertedPDF/{filename}".format(filename=filename)


class UploadHTR(models.Model):
    """
    Model representing an uploaded file with associated metadata.

    This model stores information about an uploaded file, including its filename,
    the actual file, file type, number of pages, upload timestamp, and the user who
    uploaded it.

    Attributes:
    - filename: A string representing the filename of the uploaded file.
    - file: A FileField representing the uploaded file itself.
    - file_type: A TextField indicating the type of the uploaded file (blank if not specified).
    - number_of_pages: An integer representing the number of pages in the uploaded file (nullable).
    - uploaded_on: A DateTimeField indicating the timestamp of the upload (default: current datetime).
    - uploaded_by: A foreign key to the User model representing the user who uploaded the file (nullable).

    Methods:
    - __str__: Returns a string representation of the UploadHTR instance (the filename).

    Note:
    This class assumes the presence of the following packages and modules:
    - `datetime`
    - `django.db.models.Model`
    - `django.contrib.auth.models.User`
    - `django.db.models.FileField`
    - `django.db.models.TextField`
    - `django.db.models.IntegerField`
    - `django.db.models.DateTimeField`
    - `_`: A translation function (usually imported as `_` from `django.utils.translation.gettext_lazy`).
    - `upload_to`: A function for generating the upload path of the file.
    """

    filename = models.CharField(max_length=250)
    file = models.FileField(_("File"), upload_to=upload_to)
    file_type = models.TextField(blank=True)
    number_of_pages = models.IntegerField(null=True)
    uploaded_on = models.DateTimeField(default=datetime.datetime.now)
    uploaded_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True)

    def __str__(self) -> str:
        return self.filename


class ThresholdValue(models.Model):
    """
    Model representing threshold values for image processing.

    This model stores the threshold value, dilate x-value, dilate y-value, and associated metadata
    for image processing. It is used to define the parameters for thresholding and dilation operations.

    Attributes:
    - threshold: An integer representing the threshold value.
    - dilate_x: An integer representing the x-value for dilation.
    - dilate_y: An integer representing the y-value for dilation.
    - upload_htr: A foreign key to the UploadHTR model representing the associated upload (required).
    - created_on: A DateTimeField indicating the timestamp of creation (default: current datetime).
    - created_by: A foreign key to the User model representing the user who created the threshold value (nullable).

    Methods:
    - __str__: Returns a string representation of the ThresholdValue instance (the threshold value).

    Note:
    This class assumes the presence of the following packages and modules:
    - `datetime`
    - `django.db.models.Model`
    - `django.contrib.auth.models.User`
    - `django.db.models.IntegerField`
    - `django.db.models.ForeignKey`
    - `django.db.models.DateTimeField`
    - `UploadHTR`: The model class representing the associated upload (imported from the appropriate module).
    """

    threshold = models.IntegerField()
    dilate_x = models.IntegerField()
    dilate_y = models.IntegerField()
    upload_htr = models.ForeignKey(UploadHTR, on_delete=models.CASCADE)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self) -> str:
        return self.threshold


class AutoSegmentValue(models.Model):
    """
    Model representing auto-segmentation values for document processing.

    This model stores the auto-segmentation values and associated metadata for document processing.
    It is used to define parameters related to auto-segmentation operations.

    Attributes:
    - upload_htr: A foreign key to the UploadHTR model representing the associated upload (required).
    - created_on: A DateTimeField indicating the timestamp of creation (default: current datetime).
    - created_by: A foreign key to the User model representing the user (nullable).

    Methods:
    - __str__: Returns a string representation of the AutoSegmentValue instance (the associated upload).

    Note:
    This class assumes the presence of the following packages and modules:
    - `datetime`
    - `django.db.models.Model`
    - `django.contrib.auth.models.User`
    - `django.db.models.ForeignKey`
    - `django.db.models.DateTimeField`
    - `UploadHTR`: The model class representing the associated upload (imported from the appropriate module).
    """

    upload_htr = models.ForeignKey(UploadHTR, on_delete=models.CASCADE)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self) -> str:
        return self.upload_htr


class ImageCordinate(models.Model):
    """
    Model representing image coordinates for document processing.

    This model stores the coordinates and associated metadata for image processing of a document.
    It is used to define the coordinates of an image region within the document.

    Attributes:
    - x: An integer representing the x-coordinate of the image region.
    - y: An integer representing the y-coordinate of the image region.
    - w: An integer representing the width of the image region.
    - h: An integer representing the height of the image region.
    - p: An integer representing the page number of the image region.
    - img_h: An integer representing the height of the entire image.
    - img_w: An integer representing the width of the entire image.
    - upload_htr: A foreign key to the UploadHTR model representing the associated upload (required).
    - created_on: A DateTimeField indicating the timestamp of creation (default: current datetime).
    - created_by: A foreign key to the User model representing the user who created the image coordinates (nullable).

    Note:
    This class assumes the presence of the following packages and modules:
    - `django.db.models.Model`
    - `django.contrib.auth.models.User`
    - `django.db.models.IntegerField`
    - `django.db.models.ForeignKey`
    - `django.db.models.DateTimeField`
    - `UploadHTR`: The model class representing the associated upload (imported from the appropriate module).
    """

    x = models.IntegerField()
    y = models.IntegerField()
    w = models.IntegerField()
    h = models.IntegerField()
    p = models.IntegerField()
    img_h = models.IntegerField()
    img_w = models.IntegerField()
    upload_htr = models.ForeignKey(UploadHTR, on_delete=models.CASCADE)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self) -> str:
        return self.x


class SaveData(models.Model):
    """
    Model representing saved data for annotated text.

    This model stores the annotated text data and associated metadata for saved annotations.
    It is used to store information about the annotated text, whether it has been saved, and related details.

    Attributes:
    - image_cordinate: A foreign key to the ImageCordinate model representing the image coordinates (required).
    - annotated_text: A TextField representing the annotated text data.
    - is_saved: A BooleanField indicating whether the annotation has been saved (default: False).
    - created_on: A DateTimeField indicating the timestamp of creation (default: current datetime).
    - created_by: A foreign key to the User model representing the user who created the saved data (nullable).

    Methods:
    - __str__: Returns a string representation of the SaveData instance (the x-coordinate of the image coordinates).

    Note:
    This class assumes the presence of the following packages and modules:
    - `django.db.models.Model`
    - `django.contrib.auth.models.User`
    - `django.db.models.ForeignKey`
    - `django.db.models.TextField`
    - `django.db.models.BooleanField`
    - `django.db.models.DateTimeField`
    - `ImageCordinate`: The model class representing the associated image coordinates (imported from the module).
    """

    image_cordinate = models.ForeignKey(ImageCordinate, on_delete=models.CASCADE)
    annotated_text = models.TextField()
    is_saved = models.BooleanField(default=False)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self) -> str:
        return self.x


class ExportPDF(models.Model):
    """
    Model representing an exported PDF file.

    This model stores information about an exported PDF file, including the file itself and associated metadata.

    Attributes:
    - file: A FileField representing the exported PDF file.

    Methods:
    - __str__: Returns a string representation of the ExportPDF instance (the x-coordinate).

    Note:
    This class assumes the presence of the following packages and modules:
    - `django.db.models.Model`
    - `django.db.models.FileField`
    - `upload_to_export_pdf`: A function defining the upload path for the file field (imported from the module).
    - `_`: A function used for translation of strings (imported from the appropriate module).
    """

    file = models.FileField(_("File"), upload_to=upload_to_export_pdf)

    def __str__(self) -> str:
        return self.x


class HTR(models.Model):
    """
    Model representing an HTR (Handwritten Text Recognition) entity.

    This model stores information related to the Handwritten Text Recognition process, including the input file,
    extracted text, threshold values, and associated metadata.

    Attributes:
    - filename: A CharField representing the name of the file (max length: 250 characters).
    - file: A FileField representing the input file for HTR
    - extracted_text: A TextField representing the extracted text from the file (blank=True).
    - threshold_value: An IntegerField representing the threshold value (nullable).
    - dilate_x_value: An IntegerField representing the dilate_x value (nullable).
    - dilate_y_value: An IntegerField representing the dilate_y value (nullable).
    - uploaded_on: A DateTimeField indicating the timestamp of upload (default: current datetime).
    - uploaded_by: A foreign key to the User model representing the user who uploaded the file (nullable).

    Methods:
    - __str__: Returns a string representation of the HTR instance (the filename).

    Note:
    This class assumes the presence of the following packages and modules:
    - `django.db.models.Model`
    - `django.contrib.auth.models.User`
    - `django.db.models.CharField`
    - `django.db.models.FileField`
    - `django.db.models.TextField`
    - `django.db.models.IntegerField`
    - `django.db.models.DateTimeField`
    - `_`: A function used for translation of strings (imported from the appropriate module).
    - `upload_to`: A function defining the upload path for the file field (imported from the appropriate module).
    """

    filename = models.CharField(max_length=250)
    file = models.FileField(_("File"), upload_to=upload_to)
    extracted_text = models.TextField(blank=True)
    threshold_value = models.IntegerField(null=True)
    dilate_x_value = models.IntegerField(null=True)
    dilate_y_value = models.IntegerField(null=True)
    uploaded_on = models.DateTimeField(default=datetime.datetime.now)
    uploaded_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True)

    def __str__(self) -> str:
        return self.filename
