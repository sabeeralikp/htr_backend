import datetime
from django.contrib.auth.models import User
from django.db import models
from django.utils.translation import gettext_lazy as _


def upload_to(instance, filename):
    return "documents/{uploaded_by}/{filename}".format(
        uploaded_by=instance.uploaded_by, filename=filename
    )


def upload_to_export_pdf(instance, filename):
    return "convertedPDF/{filename}".format(filename=filename)


class UploadHTR(models.Model):
    filename = models.CharField(max_length=250)
    file = models.FileField(_("File"), upload_to=upload_to)
    file_type = models.TextField(null=True)
    number_of_pages = models.IntegerField(null=True)
    uploaded_on = models.DateTimeField(default=datetime.datetime.now)
    uploaded_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True)

    def __str__(self):
        return self.filename


class ThresholdValue(models.Model):
    threshold = models.IntegerField()
    dilate_x = models.IntegerField()
    dilate_y = models.IntegerField()
    upload_htr = models.ForeignKey(UploadHTR, on_delete=models.CASCADE)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.threshold


class AutoSegmentValue(models.Model):
    upload_htr = models.ForeignKey(UploadHTR, on_delete=models.CASCADE)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.upload_htr


class ImageCordinate(models.Model):
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


class SaveData(models.Model):
    image_cordinate = models.ForeignKey(ImageCordinate, on_delete=models.CASCADE)
    annotated_text = models.TextField()
    is_saved = models.BooleanField(default=False)
    created_on = models.DateTimeField(default=datetime.datetime.now)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)


class ExportPDF(models.Model):
    file = models.FileField(_("File"), upload_to=upload_to_export_pdf)


class HTR(models.Model):
    filename = models.CharField(max_length=250)
    file = models.FileField(_("File"), upload_to=upload_to)
    extracted_text = models.TextField(null=True)
    threshold_value = models.IntegerField(null=True)
    dilate_x_value = models.IntegerField(null=True)
    dilate_y_value = models.IntegerField(null=True)
    uploaded_on = models.DateTimeField(default=datetime.datetime.now)
    uploaded_by = models.ForeignKey(User, on_delete=models.PROTECT, null=True)

    def __str__(self):
        return self.filename
