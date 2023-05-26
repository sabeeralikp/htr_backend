from rest_framework import serializers
from htr.models import (
    HTR,
    AutoSegmentValue,
    ExportPDF,
    ImageCordinate,
    SaveData,
    ThresholdValue,
    UploadHTR,
)


class HTRSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "filename",
            "file",
            "extracted_text",
            "threshold_value",
            "dilate_x_value",
            "dilate_y_value",
            "uploaded_by",
            "uploaded_on",
        )
        model = HTR


class UploadHTRSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "filename",
            "file",
            "file_type",
            "number_of_pages",
            "uploaded_by",
            "uploaded_on",
        )
        model = UploadHTR


class ExportPDFSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "file",
        )
        model = ExportPDF


class ThresholdSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "threshold",
            "dilate_x",
            "dilate_y",
            "upload_htr",
            "created_on",
            "created_by",
        )
        model = ThresholdValue


class AutoSegmentSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "upload_htr",
            "created_on",
            "created_by",
        )
        model = AutoSegmentValue


class ImageCordinateListSerialaizer(serializers.ListSerializer):
    def create(self, validated_data):
        image_cordinates = [ImageCordinate(**item) for item in validated_data]
        return ImageCordinate.objects.bulk_create(image_cordinates)


class ImageCordinateSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "x",
            "y",
            "w",
            "h",
            "p",
            "img_h",
            "img_w",
            "upload_htr",
            "created_on",
            "created_by",
        )
        list_serializer_class = ImageCordinateListSerialaizer
        model = ImageCordinate


class SaveDataListSerialaizer(serializers.ListSerializer):
    def create(self, validated_data):
        save_data = [SaveData(**item) for item in validated_data]
        return SaveData.objects.bulk_create(save_data)


class SaveDataSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            "id",
            "image_cordinate",
            "annotated_text",
            "created_on",
            "created_by",
        )
        list_serializer_class = SaveDataListSerialaizer
        model = SaveData
