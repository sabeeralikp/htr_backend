import json
from htr_api.utils import (
    autoSegmentation,
    convertPDFtoDOCX,
    extract_text,
    extract_text_from_images,
    pdf_to_images,
    saveImageFromInMemoryImage,
    saveImageFromPDF,
    thresholdValue,
)
from .serializers import (
    AutoSegmentSerializer,
    ExportPDFSerializer,
    HTRSerializer,
    ImageCordinateSerializer,
    SaveDataSerializer,
    ThresholdSerializer,
    UploadHTRSerializer,
)
from htr.models import HTR, ImageCordinate, UploadHTR
from rest_framework import generics, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

""" Concrete View Classes
#CreateAPIView
Used for create-only endpoints.
#ListAPIView
Used for read-only endpoints to represent a collection of model instances.
#RetrieveAPIView
Used for read-only endpoints to represent a single model instance.
#DestroyAPIView
Used for delete-only endpoints for a single model instance.
#UpdateAPIView
Used for update-only endpoints for a single model instance.
##ListCreateAPIView
Used for read-write endpoints to represent a collection of model instances.
RetrieveUpdateAPIView
Used for read or update endpoints to represent a single model instance.
#RetrieveDestroyAPIView
Used for read or delete endpoints to represent a single model instance.
#RetrieveUpdateDestroyAPIView
Used for read-write-delete endpoints to represent a single model instance.
"""


class HTRList(generics.ListCreateAPIView):
    queryset = HTR.objects.all()
    serializer_class = HTRSerializer


class HTRDetail(generics.RetrieveDestroyAPIView):
    queryset = HTR.objects.all()
    serializer_class = HTRSerializer


class PDF2IMG(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        print(request.data)
        print(request.data["file"].content_type.split("/")[-1])
        if request.data["file"] != None:
            request.data["file_type"] = request.data["file"].content_type
            if request.data["file"].content_type == "application/pdf":
                images = pdf_to_images(request.data["file"].read())
                request.data["number_of_pages"] = len(images)
                saveImageFromPDF(images, request.data["filename"])
            else:
                request.data["number_of_pages"] = 1
                saveImageFromInMemoryImage(
                    request.data["file"].read(),
                    request.data["filename"],
                    request.data["file"].content_type.split("/")[-1],
                )
        serializer = UploadHTRSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class Threshold(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, format=None):
        print(request.data)
        uploadHTR = UploadHTR.objects.filter(id=request.data["upload_htr"])[0]
        print(request.data["threshold"])
        print(uploadHTR.file_type.split("/")[-1])
        cordinates = thresholdValue(
            int(request.data["threshold"]),
            int(request.data["dilate_x"]),
            int(request.data["dilate_y"]),
            uploadHTR.filename,
            uploadHTR.number_of_pages,
            int(request.data["upload_htr"]),
            uploadHTR.file_type.split("/")[-1],
        )
        serializer = ThresholdSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(cordinates, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class AutoSegment(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, format=None):
        print(request.data)
        uploadHTR = UploadHTR.objects.filter(id=request.data["upload_htr"])[0]
        cordinates = autoSegmentation(
            uploadHTR.filename,
            uploadHTR.number_of_pages,
            int(request.data["upload_htr"]),
            uploadHTR.file_type.split("/")[-1],
        )
        serializer = AutoSegmentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(cordinates, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class Extract(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, format=None):
        print(request.data)
        uploadHTR = UploadHTR.objects.filter(id=request.data["upload_htr"])[0]
        extracted_text = extract_text_from_images(
            json.loads(request.data["cordinates"]),
            uploadHTR.filename,
            uploadHTR.file_type.split("/")[-1],
        )
        serializer = ImageCordinateSerializer(
            data=json.loads(request.data["cordinates"]), many=True
        )
        print(extracted_text)
        print(json.loads(request.data["cordinates"]))
        if serializer.is_valid():
            serializer.save()
            return Response(extracted_text + serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class SaveData(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, format=None):
        print(request.data)
        print(json.loads(request.data["datas"]))
        serializer = SaveDataSerializer(
            data=json.loads(request.data["datas"]), many=True
        )
        if serializer.is_valid():
            print(serializer.data)
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class ExportToDOCX(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, format=None):
        print(request.data)
        if request.data["file"] != None:
            if request.data["file"].content_type == "application/pdf":
                print(request.data["file"].read())
        serializer = ExportPDFSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            print(serializer.data["file"])
            convertPDFtoDOCX(serializer.data["file"])
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class CreateHTR(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        print(request.data)
        if request.data["file"] != None:
            if request.data["file"].content_type == "application/pdf":
                extracted_text = extract_text(
                    request.data["file"],
                    int(request.data["threshold_value"]),
                    int(request.data["dilate_x_value"]),
                    int(request.data["dilate_y_value"]),
                )
                print(extracted_text)
                request.data["extracted_text"] = extracted_text
        serializer = HTRSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)
