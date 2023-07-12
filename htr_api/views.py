"""
This module contains views and functionalities for handling HTR (Handwritten Text Recognition) related operations.

The module includes the following classes:
- HTRList: A view for listing and creating HTR objects.
- HTRDetail: A view for retrieving and deleting a specific HTR object.
- PDF2IMG: A view for converting a PDF file to images.
- Threshold: A view for applying a threshold to images and obtaining coordinates.
- AutoSegment: A view for automatically segmenting text regions in images.
- Extract: A view for extracting text from images using provided coordinates.
- SaveData: A view for saving extracted data.
- ExportToDOCX: A view for exporting extracted data to DOCX format.
- CreateHTR: A view for creating an HTR object and performing text extraction.

The module also includes various helper functions imported from other modules for performing PDF to image conversion,
image processing, text extraction, and data serialization.

Note: Adjust the type hints and descriptions as needed based on the actual functionalities
    and data types used in the module.
"""
import json
from htr_api.constant import PDF_FILE_PATH
from htr_api.utils import (
    auto_segmentation,
    convert_pdf_to_docx,
    extract_text,
    extract_text_from_images,
    pdf_to_images,
    save_image_from_in_memory_image,
    save_image_from_pdf,
    threshold_value,
)
from .serializers import (
    AutoSegmentSerializer,
    ExportPDFSerializer,
    FeedBackSerializer,
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
    """
    A view for listing and creating HTR (Handwritten Text Recognition) objects.

    Inherits from `generics.ListCreateAPIView` class provided by the Django REST framework.

    Attributes:
        queryset (QuerySet): The queryset of HTR objects to be listed.
        serializer_class (Serializer): The serializer class used for serializing/deserializing HTR objects.

    Usage:
        This class can be used to retrieve a list of existing HTR objects or create a new HTR object.

    Example:
        To retrieve a list of HTR objects:
        ```
        GET /htr/
        ```

        To create a new HTR object:
        ```
        POST /htr/
        Body:
        {
            // HTR object data
        }
        ```

    Note: Adjust the attribute descriptions based on the actual attributes used in your implementation.
    """

    queryset = HTR.objects.all()
    serializer_class = HTRSerializer


class HTRDetail(generics.RetrieveDestroyAPIView):
    """
    A view for retrieving and deleting a specific HTR (Handwritten Text Recognition) object.

    Inherits from `generics.RetrieveDestroyAPIView` class provided by the Django REST framework.

    Attributes:
        queryset (QuerySet): The queryset of HTR objects to be retrieved or deleted.
        serializer_class (Serializer): The serializer class used for serializing/deserializing HTR objects.

    Usage:
        This class can be used to retrieve the details of a specific HTR object or delete it.

    Example:
        To retrieve the details of a specific HTR object:
        ```
        GET /htr/{id}/
        ```

        To delete a specific HTR object:
        ```
        DELETE /htr/{id}/
        ```

    Note: Adjust the attribute descriptions based on the actual attributes used in your implementation.
    """

    queryset = HTR.objects.all()
    serializer_class = HTRSerializer


class PDF2IMG(APIView):
    """
    A view for converting a PDF file to a series of images.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.
        parser_classes (list): A list of parser classes used for parsing the request data.

    Usage:
        This class can be used to convert a PDF file to images by sending a POST request.

    Example:
        To convert a PDF file to images:
        ```
        POST /pdf2img/
        Request Body:
        {
            "file": <PDF file>,
            "filename": <name of the file>,
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request: any, format: any = None) -> any:
        """
        Convert a PDF file to images.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the serialized data.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To convert a PDF file to images:
            ```
            POST /pdf2img/
            Request Body:
            {
                "file": <PDF file>,
                "filename": <name of the file>,
            }
            ```
        """
        print(request.data)
        print(request.data["file"].content_type.split("/")[-1])
        if request.data["file"] != None:
            request.data["file_type"] = request.data["file"].content_type
            if request.data["file"].content_type == PDF_FILE_PATH:
                images = pdf_to_images(request.data["file"].read())
                request.data["number_of_pages"] = len(images)
                save_image_from_pdf(images, request.data["filename"])
            else:
                request.data["number_of_pages"] = 1
                save_image_from_in_memory_image(
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
    """
    A view for applying thresholding to an image.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.

    Usage:
        This class can be used to apply thresholding to an image by sending a POST request.

    Example:
        To apply thresholding:
        ```
        POST /threshold/
        Request Body:
        {
            "upload_htr": <upload_htr_id>,
            "threshold": <threshold_value>,
            "dilate_x": <dilate_x_value>,
            "dilate_y": <dilate_y_value>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: any, format: any = None) -> any:
        """
        Apply thresholding to an image.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the result coordinates.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To apply thresholding:
            ```
            POST /threshold/
            Request Body:
            {
                "upload_htr": <upload_htr_id>,
                "threshold": <threshold_value>,
                "dilate_x": <dilate_x_value>,
                "dilate_y": <dilate_y_value>
            }
            ```
        """
        print(request.data)
        upload_htr = UploadHTR.objects.filter(id=request.data["upload_htr"])[0]
        print(request.data["threshold"])
        print(upload_htr.file_type.split("/")[-1])
        cordinates = threshold_value(
            int(request.data["threshold"]),
            int(request.data["dilate_x"]),
            int(request.data["dilate_y"]),
            upload_htr.filename,
            upload_htr.number_of_pages,
            int(request.data["upload_htr"]),
            upload_htr.file_type.split("/")[-1],
        )
        serializer = ThresholdSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(cordinates, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class AutoSegment(APIView):
    """
    A view for automatic image segmentation.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.

    Usage:
        This class can be used to perform automatic image segmentation by sending a POST request.

    Example:
        To perform automatic image segmentation:
        ```
        POST /auto-segment/
        Request Body:
        {
            "upload_htr": <upload_htr_id>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: any, format: any = None) -> any:
        """
        Perform automatic image segmentation.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the segmented coordinates.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To perform automatic image segmentation:
            ```
            POST /auto-segment/
            Request Body:
            {
                "upload_htr": <upload_htr_id>
            }
            ```
        """
        print(request.data)
        upload_htr = UploadHTR.objects.filter(id=request.data["upload_htr"])[0]
        cordinates = auto_segmentation(
            upload_htr.filename,
            upload_htr.number_of_pages,
            int(request.data["upload_htr"]),
            upload_htr.file_type.split("/")[-1],
        )
        serializer = AutoSegmentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(cordinates, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class Extract(APIView):
    """
    A view for extracting text from images.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.

    Usage:
        This class can be used to extract text from images by sending a POST request.

    Example:
        To extract text from images:
        ```
        POST /extract/
        Request Body:
        {
            "upload_htr": <upload_htr_id>,
            "cordinates": <list_of_coordinates>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: any, format: any = None) -> any:
        """
        Extract text from images.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the extracted text and serialized coordinates.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To extract text from images:
            ```
            POST /extract/
            Request Body:
            {
                "upload_htr": <upload_htr_id>,
                "cordinates": <list_of_coordinates>
            }
            ```
        """
        print(request.data)
        upload_htr = UploadHTR.objects.filter(id=request.data["upload_htr"])[0]
        extracted_text = extract_text_from_images(
            json.loads(request.data["cordinates"]),
            upload_htr.filename,
            upload_htr.file_type.split("/")[-1],
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
    """
    A view for saving data.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.

    Usage:
        This class can be used to save data by sending a POST request.

    Example:
        To save data:
        ```
        POST /save-data/
        Request Body:
        {
            "datas": <list_of_data>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: any, format: any = None) -> any:
        """
        Save data.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the serialized saved data.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To save data:
            ```
            POST /save-data/
            Request Body:
            {
                "datas": <list_of_data>
            }
            ```
        """
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
    """
    A view for exporting to DOCX format.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.

    Usage:
        This class can be used to export data to DOCX format by sending a POST request.

    Example:
        To export data to DOCX:
        ```
        POST /export-to-docx/
        Request Body:
        {
            "file": <file_object>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: any, format: any = None) -> any:
        """
        Export data to DOCX format.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the exported DOCX file.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To export data to DOCX:
            ```
            POST /export-to-docx/
            Request Body:
            {
                "file": <file_object>
            }
            ```
        """

        print(request.data)
        if request.data["file"] != None:
            if request.data["file"].content_type == PDF_FILE_PATH:
                print(request.data["file"].read())
        serializer = ExportPDFSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            print(serializer.data["file"])
            convert_pdf_to_docx(serializer.data["file"])
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class FeedBackView(APIView):
    """
    A view for exporting to DOCX format.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.

    Usage:
        This class can be used to gather feedback from by sending a POST request.

    Example:
        To gather feedback:
        ```
        POST /feedback/
        Request Body:
        {
            "raiting": <float>
            "remarks": <string>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: any, format: any = None) -> any:
        print(request.data)
        serializer = FeedBackSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)


class CreateHTR(APIView):
    """
    A view for creating HTR (Handwritten Text Recognition) data.

    Inherits from `APIView` class provided by the Django REST framework.

    Attributes:
        permission_classes (list): A list of permission classes applied to the view.
        parser_classes (list): A list of parser classes for parsing the request.

    Usage:
        This class can be used to create HTR data by sending a POST request.

    Example:
        To create HTR data:
        ```
        POST /create-htr/
        Request Body:
        {
            "file": <file_object>,
            "threshold_value": <integer>,
            "dilate_x_value": <integer>,
            "dilate_y_value": <integer>
        }
        ```

    Note: Adjust the attribute descriptions and example based on the actual implementation and requirements.
    """

    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request: any, format: any = None) -> any:
        """
        Create HTR data.

        Args:
            request (Request): The HTTP request object.
            format (str): The format of the response.

        Returns:
            Response: The response object containing the created HTR data.

        Raises:
            HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example:
            To create HTR data:
            ```
            POST /create-htr/
            Request Body:
            {
                "file": <file_object>,
                "threshold_value": <integer>,
                "dilate_x_value": <integer>,
                "dilate_y_value": <integer>
            }
            ```
        """
        print(request.data)
        if request.data["file"] != None:
            if request.data["file"].content_type == PDF_FILE_PATH:
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
