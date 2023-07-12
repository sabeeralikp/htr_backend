"""
URL patterns for the htr_api Django app.

These patterns map the specified URLs to their corresponding views. The `app_name` variable is set
    to "htr_api" to provide a namespace for the URLs.

- /document/post: Maps to the `CreateHTR` view for posting a document.
- /document/postHTR: Maps to the `PDF2IMG` view for processing a document using OCR.
- /document/threshold: Maps to the `Threshold` view for setting the threshold for document processing.
- /document/autoSegment: Maps to the `AutoSegment` view for automatically segmenting a document.
- /document/extract: Maps to the `Extract` view for extracting data from a document.
- /document/saveData: Maps to the `SaveData` view for saving extracted data.
- /document/exportDoc: Maps to the `ExportToDOCX` view for exporting a document as a DOCX file.
"""

from django.urls import path
from .views import (
    PDF2IMG,
    AutoSegment,
    ExportToDOCX,
    Extract,
    CreateHTR,
    FeedBackView,
    SaveData,
    Threshold,
)

app_name = "htr_api"

urlpatterns = [
    path("document/post", CreateHTR.as_view(), name="postdocument"),
    path("document/postHTR", PDF2IMG.as_view(), name="postHTRdocument"),
    path("document/threshold", Threshold.as_view(), name="threshold"),
    path("document/autoSegment", AutoSegment.as_view(), name="autoSegment"),
    path("document/extract", Extract.as_view(), name="extract"),
    path("document/saveData", SaveData.as_view(), name="saveData"),
    path("document/exportDoc", ExportToDOCX.as_view(), name="exportDoc"),
    path("document/feedback", FeedBackView.as_view(), name="feedback"),
]
