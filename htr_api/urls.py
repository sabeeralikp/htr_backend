from django.urls import path
from .views import (
    PDF2IMG,
    AutoSegment,
    ExportToDOCX,
    Extract,
    HTRList,
    HTRDetail,
    CreateHTR,
    SaveData,
    Threshold,
)

app_name = "htr_api"

urlpatterns = [
    # path('<int:pk>/', HTRDetail.as_view(), name='detailcreate'),
    # path('', HTRList.as_view(), name='listcreate'),
    path("document/post", CreateHTR.as_view(), name="postdocument"),
    path("document/postHTR", PDF2IMG.as_view(), name="postHTRdocument"),
    path("document/threshold", Threshold.as_view(), name="threshold"),
    path("document/autoSegment", AutoSegment.as_view(), name="autoSegment"),
    path("document/extract", Extract.as_view(), name="extract"),
    path("document/saveData", SaveData.as_view(), name="saveData"),
    path("document/exportDoc", ExportToDOCX.as_view(), name="exportDoc"),
]
