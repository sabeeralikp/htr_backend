from django.urls import path
from django.views.generic import TemplateView

app_name = "htr"

urlpatterns = [
    path("", TemplateView.as_view(template_name="htr/index.html")),
]
