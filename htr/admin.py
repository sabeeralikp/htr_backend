from django.contrib import admin
from . import models


# @admin.site.register(models.HTR)
@admin.register(models.HTR)
class AuthorAdmin(admin.ModelAdmin):
    list_display = ("filename", "uploaded_on", "uploaded_by")
