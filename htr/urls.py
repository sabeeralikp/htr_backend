"""
The `urls` module defines the URL patterns for the 'htr' Django application.

This module is responsible for mapping URLs to corresponding views or template views
within the 'htr' application. It sets up the routing configuration for the application's
web pages.

URL Patterns:
- /: The root URL pattern maps to a TemplateView displaying the 'index.html' template.

Note:
- This module assumes the presence of the following packages and modules:
    - `django.urls.path`: For defining URL patterns.
    - `django.views.generic.TemplateView`: For using a generic class-based view to render templates.

- The 'app_name' variable is defined to specify the application namespace. This can be used to
  distinguish the URLs of this application from others in the project.

- Ensure that the 'index.html' template exists within the 'htr' templates directory.

- Additional URL patterns can be added to this module as needed to handle different pages and views
  within the 'htr' application.

- Make sure to include this module's URL patterns in the project's main 'urls.py' file for them to be recognized.
"""

from django.urls import path
from django.views.generic import TemplateView

app_name = "htr"

urlpatterns = [
    path("", TemplateView.as_view(template_name="htr/index.html")),
    path("privacypolicy", TemplateView.as_view(template_name="htr/privacypolicy.html")),
]
