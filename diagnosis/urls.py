from django.urls import path
from . import views

urlpatterns = [
    # This connects the empty path '' to your home view
    path('', views.home, name='home'),
]