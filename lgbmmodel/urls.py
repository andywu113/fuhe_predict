#coding=utf-8
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dj_configure.settings")
django.setup()

from django.conf.urls import url, include
from .views import *

urlpatterns = [
        url(r'^weatherInfo/$',weatherInfo),
        url(r'^model_features_score/$',model_features_score),
        url(r'^model_optimize_parameters/$',model_optimize_parameters),
        url(r'^incremental_predict/$', incremental_predict),
        url(r'^origion_model_train/$', origion_model_train),

]

