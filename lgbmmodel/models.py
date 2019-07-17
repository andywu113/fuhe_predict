from django.db import models

from django.db import models
import MySQLdb
from django.contrib.auth.models import User

class WeatherInfomation(models.Model):
    temperature = models.CharField(verbose_name="温度",max_length=255,default='')
    pressure = models.CharField(verbose_name="气压",max_length=255,default='')
    humdity = models.CharField(verbose_name="湿度",max_length=255,default='')
    wind_speed = models.CharField(verbose_name="风速",max_length=255,default='')
    wind_direction = models.CharField(verbose_name="风向",max_length=255,default='')
    rainfall = models.CharField(verbose_name="降水量",max_length=255,default='')
