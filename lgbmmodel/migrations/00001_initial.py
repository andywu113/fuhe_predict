# Generated by Django 2.0.4 on 2019-05-15 14:02

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='WeatherInfomation',
            fields=[
                ('temperature', models.CharField(verbose_name="温度", max_length=255, default='')),
                ('pressure', models.CharField(verbose_name="气压", max_length=255, default='')),
                ('humdity', models.CharField(verbose_name="湿度", max_length=255, default='')),
                ('wind_speed', models.CharField(verbose_name="风速", max_length=255, default='')),
                ('wind_direction', models.CharField(verbose_name="风向", max_length=255, default='')),
                ('rainfall', models.CharField(verbose_name="降水量", max_length=255, default='')),
            ],
        )

    ]
