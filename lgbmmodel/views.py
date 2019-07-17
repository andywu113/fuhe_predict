# Create your views here.
import django
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
from lgbmmodel import models
# from datetime import datetime, timedelta
# from django.db import connection
from django.http import HttpResponse
import json
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import lightgbm as lgbm
import datetime
import warnings
from lgbmmodel.libs.get_data_fangrong import getData
from lgbmmodel.libs.create_features import GetFeatures
from lgbmmodel.libs.weatherServer import weather_info
from lgbmmodel.libs.other_models import get_data,get_features,next_node_predict,inner_former_prediction
from lgbmmodel.libs.xgb_model import XGBModel
from lgbmmodel.libs.lgbm_model import LGBMODEL
warnings.filterwarnings("ignore")
from celery_tasks.fuhe.tasks import origin_lgbmodel_train,incre_predict
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fuhe_predict.settings")
def weatherInfo(request):
    key = settings.WEATHER["WEATHER_KEY"]
    secret = settings.WEATHER["WEATHER_SECRET"]
    length = request.GET["length"]
    length = int(length)
    longitude = request.GET["longitude"]
    latitude = request.GET["latitude"]
    startTime = request.GET["startTime"]
    enos_ws = weather_info(key,secret)
    weather_df = enos_ws.get_weather_info(length,longitude,latitude,startTime)

    # info={"username":u,"sex":e,"email":e}
    # models.UserInfor.objects.create(**info)

    return JsonResponse(weather_df.to_json(orient='columns'), safe=False)

@csrf_exempt
def model_features_score(request):
    """
    :param request:
    :return:features's names and their scores
    """
    number = request.GET["number"]
    number = int(number)
    dataset = get_data(number)
    new_features = get_features(dataset)
    model = LGBMODEL(new_features,dataset)
    importance = model.features_score()
    return JsonResponse(json.dumps(importance),safe=False)

#包有问题
@csrf_exempt
def model_optimize_parameters(request):
    """
    :param request:
    :return: best params
    """
    number = request.GET["number"]
    number = int(number)
    dataset = get_data(number)
    new_features = get_features(dataset)
    model = LGBMODEL(new_features,dataset)
    best_params = model.optimize_parameters()
    return JsonResponse(json.dumps(best_params),safe=False)


@csrf_exempt
def origion_model_train(request):
    number = request.GET["number"]
    preNum = request.GET["predict_number"]
    number=4
    preNum = 96
    number = int(number)
    preNum = int(preNum)
    origin_lgbmodel_train.delay(number,preNum)
    return HttpResponse("original model saved!")


@csrf_exempt
def incremental_predict(request):
    """
    :param number: company_number
    :param predict_number: predict_number
    :return : prediction series
    """
    number = request.GET["number"]
    preNum = request.GET["predict_number"]
    number = int(number)
    preNum = int(preNum)
    response = incre_predict.delay(number,preNum)
    response.get()
    return JsonResponse(response.get(),safe=False)


# celery -A celery_tasks worker -l info -P eventlet