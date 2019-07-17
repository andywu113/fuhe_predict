#coding = "utf-8"
__author :"lokey"

import urllib.request
import urllib.parse
import urllib.error
import simplejson
import hashlib
import itertools
import pandas as pd
from hashlib import sha1
from datetime import datetime

class weather_info(object):
    def __init__(self,weatherServer_appkey,weatherServer_secret):
        self.app_key = weatherServer_appkey
        self.secret = weatherServer_secret
        self.url = "https://ag-cn2.envisioniot.com/en-weather/api/v1/forecast/hourly?sourceAttributes=TMP,PRES,RH,WS,WD,RAINFALL&"
    def get_sign(self,attrs):
        args = ''.join(itertools.chain.from_iterable(sorted(attrs.items())))
        sign = hashlib.sha1((self.app_key + args + self.secret).encode('utf-8')).hexdigest().upper()
        return sign

    def get_weather_info(self,length,longitude,latitude ,startTime ):
        attrs = {
            "length": str(length),
            "longitude": str(longitude),
            "latitude": str(latitude),
            "sourceAttributes": "TMP,PRES,RH,WS,WD,RAINFALL",
            "startTime": str(startTime),
        }
        sign = self.get_sign(attrs)
        print(sign)
        url = self.url + "startTime=%s&length=%d&longitude=%s&latitude=%s&appKey=%s&sign=%s"%(startTime,length,str(longitude),str(latitude),self.app_key,sign)
        print(url)
        request = urllib.request.Request(url)
        request.add_header("Content-Type", "application/x-www-form-urlencoded;charset=utf-8")
        f = urllib.request.urlopen(request)
        rtn_str = f.read().decode('utf-8')
        rtn_dict = simplejson.loads(rtn_str)
        index = pd.date_range(startTime,periods=length,freq="1H")
        df = pd.DataFrame(rtn_dict)
        df.index = index
        return df


#接口使用方式：创建weather_info类，调用get_weather_info函数
##get_weather_info需要传入的参数为：length:startTime之后的多少个点；longitude：经度；latitude：纬度；startTime：起始时间
##返回dataFrame,默认获取对应时间点温度、气压、湿度、风速、风向、降水量数据

#使用方式
# import os
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "你的project.settings")
# def weatherInfo(request):
#     key = settings.WEATHER["WEATHER_KEY"]
#     secret = settings.WEATHER["WEATHER_SECRET"]
#     length = request.GET["length"]
#     length = int(length)
#     longitude = request.GET["longitude"]
#     latitude = request.GET["latitude"]
#     startTime = request.GET["startTime"]
#     enos_ws = weather_info(key,secret)
#     weather_df = enos_ws.get_weather_info(length,longitude,latitude,startTime)
#     return JsonResponse(weather_df.to_json(orient='columns'), safe=False)
