from __future__ import absolute_import, unicode_literals
import os
from celery import Celery


app = Celery('fuhe_predict',
             broker='amqp://guest:guest@localhost:5672//',
             backend='amqp://guest:guest@localhost:5672//',
             include=['celery_tasks.fuhe.tasks',

                      ])

app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()

#rabbitma开启插件： rabbitmq-plugins.bat enable rabbitmq_management
#rabbitmq开启服务：rabbitmq-server
#登录：localhost:15672