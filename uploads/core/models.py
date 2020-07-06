from __future__ import unicode_literals

from django.db import models


class Document(models.Model):
    nms_threshold = models.CharField(max_length=255, blank=True)
    conf_threshold = models.CharField(max_length=255, blank=True)
    detection = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
