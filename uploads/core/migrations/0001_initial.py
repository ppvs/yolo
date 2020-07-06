# -*- coding: utf-8 -*-
# Generated by Django 1.9.8 on 2016-08-01 08:14
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('conf_threshold', models.CharField(blank=True, max_length=255)),
		('nms_threshold', models.CharField(blank=True, max_length=255)),
		('detection', models.CharField(blank=True, max_length=255)),
                ('document', models.FileField(upload_to=b'')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]