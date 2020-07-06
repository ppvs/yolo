from django import forms

from uploads.core.models import Document


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('detection', 'conf_threshold','nms_threshold', 'document', )

