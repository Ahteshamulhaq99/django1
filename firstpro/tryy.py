import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MEDIA_ROOT =  os.path.join(BASE_DIR, 'media')
print(MEDIA_ROOT)