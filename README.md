# Detector

Предисловие:
нашел библиотеку довольно свежую lightning-flash на основое pytorch-lightning

1. Создать окружение из yaml, используя miniconda/anaconda
2. запустить обучение либо скриптом, либо в юпитер ноутбуке
3. подключить тензорборд командой tensorboard --lightning-logs
4. после обучение модель будет сохранена, также в логах можно найти сохраненные чекпоинты
5. с этим же окружением запустить detector.py и отправить команду 

```[python] import base64
from pathlib import Path

import requests

with open("coco128/images/train2017/000000000626.jpg","rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"session": "UUID", "payload": {'inputs': {'data': imgstr}}}
resp = requests.post("http://127.0.0.1:8000/predict", json=body)
print(resp.json)
```

6. Докер поднимается, но скрипт не хочет отрабатывать, причину пытался выяснить 2 дня, ничего не помогло.


Выбранная метрика Mean Avarage Precision: классическая метрика для такой задачи, 
сразу видно какая точность или полнота нужна(какой порог выставлять), на paperwithcode служит бенчмарком моделей.
