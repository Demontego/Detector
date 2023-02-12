import flash
from icevision.all import *
from flash.image import ObjectDetectionData, ObjectDetector


datamodule = ObjectDetectionData.from_coco(
    train_folder="coco128/images/train2017/",
    train_ann_file="coco128/annotations/instances_train2017.json",
    val_split=0.1,
    transform_kwargs={"image_size": 512},
    batch_size=64,
)


heads = ObjectDetector.available_heads()
for i in heads:
    print(i)
    backbones = ObjectDetector.available_backbones(i)
    print('BACKBONES', *backbones, sep='\n')
    

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
model = ObjectDetector(
    head='efficientdet',
    backbone='d0',
    num_classes=datamodule.num_classes,
    image_size=512,
    optimizer="Adam",
    learning_rate=1e-4,
    metrics=metrics,
)

trainer = flash.Trainer(max_epochs=1, accelerator='gpu', devices=1, precision=16)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

trainer.save_checkpoint("object_detection_model_1epoch.pt")
