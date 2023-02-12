from flash.image import ObjectDetector
from flash.core.data.io.output import Output
from typing import Any, Dict, List, Optional
import os

from flash.image import ObjectDetector
from flash.core.data.io.input import DataKeys
from flash.core.data.io.output import Output
from flash.core.model import Task


class DetectionLabelsOutput(Output):

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        super().__init__()
        self._labels = labels
        self.threshold = threshold

    @classmethod
    def from_task(cls, task: Task, **kwargs) -> Output:
        return cls(labels=getattr(task, "labels", None))

    def transform(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        if DataKeys.METADATA not in sample:
            raise ValueError(
                "sample requires DataKeys.METADATA to use a FiftyOneDetectionLabelsOutput output."
            )

        height, width = sample[DataKeys.METADATA]["output_size"]
        
        real_height, real_width = sample[DataKeys.METADATA]["size"]

        detections = []

        preds = sample[DataKeys.PREDS]

        for bbox, label, score in zip(preds["bboxes"], preds["labels"], preds["scores"]):
            confidence = score.tolist()

            if self.threshold is not None and confidence < self.threshold:
                continue
            
            xmin, ymin, box_width, box_height = bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]
            box = [
                int((xmin / width * real_width).item()),
                int((ymin / height * real_height).item()),
                int((box_width / width * real_width).item()),
                int((box_height / height * real_height).item()),
            ]

            label = label.item()
            if self._labels is not None:
                label = self._labels[label]
            else:
                label = str(int(label))

            detections.append(
                {
                    "confidence": round(confidence, 4),
                    "label": label,
                    "points": box,
                }
            )
        return detections


if __name__ == "__main__":
    model = ObjectDetector.load_from_checkpoint("object_detection_model_1epoch.pt")
    model.serve(host="0.0.0.0", sanity_check=False, transform_kwargs={"image_size": (512,512)}, output=DetectionLabelsOutput())