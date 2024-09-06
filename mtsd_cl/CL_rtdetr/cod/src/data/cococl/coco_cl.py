
import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register

from .coco_cache import CocoCache
from .cl_utils import data_setting

__all__ = ["CocoDetectionCL"]


@register
class CocoDetectionCL(CocoCache):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode,
        task_idx,
        data_ratio,
        buffer_mode,
        buffer_rate,
        remap_mscoco_category=False,
        img_ids=None,
    ):
        self.task_idx = task_idx
        self.data_ratio = data_ratio
        divided_classes = data_setting(data_ratio)
        class_ids_current = divided_classes[self.task_idx]
        buffer_ids = list(set(list(range(0, 221))) - set(class_ids_current))

        super().__init__(
            img_folder,
            ann_file,
            class_ids=class_ids_current,
            buffer_ids=buffer_ids,
            cache_mode=cache_mode,
            ids_list=img_ids,
            buffer_rate=buffer_rate,
            buffer_mode=buffer_mode,
        )

        cats = {}
        for class_id in class_ids_current:
            try:
                cats[class_id] = self.coco.cats[class_id]
            except KeyError:
                pass
        self.coco.cats = cats

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(CocoDetectionCL, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if "boxes" in target:
            target["boxes"] = datapoints.BoundingBox(
                target["boxes"],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img.size[::-1],
            )

        if "masks" in target:
            target["masks"] = datapoints.Mask(target["masks"])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s


def convert_coco_poly_to_mask(segmentation, height, width):
    masks = []
    for polygons in segmentation:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    0: 'regulatory--one-way-left',
 1: 'regulatory--one-way-right',
 2: 'regulatory--one-way-straight',
 3: 'complementary--distance',
 4: 'warning--railroad-intersection',
 5: 'warning--traffic-merges-right',
 6: 'regulatory--maximum-speed-limit-35',
 7: 'information--dead-end',
 8: 'regulatory--no-bicycles',
 9: 'regulatory--no-overtaking',
 10: 'information--parking',
 11: 'complementary--maximum-speed-limit-70',
 12: 'regulatory--maximum-speed-limit-70',
 13: 'complementary--maximum-speed-limit-30',
 14: 'regulatory--no-straight-through',
 15: 'warning--curve-right',
 16: 'information--end-of-built-up-area',
 17: 'information--end-of-motorway',
 18: 'information--end-of-pedestrians-only',
 19: 'information--end-of-living-street',
 20: 'information--end-of-limited-access-road',
 21: 'warning--divided-highway-ends',
 22: 'warning--road-widens-right',
 23: 'warning--kangaloo-crossing',
 24: 'complementary--except-bicycles',
 25: 'regulatory--no-entry',
 26: 'regulatory--mopeds-and-bicycles-only',
 27: 'regulatory--give-way-to-oncoming-traffic',
 28: 'regulatory--end-of-maximum-speed-limit-70',
 29: 'regulatory--no-right-turn',
 30: 'warning--wild-animals',
 31: 'regulatory--maximum-speed-limit-30',
 32: 'warning--double-curve-first-right',
 33: 'warning--road-widens',
 34: 'regulatory--passing-lane-ahead',
 35: 'warning--bicycles-crossing',
 36: 'regulatory--maximum-speed-limit-45',
 37: 'complementary--maximum-speed-limit-55',
 38: 'warning--pass-left-or-right',
 39: 'regulatory--no-overtaking-by-heavy-goods-vehicles',
 40: 'regulatory--maximum-speed-limit-40',
 41: 'regulatory--maximum-speed-limit-led-60',
 42: 'regulatory--maximum-speed-limit-led-100',
 43: 'regulatory--maximum-speed-limit-led-80',
 44: 'warning--turn-right',
 45: 'regulatory--no-pedestrians',
 46: 'warning--junction-with-a-side-road-acute-right',
 47: 'warning--horizontal-alignment-right',
 48: 'regulatory--maximum-speed-limit-120',
 49: 'warning--y-roads',
 50: 'information--hospital',
 51: 'warning--added-lane-right',
 52: 'information--gas-station',
 53: 'regulatory--maximum-speed-limit-90',
 54: 'regulatory--maximum-speed-limit-50',
 55: 'regulatory--maximum-speed-limit-20',
 56: 'regulatory--maximum-speed-limit-60',
 57: 'regulatory--maximum-speed-limit-110',
 58: 'regulatory--maximum-speed-limit-5',
 59: 'regulatory--maximum-speed-limit-80',
 60: 'regulatory--maximum-speed-limit-25',
 61: 'regulatory--maximum-speed-limit-55',
 62: 'regulatory--maximum-speed-limit-10',
 63: 'regulatory--maximum-speed-limit-15',
 64: 'regulatory--maximum-speed-limit-100',
 65: 'regulatory--maximum-speed-limit-65',
 66: 'warning--height-restriction',
 67: 'warning--road-narrows-right',
 68: 'complementary--maximum-speed-limit-35',
 69: 'warning--road-narrows',
 70: 'warning--road-narrows-left',
 71: 'regulatory--no-stopping',
 72: 'warning--winding-road-first-left',
 73: 'complementary--turn-left',
 74: 'complementary--turn-right',
 75: 'complementary--one-direction-left',
 76: 'warning--winding-road-first-right',
 77: 'warning--trail-crossing',
 78: 'regulatory--go-straight-or-turn-left',
 79: 'regulatory--go-straight-or-turn-right',
 80: 'complementary--buses',
 81: 'regulatory--no-u-turn',
 82: 'warning--road-bump',
 83: 'regulatory--buses-only',
 84: 'regulatory--end-of-priority-road',
 85: 'information--disabled-persons',
 86: 'warning--dual-lanes-right-turn-or-go-straight',
 87: 'warning--other-danger',
 88: 'warning--junction-with-a-side-road-perpendicular-left',
 89: 'regulatory--end-of-bicycles-only',
 90: 'information--highway-interstate-route',
 91: 'warning--railroad-crossing-without-barriers',
 92: 'warning--school-zone',
 93: 'regulatory--no-left-turn',
 94: 'warning--horizontal-alignment-left',
 95: 'regulatory--yield',
 96: 'regulatory--road-closed',
 97: 'regulatory--priority-road',
 98: 'complementary--maximum-speed-limit-25',
 99: 'warning--traffic-merges-left',
 100: 'warning--railroad-crossing',
 101: 'regulatory--no-parking',
 102: 'regulatory--no-heavy-goods-vehicles',
 103: 'regulatory--no-turn-on-red',
 104: 'regulatory--no-parking-or-no-stopping',
 105: 'regulatory--no-vehicles-carrying-dangerous-goods',
 106: 'regulatory--no-motor-vehicle-trailers',
 107: 'regulatory--no-stopping--g15',
 108: 'regulatory--no-motor-vehicles',
 109: 'regulatory--no-motorcycles',
 110: 'regulatory--no-buses',
 111: 'regulatory--no-mopeds-or-bicycles',
 112: 'regulatory--no-motor-vehicles-except-motorcycles',
 113: 'regulatory--no-turns',
 114: 'regulatory--no-hawkers',
 115: 'regulatory--no-heavy-goods-vehicles-or-buses',
 116: 'regulatory--no-pedestrians-or-bicycles',
 117: 'regulatory--turn-right',
 118: 'regulatory--turn-left',
 119: 'regulatory--turn-right-ahead',
 120: 'regulatory--turn-left-ahead',
 121: 'regulatory--turning-vehicles-yield-to-pedestrians',
 122: 'regulatory--triple-lanes-turn-left-center-lane',
 123: 'information--pedestrians-crossing',
 124: 'complementary--maximum-speed-limit-45',
 125: 'complementary--one-direction-right',
 126: 'warning--double-curve-first-left',
 127: 'warning--double-turn-first-right',
 128: 'warning--double-reverse-curve-right',
 129: 'warning--pedestrians-crossing',
 130: 'information--bike-route',
 131: 'complementary--maximum-speed-limit-20',
 132: 'complementary--maximum-speed-limit-75',
 133: 'complementary--maximum-speed-limit-40',
 134: 'complementary--maximum-speed-limit-15',
 135: 'complementary--maximum-speed-limit-50',
 136: 'complementary--trucks',
 137: 'regulatory--keep-left',
 138: 'complementary--both-directions',
 139: 'information--children',
 140: 'complementary--go-left',
 141: 'complementary--go-right',
 142: 'regulatory--stop-here-on-red-or-flashing-light',
 143: 'regulatory--bicycles-only',
 144: 'complementary--chevron-left',
 145: 'regulatory--dual-lanes-go-straight-on-right',
 146: 'information--interstate-route',
 147: 'warning--texts',
 148: 'regulatory--priority-over-oncoming-vehicles',
 149: 'warning--domestic-animals',
 150: 'information--road-bump',
 151: 'regulatory--dual-path-bicycles-and-pedestrians',
 152: 'regulatory--end-of-maximum-speed-limit-30',
 153: 'regulatory--end-of-prohibition',
 154: 'regulatory--end-of-speed-limit-zone',
 155: 'regulatory--end-of-no-parking',
 156: 'regulatory--end-of-buses-only',
 157: 'regulatory--pass-on-either-side',
 158: 'regulatory--u-turn',
 159: 'regulatory--stop',
 160: 'regulatory--reversible-lanes',
 161: 'regulatory--shared-path-pedestrians-and-bicycles',
 162: 'regulatory--shared-path-bicycles-and-pedestrians',
 163: 'warning--children',
 164: 'warning--crossroads',
 165: 'regulatory--left-turn-yield-on-green',
 166: 'warning--curve-left',
 167: 'regulatory--dual-lanes-go-straight-on-left',
 168: 'warning--emergency-vehicles',
 169: 'regulatory--roundabout',
 170: 'warning--flaggers-in-road',
 171: 'warning--turn-left',
 172: 'complementary--keep-right',
 173: 'information--trailer-camping',
 174: 'complementary--keep-left',
 175: 'regulatory--radar-enforced',
 176: 'information--bus-stop',
 177: 'regulatory--road-closed-to-vehicles',
 178: 'information--safety-area',
 179: 'warning--hairpin-curve-right',
 180: 'warning--roadworks',
 181: 'warning--steep-ascent',
 182: 'warning--slippery-road-surface',
 183: 'warning--junction-with-a-side-road-acute-left',
 184: 'warning--traffic-signals',
 185: 'information--food',
 186: 'regulatory--weight-limit',
 187: 'information--airport',
 188: 'regulatory--do-not-block-intersection',
 189: 'regulatory--parking-restrictions',
 190: 'warning--falling-rocks-or-debris-right',
 191: 'regulatory--pedestrians-only',
 192: 'complementary--obstacle-delineator',
 193: 'information--living-street',
 194: 'warning--trucks-crossing',
 195: 'information--telephone',
 196: 'warning--two-way-traffic',
 197: 'warning--roundabout',
 198: 'complementary--trucks-turn-right',
 199: 'information--highway-exit',
 200: 'warning--uneven-road',
 201: 'information--emergency-facility',
 202: 'warning--t-roads',
 203: 'warning--narrow-bridge',
 204: 'complementary--chevron-right-unsure',
 205: 'complementary--chevron-right',
 206: 'regulatory--wrong-way',
 207: 'information--motorway',
 208: 'warning--railroad-crossing-with-barriers',
 209: 'complementary--tow-away-zone',
 210: 'regulatory--lane-control',
 211: 'regulatory--keep-right',
 212: 'warning--stop-ahead',
 213: 'regulatory--stop-signals',
 214: 'regulatory--height-limit',
 215: 'warning--crossroads-with-priority-to-the-right',
 216: 'regulatory--go-straight',
 217: 'information--limited-access-road',
 218: 'warning--junction-with-a-side-road-perpendicular-right',
 219: 'complementary--pass-right',
 220: 'information--tram-bus-stop',
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
