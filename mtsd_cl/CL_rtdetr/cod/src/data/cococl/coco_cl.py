
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


mscoco_category2name = {0: 'regulatory--go-straight',
 1: 'regulatory--one-way-left',
 2: 'information--children',
 3: 'regulatory--pass-on-either-side',
 4: 'information--gas-station',
 5: 'information--airport',
 6: 'regulatory--do-not-block-intersection',
 7: 'information--highway-interstate-route',
 8: 'regulatory--no-right-turn',
 9: 'regulatory--maximum-speed-limit-5',
 10: 'regulatory--maximum-speed-limit-100',
 11: 'regulatory--dual-lanes-go-straight-on-right',
 12: 'regulatory--maximum-speed-limit-65',
 13: 'regulatory--maximum-speed-limit-50',
 14: 'regulatory--no-turn-on-red',
 15: 'regulatory--no-vehicles-carrying-dangerous-goods',
 16: 'regulatory--parking-restrictions',
 17: 'regulatory--no-parking',
 18: 'information--trailer-camping',
 19: 'regulatory--maximum-speed-limit-120',
 20: 'regulatory--one-way-right',
 21: 'regulatory--end-of-prohibition',
 22: 'regulatory--no-mopeds-or-bicycles',
 23: 'regulatory--road-closed',
 24: 'regulatory--maximum-speed-limit-70',
 25: 'regulatory--mopeds-and-bicycles-only',
 26: 'regulatory--no-u-turn',
 27: 'regulatory--end-of-priority-road',
 28: 'regulatory--no-motorcycles',
 29: 'regulatory--dual-path-bicycles-and-pedestrians',
 30: 'regulatory--no-motor-vehicle-trailers',
 31: 'information--highway-exit',
 32: 'regulatory--no-heavy-goods-vehicles-or-buses',
 33: 'regulatory--end-of-buses-only',
 34: 'regulatory--no-stopping--g15',
 35: 'regulatory--pedestrians-only',
 36: 'information--motorway',
 37: 'regulatory--maximum-speed-limit-60',
 38: 'regulatory--end-of-no-parking',
 39: 'regulatory--turn-left',
 40: 'information--end-of-built-up-area',
 41: 'information--end-of-limited-access-road',
 42: 'regulatory--keep-left',
 43: 'regulatory--radar-enforced',
 44: 'regulatory--end-of-speed-limit-zone',
 45: 'information--dead-end',
 46: 'regulatory--no-heavy-goods-vehicles',
 47: 'regulatory--shared-path-bicycles-and-pedestrians',
 48: 'regulatory--no-left-turn',
 49: 'regulatory--keep-right',
 50: 'regulatory--shared-path-pedestrians-and-bicycles',
 51: 'regulatory--no-turns',
 52: 'regulatory--turn-left-ahead',
 53: 'information--bus-stop',
 54: 'information--tram-bus-stop',
 55: 'regulatory--passing-lane-ahead',
 56: 'information--bike-route',
 57: 'regulatory--maximum-speed-limit-25',
 58: 'regulatory--no-overtaking-by-heavy-goods-vehicles',
 59: 'regulatory--turn-right-ahead',
 60: 'regulatory--end-of-bicycles-only',
 61: 'regulatory--triple-lanes-turn-left-center-lane',
 62: 'regulatory--turning-vehicles-yield-to-pedestrians',
 63: 'regulatory--maximum-speed-limit-10',
 64: 'regulatory--end-of-maximum-speed-limit-70',
 65: 'regulatory--weight-limit',
 66: 'regulatory--buses-only',
 67: 'regulatory--no-stopping',
 68: 'regulatory--no-motor-vehicles-except-motorcycles',
 69: 'regulatory--maximum-speed-limit-55',
 70: 'regulatory--maximum-speed-limit-35',
 71: 'regulatory--no-pedestrians-or-bicycles',
 72: 'regulatory--one-way-straight',
 73: 'information--hospital',
 74: 'regulatory--maximum-speed-limit-90',
 75: 'regulatory--roundabout',
 76: 'information--living-street',
 77: 'regulatory--end-of-maximum-speed-limit-30',
 78: 'regulatory--go-straight-or-turn-left',
 79: 'information--telephone',
 80: 'information--disabled-persons',
 81: 'regulatory--maximum-speed-limit-20',
 82: 'regulatory--u-turn',
 83: 'regulatory--left-turn-yield-on-green',
 84: 'regulatory--maximum-speed-limit-30',
 85: 'regulatory--no-pedestrians',
 86: 'regulatory--priority-road',
 87: 'information--parking',
 88: 'information--food',
 89: 'regulatory--no-motor-vehicles',
 90: 'regulatory--maximum-speed-limit-45',
 91: 'regulatory--maximum-speed-limit-led-100',
 92: 'regulatory--stop',
 93: 'regulatory--dual-lanes-go-straight-on-left',
 94: 'regulatory--give-way-to-oncoming-traffic',
 95: 'information--end-of-pedestrians-only',
 96: 'regulatory--turn-right',
 97: 'regulatory--stop-signals',
 98: 'information--emergency-facility',
 99: 'information--interstate-route',
 100: 'regulatory--no-entry',
 101: 'regulatory--bicycles-only',
 102: 'information--limited-access-road',
 103: 'regulatory--no-bicycles',
 104: 'regulatory--priority-over-oncoming-vehicles',
 105: 'regulatory--no-hawkers',
 106: 'regulatory--wrong-way',
 107: 'regulatory--maximum-speed-limit-15',
 108: 'information--safety-area',
 109: 'regulatory--go-straight-or-turn-right',
 110: 'regulatory--no-straight-through',
 111: 'regulatory--reversible-lanes',
 112: 'regulatory--road-closed-to-vehicles',
 113: 'regulatory--yield',
 114: 'regulatory--maximum-speed-limit-led-80',
 115: 'information--end-of-motorway',
 116: 'regulatory--lane-control',
 117: 'regulatory--no-parking-or-no-stopping',
 118: 'regulatory--maximum-speed-limit-80',
 119: 'regulatory--no-buses',
 120: 'information--road-bump',
 121: 'regulatory--no-overtaking',
 122: 'regulatory--stop-here-on-red-or-flashing-light',
 123: 'regulatory--maximum-speed-limit-40',
 124: 'information--pedestrians-crossing',
 125: 'regulatory--maximum-speed-limit-led-60',
 126: 'regulatory--maximum-speed-limit-110',
 127: 'information--end-of-living-street',
 128: 'regulatory--height-limit',
 129: 'complementary--maximum-speed-limit-55',
 130: 'complementary--go-left',
 131: 'warning--horizontal-alignment-left',
 132: 'complementary--turn-right',
 133: 'warning--uneven-road',
 134: 'warning--trail-crossing',
 135: 'warning--winding-road-first-left',
 136: 'warning--horizontal-alignment-right',
 137: 'warning--dual-lanes-right-turn-or-go-straight',
 138: 'warning--junction-with-a-side-road-perpendicular-right',
 139: 'warning--children',
 140: 'warning--junction-with-a-side-road-acute-right',
 141: 'warning--junction-with-a-side-road-acute-left',
 142: 'warning--flaggers-in-road',
 143: 'warning--narrow-bridge',
 144: 'complementary--maximum-speed-limit-35',
 145: 'warning--road-narrows-right',
 146: 'warning--wild-animals',
 147: 'warning--pedestrians-crossing',
 148: 'warning--crossroads-with-priority-to-the-right',
 149: 'warning--traffic-merges-right',
 150: 'warning--domestic-animals',
 151: 'complementary--maximum-speed-limit-30',
 152: 'warning--roundabout',
 153: 'complementary--chevron-right-unsure',
 154: 'warning--double-turn-first-right',
 155: 'warning--traffic-merges-left',
 156: 'warning--curve-right',
 157: 'warning--added-lane-right',
 158: 'warning--bicycles-crossing',
 159: 'warning--emergency-vehicles',
 160: 'warning--other-danger',
 161: 'warning--slippery-road-surface',
 162: 'complementary--trucks',
 163: 'complementary--one-direction-left',
 164: 'complementary--distance',
 165: 'complementary--maximum-speed-limit-20',
 166: 'warning--double-curve-first-left',
 167: 'warning--crossroads',
 168: 'complementary--one-direction-right',
 169: 'complementary--pass-right',
 170: 'warning--falling-rocks-or-debris-right',
 171: 'complementary--chevron-right',
 172: 'warning--railroad-crossing',
 173: 'complementary--obstacle-delineator',
 174: 'warning--height-restriction',
 175: 'warning--double-reverse-curve-right',
 176: 'warning--road-widens-right',
 177: 'warning--road-narrows-left',
 178: 'complementary--trucks-turn-right',
 179: 'warning--kangaloo-crossing',
 180: 'complementary--maximum-speed-limit-40',
 181: 'complementary--maximum-speed-limit-45',
 182: 'warning--t-roads',
 183: 'complementary--keep-left',
 184: 'warning--trucks-crossing',
 185: 'warning--winding-road-first-right',
 186: 'complementary--maximum-speed-limit-15',
 187: 'complementary--chevron-left',
 188: 'complementary--maximum-speed-limit-75',
 189: 'complementary--maximum-speed-limit-50',
 190: 'warning--divided-highway-ends',
 191: 'warning--road-narrows',
 192: 'warning--turn-right',
 193: 'warning--y-roads',
 194: 'complementary--turn-left',
 195: 'complementary--go-right',
 196: 'warning--school-zone',
 197: 'warning--roadworks',
 198: 'complementary--except-bicycles',
 199: 'complementary--maximum-speed-limit-25',
 200: 'warning--double-curve-first-right',
 201: 'warning--texts',
 202: 'warning--road-bump',
 203: 'warning--road-widens',
 204: 'warning--two-way-traffic',
 205: 'warning--traffic-signals',
 206: 'warning--turn-left',
 207: 'warning--junction-with-a-side-road-perpendicular-left',
 208: 'complementary--keep-right',
 209: 'complementary--maximum-speed-limit-70',
 210: 'warning--railroad-crossing-with-barriers',
 211: 'warning--railroad-intersection',
 212: 'complementary--tow-away-zone',
 213: 'warning--pass-left-or-right',
 214: 'warning--stop-ahead',
 215: 'complementary--both-directions',
 216: 'warning--curve-left',
 217: 'warning--hairpin-curve-right',
 218: 'warning--railroad-crossing-without-barriers',
 219: 'warning--steep-ascent',
 220: 'complementary--buses'}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
