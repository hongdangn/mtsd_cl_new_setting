import numpy as np


# * Divide classes
def divide_classes_randomly(total_classes, ratios, seed=123):
    np.random.seed(seed)
    np.random.shuffle(total_classes)
    divided_classes = []
    start_idx = 0

    for ratio in ratios:
        end_idx = start_idx + ratio
        divided_classes.append(total_classes[start_idx:end_idx])
        start_idx = end_idx

    return divided_classes


def data_setting(ratio: str, random_setting: bool = False):
    flatten_list = lambda nested_list: [
        item for sublist in nested_list for item in sublist
    ]
    total_classes = list(range(0, 221))
    divided_classes = [
        list(range(0, 129)),  # 45 classes
        list(range(130, 221)),  # 10 classes

    ]

    ratio_to_classes = {
    "12992": [flatten_list(divided_classes[:-1]), divided_classes[-1]]
    }

    divided_classes_detail = ratio_to_classes.get(ratio, total_classes)

    # * Various order testing in CL-DETR
    #if random_setting:
    #    ratios = [150,71]

    #    divided_classes_detail = divide_classes_randomly(total_classes, ratios)

    return divided_classes_detail
