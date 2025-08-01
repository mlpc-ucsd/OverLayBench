import json
import re
import tqdm
import logging

def _setup_logger(save_dir):
    """Set up the logger for OverLayBench."""
    logger = logging.getLogger("OverLayBench")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        # Console handler
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File handler
        file_handler = logging.FileHandler(save_dir)
        file_handler.setFormatter(stream_formatter)
        logger.addHandler(file_handler)
    return logger

def bbox_prompt(annotations, img, *args, **kwargs):
    categories = list(set(e['category'] for e in annotations['annotations'].values()))
    categories = sorted(categories)
    prompt = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are Qwen-VL, created by Alibaba Cloud. You are a helpful assistant."
            }]
        },
        {
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': f"You are required to detect all the instances of the following categories {categories} in the image."
                        "Response in json format:"
                        "{"
                        "'category_1': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],"
                        "'category_2': [[x1, y1, x2, y2], ...]"
                        "..."
                        "}"
                        "For each category, provide a list of bounding boxes of all its instances in the image."
                        "Each bounding box must correspond to a single, distinct individual object — never a group or collection."
                        "Strictly follow this instruction without exceptions or interpretation."
                        "Strictly follow the format in English, without any irrelevant words."
            },
                {
                    "type": "image",
                    "image": img,
                }],
        }]
    return prompt


def entity_prompt(annotations, img, resolution=1024):
    if resolution == 512:
        for key in annotations['annotations'].keys():
            annotations['annotations'][key]['bbox'] = [int(b / 1024 * 512) for b in
                                                       annotations['annotations'][key]['bbox']]
    prompt = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are Qwen-VL, created by Alibaba Cloud. You are a helpful assistant."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "You are required to answer whether the instances in an image matches the corresponding descriptions, based on its bounding box."
                        "Here are the instance name, the corresponding bbox and the instance description:"
                        f"{annotations['annotations']}"
                        "Please follow these rules:"
                        "Check if the generated instance visually matches its local_prompt description."
                        "If the instance is clearly generated and not corrupted, and its key attributes described in the local_prompt are present, answer 'Yes'."
                        "If the instance is missing, corrupted, or the key attributes are not present, answer 'No'."
                        "Response in the following format:"
                        "```json"
                        "{"
                        '  Instance_name: "Yes"/"No",'
                        "  ..."
                        "}"
                        "```"
                        "Each key must be from the dict of the instance name, the corresponding bbox and the instance description."
                        "Each value must be 'Yes' or 'No'."
                        "If the instance name is not in the image, the answer should be 'No'."
                        "Strictly follow the format in English, without any irrelevant words."
            }, {
                "type": "image",
                "image": img,
            }]
        }]
    return prompt


def relationship_prompt(annotations, img, resolution=1024):
    instances = list(annotations['annotations'].keys())
    bbox = {instance: annotations['annotations'][instance]['bbox'] for instance in instances}
    if resolution == 512:
        for key in bbox.keys():
            bbox[key] = [int(b / 1024 * 512) for b in bbox[key]]
    relations = annotations['relations']
    prompt = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are Qwen-VL, created by Alibaba Cloud. You are a helpful assistant."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "You are required to answer whether the relationship between two instances in an image matches the description."
                        "Here are the instance name and the bbox:"
                        f"{bbox}"
                        "Here is dict of the instance pair and the ground truth relationship descriptions:"
                        f"{relations}"
                        "Please follow these rules:"
                        "For proximity relations like near, beside, close to, next to, if the two instances are generated well (not corrupted or fused into one) and their bounding boxes are close, you can consider the description as matched."
                        "For directional or positional relations like behind, in front of, you must strictly check if the spatial arrangement in the image actually matches the description, because bounding boxes alone are not enough."

                        "Response in the following format:"
                        "```json"
                        "{"
                        '  "(Instance_1, Instance_2)": "Yes"/"No",'
                        "  ..."
                        "}"
                        "```"
                        "Each key must be a tuple from the dict of the instance pair and the ground truth relationship descriptions."
                        "Each value must be 'Yes' or 'No'."
                        "'Yes' means the action/spatial relationship between the two instances matches the description."
                        "You shouldn't pay too much attention on how well the bounding boxes are aligned."
                        "Strictly follow the format in English, without any irrelevant words."
            }, {
                "type": "image",
                "image": img,
            }]
        }]
    return prompt


def extract_and_parse_json(text):
    # Step 1: Extract content inside ```json ... ```
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if not match:
        try:
            result = json.loads(text)
            return result
        except (json.JSONDecodeError, ValueError):
            return {}

    json_str = match.group(1)

    # Step 2: Parse and validate JSON
    try:
        result = json.loads(json_str)
        return result
    except (json.JSONDecodeError, ValueError):
        return {}


def _calculate_success_rate(pred, gt, gt_key='annotations'):
    valid = 0
    total = 0
    for img_id in gt.keys():
        if img_id not in pred.keys():
            total += len(gt[img_id][gt_key])
            continue
        for key in gt[img_id][gt_key]:
            total += 1
            if key in pred[img_id]:
                if (pred[img_id][key]).lower() == 'yes':
                    valid += 1
    return valid / total


def calculate_success_rate(pred, gt, mode):
    if mode == 'Relationship VQA':
        return _calculate_success_rate(pred, gt, gt_key='relations')
    elif mode == 'Entity VQA':
        return _calculate_success_rate(pred, gt, gt_key='annotations')
    else:
        raise ValueError(f"Unknown mode: {mode}")


def resize_bbox(bboxes, resolution):
    resize_ratio = 1024 / resolution
    ret = []
    for bbox in bboxes:
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        valid = True
        for i in range(4):
            if isinstance(bbox[i], int) or isinstance(bbox[i], float):
                bbox[i] = int(bbox[i] * resize_ratio)
            elif isinstance(bbox[i], str):
                bbox[i] = int(float(bbox[i]) * resize_ratio)
            else:
                valid = False
                break
        if valid:
            ret.append(bbox)
    return ret
    # if resolution == 1024:
    #     return bboxes
    # ret = []
    # for bbox in bboxes:
    #     if len(bbox) == 0:
    #         continue
    #     if type(bbox[0]) == int:
    #         ret.append([bbox[0] * 2, bbox[1] * 2, bbox[2] * 2, bbox[3] * 2])
    #     elif type(bbox[0]) == str:
    #         ret.append([int(bbox[0]) * 2, int(bbox[1]) * 2, int(bbox[2]) * 2, int(bbox[3]) * 2])
    #     else:
    #         raise ValueError(f"type of bbox {bbox} is not int or str")
    # return ret


def get_iou(bbox_pred, bbox_gt):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    :param bbox_pred: list of 4 floats [x1, y1, x2, y2] representing the predicted bounding boxes
    :param bbox_gt: list of 4 floats [x1, y1, x2, y2] representing the ground truth bounding boxes
    :return: float representing the IoU value
    """
    # Calculate the intersection area
    if len(bbox_pred) != 4:
        return 0.0
    x1 = max(bbox_pred[0], bbox_gt[0])
    y1 = max(bbox_pred[1], bbox_gt[1])
    x2 = min(bbox_pred[2], bbox_gt[2])
    y2 = min(bbox_pred[3], bbox_gt[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the union area
    bbox_1_area = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
    bbox_2_area = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1])

    union_area = bbox_1_area + bbox_2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def get_new_idx(bbox_list, bbox_gt_list, idx1, idx2):
    new_idx1 = bbox_gt_list.index(bbox_list[idx1])
    new_idx2 = bbox_gt_list.index(bbox_list[idx2])

    assert new_idx1 != -1 and new_idx2 != -1
    return new_idx1, new_idx2


def get_overlap_bbox(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    return [x1, y1, x2, y2]


def match_and_calculate_iou(ground_truths, predictions):
    """
    Match ground truth bounding boxes with predicted bounding boxes based on IoU and calculate IoU scores.
    :param ground_truths: list of ground truth bounding boxes, each represented as a list of 4 floats [x1, y1, x2, y2]
    :param predictions: list of predicted bounding boxes, each represented as a list of 4 floats [x1, y1, x2, y2]
    :return: iou_scores: list of IoU scores for matched pairs
    """
    matched_gt = set()
    matched_pred = set()
    iou_scores = []
    iou_dict = []
    for idx, gt in enumerate(ground_truths):
        for jdx, pred in enumerate(predictions):
            iou_dict.append({'iou': get_iou(gt, pred), 'gt_idx': idx, 'pred_idx': jdx})

    iou_dict = sorted(iou_dict, key=lambda x: x['iou'], reverse=True)
    bbox_gt_list = []
    bbox_pred_list = []
    for iou_item in iou_dict:
        if iou_item['gt_idx'] in matched_gt or iou_item['pred_idx'] in matched_pred:
            continue
        matched_gt.add(iou_item['gt_idx'])
        matched_pred.add(iou_item['pred_idx'])
        bbox_gt_list.append(ground_truths[iou_item['gt_idx']])
        bbox_pred_list.append(predictions[iou_item['pred_idx']])
        iou_scores.append(iou_item['iou'])

    iou_scores.extend([0] * (len(ground_truths) - len(iou_scores)))
    discrepancy_list = set(map(tuple, ground_truths)) - set(map(tuple, bbox_gt_list))
    bbox_gt_list.extend([list(discrepancy) for discrepancy in discrepancy_list])

    bbox_pred_list.extend([[0, 0, 0, 0]] * (len(ground_truths) - len(bbox_pred_list)))
    # assert len(bbox_gt_list) == len(ground_truths)
    if len(bbox_gt_list) < len(ground_truths):
        bbox_gt_list.extend([ground_truths[0]] * (len(ground_truths) - len(bbox_gt_list)))
    assert len(bbox_pred_list) == len(ground_truths)
    return iou_scores, bbox_gt_list, bbox_pred_list


def calculate_bbox_iou(gt_dict, pred_dict, resolution):
    """
    Calculate the average IoU of bounding boxes between ground truth and predicted annotations.
    :param gt_dict: ground truth annotations dictionary, where keys are image IDs and values are dictionaries containing 'annotations'
    :param pred_dict: predicted annotations dictionary, where keys are image IDs and values are dictionaries containing category-wise bounding boxes
    :param resolution: resolution of the images, used to resize bounding boxes if necessary
    :return: average IoU score across all images
    """
    iou = 0
    N = 0
    for img_id in tqdm.tqdm(gt_dict.keys()):
        gt = gt_dict[img_id]['annotations']
        pred = pred_dict[img_id] if img_id in pred_dict.keys() else {}
        cat_list = set(list(anno['category'] for obj, anno in gt.items()))
        iou_scores = []
        for cat in cat_list:
            bbox_gt = list(anno['bbox'] for obj, anno in gt.items() if anno['category'] == cat)
            bbox_pred = pred[cat] if cat in pred.keys() else []
            bbox_gt = resize_bbox(bbox_gt, 1024)
            bbox_pred_ = resize_bbox(bbox_pred, resolution)
            iou_score, _, _ = match_and_calculate_iou(bbox_gt, bbox_pred_)
            iou_scores.extend(iou_score)

        iou += (sum(iou_scores) / len(iou_scores))
        N += 1

    return iou / N if N > 0 else 0


def calculate_overlap_iou(gt_dict, pred_dict, resolution):
    """
    Calculate the average IoU of overlapping bounding boxes between ground truth and predicted annotations.
    :param gt_dict: ground truth annotations dictionary, where keys are image IDs and values are dictionaries containing 'annotations' and 'valid_overlap'
    :param pred_dict: predicted annotations dictionary, where keys are image IDs and values are dictionaries containing category-wise bounding boxes
    :param resolution: resolution of the images, used to resize bounding boxes if necessary
    :return: average overlap IoU score across all images
    """
    iou = 0
    o_iou = 0
    N = 0
    for img_id in tqdm.tqdm(gt_dict.keys()):
        gt = gt_dict[img_id]['annotations']
        overlap_list = gt_dict[img_id]['valid_overlap']
        pred = pred_dict[img_id] if img_id in pred_dict.keys() else {}
        cat_list = set(list(anno['category'] for obj, anno in gt.items()))
        bbox_list = list(anno['bbox'] for obj, anno in gt.items())

        bbox_gt_list = []
        bbox_pred_list = []

        iou_scores = []

        for cat in cat_list:
            bbox_gt = list(anno['bbox'] for obj, anno in gt.items() if anno['category'] == cat)
            bbox_pred = pred[cat] if cat in pred.keys() else []
            bbox_gt = resize_bbox(bbox_gt, 1024)
            bbox_pred_ = resize_bbox(bbox_pred, resolution)
            iou_score, bbox_gt_cat, bbox_pred_cat = match_and_calculate_iou(bbox_gt, bbox_pred_)

            iou_scores.extend(iou_score)

            bbox_gt_list.extend(bbox_gt_cat)
            bbox_pred_list.extend(bbox_pred_cat)

        iou += (sum(iou_scores) / len(iou_scores))

        assert len(bbox_gt_list) == len(bbox_list)
        assert len(bbox_pred_list) == len(bbox_list)

        overlap_iou = []
        for overlap_item in overlap_list:
            idx1, idx2 = overlap_item['overlap_box'][0], overlap_item['overlap_box'][1]
            new_idx1, new_idx2 = get_new_idx(bbox_list, bbox_gt_list, idx1, idx2)
            overlap_bbox_gt = get_overlap_bbox(bbox_gt_list[new_idx1], bbox_gt_list[new_idx2])
            overlap_bbox_pred = get_overlap_bbox(bbox_pred_list[new_idx1], bbox_pred_list[new_idx2])
            overlap_iou.append(get_iou(overlap_bbox_gt, overlap_bbox_pred))

        o_iou += (sum(overlap_iou) / len(overlap_iou))
        N += 1
    return iou / N if N > 0 else 0, o_iou / N if N > 0 else 0
