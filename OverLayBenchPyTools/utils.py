import json
import re

def bbox_prompt(annotations, img, *args, **kwargs):
    categories = list(set(e['category'] for e in annotations['annotations'].values()))
    categories = sorted(categories)
    prompt = [{
        'role': 'user',
        'content': [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are Qwen-VL, created by Alibaba Cloud. You are a helpful assistant."
                }]
            },
            {
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

def entity_prompt(annotations, img, *args, **kwargs):
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
    prompt = [{
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