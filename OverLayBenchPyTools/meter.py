import json
from collections import defaultdict

import torch
import os.path
import logging

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ast import literal_eval
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from utils import bbox_prompt, entity_prompt, relationship_prompt, extract_and_parse_json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class OverLayBenchMeter:
    def __init__(self,
                 root,
                 extension='png',
                 save_dir='./metrics',
                 split='simple',
                 resolution=1024,
                 bs_qwen=10,
                 bs_clip=1024):
        self.root = root
        self.extension = extension
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.split = split
        self.resolution = resolution
        self.bs_qwen = bs_qwen
        self.bs_clip = bs_clip
        self.logger = self._setup_logger()
        self.logger.info(f"Initializing OverLayBenchMeter with:\n"
                         f"root: {self.root}\n"
                         f"extension: {self.extension}\n"
                         f"save_dir: {self.save_dir}\n"
                         f"split: {self.split}\n"
                         f"resolution: {self.resolution}\n"
                         f"bs_qwen: {bs_qwen}\n"
                         f"bs_clip: {bs_clip}")

        # Load the dataset
        self.logger.info(f"Loading OverLayBench dataset {self.split} split...")
        self.dataset = load_dataset("cywang143/OverLayBench_Dataset",
                                    split=self.split)
        self.logger.info(f"Dataset loaded with {len(self.dataset)} samples.")

        # Load the model and processor
        self.logger.info("Loading Qwen2.5-VL-32B-Instruct model and processor...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True,
                                                       padding_side='left')
        self.logger.info("Model and processor loaded successfully.")

    def _setup_logger(self):
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
            file_handler = logging.FileHandler(os.path.join(self.save_dir, 'overlay_bench.log'))
            file_handler.setFormatter(stream_formatter)
            logger.addHandler(file_handler)
        return logger

    def _batch_message(self, anns, prompt_fn=bbox_prompt):
        messages = []
        for ann in anns:
            # TODO: current huggingface dataset version does not contain img id
            img = Image.open(os.path.join(f"{self.root}/{ann['img_id']}.{self.extension}")).convert("RGB")
            messages.append(prompt_fn(ann, img, self.resolution))
        return messages

    def _batch_response(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        try:
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        except torch.cuda.OutOfMemoryError:
            if len(messages) == 1:
                raise torch.cuda.OutOfMemoryError
            self.logger.warning(f"Out of memory error with batch size {len(messages)}")
            del text, image_inputs, video_inputs, inputs
            torch.cuda.empty_cache()
            safe_bs = len(messages) // 2
            safe_output_text = []
            for i in range(2):
                safe_output_text.extend(self._batch_response(messages[i * safe_bs:(i + 1) * safe_bs]))
            return safe_output_text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def _qwen_vqa(self, mode, prompt_fn, key, file_name):
        res = defaultdict(dict)
        self.logger.info(f"Starting {mode} VQA...")
        for i in tqdm(range(0, len(self.dataset), self.bs_qwen)):
            batch = self.dataset[i: min(i + self.bs_qwen, len(self.dataset))]
            batch = [literal_eval(ann) for ann in batch['annotations']]
            messages = self._batch_message(batch, prompt_fn)
            responses = self._batch_response(messages)
            for j in range(len(responses)):
                img_id = batch[j]['img_id']
                output = responses[j]
                res[img_id][key] = output
        self.logger.info(f"{mode} VQA completed.")

        # Save results
        output_file = os.path.join(self.save_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=4)
        self.logger.info(f"Results saved to {output_file}")

        # TODO: calculate accuracy for entity and relationship VQA

    def bbox_predication(self):
        res = {}
        self.logger.info("Starting bbox prediction...")
        for i in tqdm(range(0, len(self.dataset), self.bs_qwen)):
            batch = self.dataset[i: min(i + self.bs_qwen, len(self.dataset))]
            batch = [literal_eval(ann) for ann in batch['annotations']]
            # batch
            messages = self._batch_message(batch, bbox_prompt)
            responses = self._batch_response(messages)
            res.update({f"{ann['img_id']:08}": extract_and_parse_json(text) for text, ann in
                        zip(responses, batch)})
        self.logger.info("BBox prediction completed.")

        # Save results
        output_file = os.path.join(self.save_dir, 'bbox_predictions.json')
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=4)
        self.logger.info(f"Results saved to {output_file}")

        # TODO: calculate mIoU and O-mIoU

    def entity_vqa(self):
        self._qwen_vqa('Entity', entity_prompt, 'entity_vqa', 'entity_VQA.json')

    def relationship_vqa(self):
        self._qwen_vqa('Relationship', relationship_prompt, 'relation_vqa', 'relation_VQA.json')

    def clip_score(self):
        # TODO: Implement CLIP score calculation
        pass

if __name__ == "__main__":
    # Example usage
    meter = OverLayBenchMeter(root='./data', extension='png', save_dir='./metrics', split='simple')
    meter.bbox_predication()
    meter.entity_vqa()
    meter.relationship_vqa()
    meter.clip_score()

