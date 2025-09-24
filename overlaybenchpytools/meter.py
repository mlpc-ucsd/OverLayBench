import json
import torch
import os.path

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from .clip import CLIPScore
from ast import literal_eval
from datasets import load_dataset
from prettytable import PrettyTable
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from .utils import (
    bbox_prompt, entity_prompt, relationship_prompt,
    extract_and_parse_json, calculate_success_rate,
    calculate_overlap_iou, calculate_bbox_iou,
    _setup_logger
)


class OverLayBenchMeter:
    def __init__(self,
                 root,
                 extension='png',
                 save_dir='./metrics',
                 resolution=1024,
                 bs_qwen=8,
                 bs_clip=1024,
                 use_vllm=False,
                 vllm_args=None):
        """
        Initializes the OverLayBenchMeter for evaluating the OverLayBench dataset.

        :param root: The root directory where the generated images are stored, the image filename should be in f"{img_id}".{extension}.
        :param extension: The extension of the image files, default is 'png'.
        :param save_dir: The directory where the results will be saved.
        :param resolution: The resolution of the images, default is 1024, can be 1024 or 512.
        :param bs_qwen: The batch size for Qwen model inference, default is 8. We recommend using `all` when `use_vllm` is True, which will set the batch size to the length of the dataset.
        :param bs_clip: The batch size for CLIPScore model inference, default is 1024.
        :param use_vllm: Whether to use vLLM for inference, default is False.
        :param vllm_args: Additional arguments for vLLM, if `use_vllm` is True. Please refer to the vLLM documentation (https://docs.vllm.ai/en/latest/api/vllm/index.html#vllm.LLM) for reference.
        """
        self.root = root
        self.extension = extension
        self.save_root = save_dir
        Path(self.save_root).mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.bs_qwen = bs_qwen
        self.bs_clip = bs_clip
        self.use_vllm = use_vllm
        self.logger = _setup_logger(os.path.join(self.save_root, 'overlay_bench.log'))
        self.logger.info(f"Initializing OverLayBenchMeter with:\n"
                         f"root: {self.root}\n"
                         f"extension: {self.extension}\n"
                         f"save_dir: {self.save_root}\n"
                         f"resolution: {self.resolution}\n"
                         f"bs_qwen: {bs_qwen}\n"
                         f"bs_clip: {bs_clip}")

        # Load the dataset
        self.split = None
        self.seed = None
        self.save_dir = None
        self.dataset = None
        self.annotations = None
        self.images = None

        # Load the model and processor
        self.logger.info("Loading Qwen2.5-VL-32B-Instruct model and processor...")
        if self.use_vllm:
            from vllm import LLM
            vllm_args = vllm_args or {}
            self.model = LLM(
                model="Qwen/Qwen2.5-VL-32B-Instruct", **vllm_args,
            )
            self.sampling_params = self.model.get_default_sampling_params()
            self.sampling_params.max_tokens = 1024
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-32B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True,
                                                       padding_side='left')
        self.logger.info("Model and processor loaded successfully.")


    def _generate_annotations_dict(self):
        """
        Generates a dictionary of annotations from the dataset.

        :return: A dictionary where keys are image IDs and values are the corresponding annotations.
        """
        ann_dict = {}
        for data in self.dataset:
            img_id = data['image_id']
            ann = literal_eval(data['annotation'])
            ann_dict[img_id] = ann
        return ann_dict

    def _read_images(self):
        images = {}
        for img_id in tqdm(self.annotations, desc="Loading images"):
            images[img_id] = Image.open(os.path.join(self.root, self.split,
                                                     str(self.seed), f"{img_id}.{self.extension}")).convert("RGB")
        return images

    def set_split(self, split, seed):
        self.split = split
        self.seed = seed
        self.save_dir = os.path.join(self.save_root, f"{self.split}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Loading OverLayBench dataset {self.split} split...")
        self.dataset = load_dataset("cywang143/OverLayBench_Eval",
                                    split=self.split)
        self.annotations = self._generate_annotations_dict()
        self.images = self._read_images()
        self.logger.info(f"Dataset loaded with {len(self.dataset)} samples.")

    def _batch_message(self, anns, prompt_fn=bbox_prompt):
        """
        Batches the messages for the Qwen model based on the annotations.

        :param anns: A list of dictionaries containing annotations and image IDs.
        :param prompt_fn: A function to generate the prompt for each annotation, default is bbox_prompt.
        :return: A list of messages formatted for the Qwen model.
        """
        messages = []
        for ann in anns:
            img = self.images[ann['img_id']]
            messages.append(prompt_fn(ann, img, self.resolution))
        return messages

    def _batch_response(self, messages):
        """
        Processes a batch of messages and generates responses using the Qwen model.

        :param messages: A list of messages formatted for the Qwen model.
        :return: A list of generated responses from the model.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if self.use_vllm:
            inputs = [
                {
                    "prompt": text_single,
                    "multi_modal_data": {"image": [image_input_single]},
                }
                for text_single, image_input_single in zip(text, image_inputs)
            ]
        else:
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
            if self.use_vllm:
                outputs = self.model.generate(
                    inputs,
                    sampling_params=self.sampling_params,
                )
            else:
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
        if self.use_vllm:
            output_text = [output.outputs[0].text for output in outputs]
        else:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text

    def _qwen_vqa(self, task, prompt_fn, file_name):
        """
        General method to handle the Qwen VQA tasks (BBox Prediction, Entity VQA, Relationship VQA).

        :param task: The task to perform, can be 'BBox Prediction', 'Entity VQA', or 'Relationship VQA'.
        :param prompt_fn: A function to generate the prompt for the task.
        :param file_name: The name of the file to save the results.
        :return: A dictionary containing the results of the task.
        """
        valid_tasks = ['BBox Prediction', 'Entity VQA', 'Relationship VQA']
        assert task in valid_tasks, \
            f"Mode must be in {valid_tasks}, but get {task}."
        res = {}
        self.logger.info(f"Starting {task}...")
        for i in tqdm(range(0, len(self.dataset), self.bs_qwen)):
            batch = self.dataset[i: min(i + self.bs_qwen, len(self.dataset))]
            img_ids = batch['image_id']
            batch = [literal_eval(ann) for ann in batch['annotation']]
            batch = [{'img_id': img_id, **ann} for img_id, ann in zip(img_ids, batch)]
            messages = self._batch_message(batch, prompt_fn)
            responses = self._batch_response(messages)
            res.update({f"{ann['img_id']:08}": extract_and_parse_json(text) for text, ann in
                        zip(responses, batch)})
        self.logger.info(f"{task} completed.")

        # Save results
        output_file = os.path.join(self.save_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=4)
        self.logger.info(f"Results saved to {output_file}")
        return res


    def bbox_predication(self):
        """
        Performs bounding box prediction using the Qwen model and calculates IoU metrics.

        :return: A tuple containing the bounding box IoU and overlap IoU.
        """
        self.check_set_split()
        res = self._qwen_vqa('BBox Prediction', bbox_prompt, 'baseline_bbox_predictions.json')
        bbox_iou, overlap_iou = calculate_overlap_iou(self.annotations, res, self.resolution)
        self.logger.info(f"Bbox IoU: {bbox_iou:.4f}, Overlap IoU: {overlap_iou:.4f}")
        return bbox_iou, overlap_iou


    def entity_vqa(self):
        """
        Performs Entity VQA using the Qwen model and calculates the success rate.
        :return: The success rate of the Entity VQA task.
        """
        self.check_set_split()
        task = 'Entity VQA'
        res = self._qwen_vqa(task, entity_prompt,  'entity_VQA.json')
        success_rate = calculate_success_rate(res, self.annotations, task)
        self.logger.info(f"Success rate for {task}: {success_rate:.4f}")
        return success_rate

    def relationship_vqa(self):
        """
        Performs Relationship VQA using the Qwen model and calculates the success rate.
        :return: The success rate of the Relationship VQA task.
        """
        self.check_set_split()
        task = 'Relationship VQA'
        res = self._qwen_vqa(task, relationship_prompt,  'relation_VQA.json')
        success_rate = calculate_success_rate(res, self.annotations, task)
        self.logger.info(f"Success rate for {task}: {success_rate:.4f}")
        return success_rate

    def clip_score(self):
        """
        Calculates the CLIP score for the generated images against the ground truth annotations.
        :return: The CLIP score evaluation results, containing a global CLIPScore and a local CLIPScore.
        """
        self.check_set_split()
        # Load the CLIPScore model
        self.logger.info("Initializing CLIPScore model...")
        clip_score_model = CLIPScore(
            root=os.path.join(self.root, self.split),
            gt_json=self.annotations,
            save_dir=self.save_dir,
            extension=self.extension,
            resolution=self.resolution,
            batch_size=self.bs_clip,
            images=self.images,
            logger=self.logger
        )
        self.logger.info("CLIPScore model initialized successfully.")
        ret = clip_score_model.evaluation()
        del clip_score_model  # Free memory
        torch.cuda.empty_cache()

        return ret

    def check_set_split(self):
        if self.split is None:
            raise ValueError("Split is not set. Please set the split using set_split() before evaluating.")

        if self.bs_qwen == "all":
            self.bs_qwen = len(self.dataset)

    def evaluate(self):
        """
        Runs all evaluation tasks: bounding box prediction, entity VQA, relationship VQA, and CLIP score.
        :return: None
        """
        self.check_set_split()

        pt = PrettyTable()
        pt.field_names = ["Split", "Seed", "mIoU", "O-mIoU", "SR_E", "SR_R", "CLIP_global", "CLIP_local"]

        global_clip, local_clip = self.clip_score()
        iou, o_iou = self.bbox_predication()
        sr_e = self.entity_vqa()
        sr_r = self.relationship_vqa()

        pt.add_row([self.split,
                    self.seed,
                    f"{iou:.4f}",
                    f"{o_iou:.4f}",
                    f"{sr_e:.4f}",
                    f"{sr_r:.4f}",
                    f"{global_clip:.4f}",
                    f"{local_clip:.4f}"])

        self.logger.info(f"Evaluation results for {self.split} split:\n{pt}")

