import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn
import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

    def encode_text(self, text_list, device='cuda'):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)
    
def test():
    clip = 'huggingface'
    sam = sam_model_registry["vit_h"](checkpoint='ckpts/sam_vit_h_4b8939.pth').to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    if clip=='clip':
        clip_model, preprocess = clip.load("ViT-B/16", device="cuda", jit=False)
    elif clip=='open_clip':
        clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    elif clip=='huggingface':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        image = Image.open('../project/data/lerf_ovs/waldo_kitchen/images/00066.jpg')
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        text_embeds=outputs['text_embeds']
        image_embeds=outputs['image_embeds']
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    @torch.no_grad()
    def query(image_path, labels):
        img_dir = image_path
        img_size=None
        image = cv2.imread(str(img_dir))
        if img_size:
            image = cv2.resize(image, (img_size[1], img_size[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks, masks_s, masks_m, masks_l = mask_generator.generate(image)

        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        pad_imgs = []
        segs = []
        scores = []
        for mask in sorted_masks:
            bbox = mask["bbox"]
            seg_mask = mask["segmentation"]
            score = mask["stability_score"] * mask["predicted_iou"]
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

            crop = (image * seg_mask[:, :, np.newaxis])[y1:y2, x1:x2]
            h, w, _ = crop.shape

            l = max(h, w)
            pad = np.zeros((l, l, 3), dtype=np.uint8)
            if h > w:
                pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = crop
            else:
                pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = crop
            pad_imgs.append(cv2.resize(pad, (224, 224)))
            segs.append(seg_mask)
            scores.append(score)

        if len(pad_imgs) == 0:
            print("Error: no mask detected!")
            return torch.zeros((512, image.shape[0], image.shape[1]), dtype=torch.half)

        pad_imgs = np.stack(pad_imgs, axis=0)  # B, H, W, 3
        pad_imgs = torch.from_numpy(pad_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
        pad_imgs = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )(pad_imgs).cuda()
        mask_features=[]
        for pad_img in pad_imgs:
            mask_features.append(clip_model.encode_image(pad_img[None]).cpu())
        mask_features = torch.cat(mask_features).cuda()
        mask_features = mask_features / (mask_features.norm(dim=0, keepdim=True) + 1e-8)
        print(f'{mask_features.shape=}')

        pixel_features = torch.zeros((512, image.shape[0], image.shape[1]), dtype=torch.half, device='cuda')
        for idx, seg_mask in enumerate(segs):
            pixel_features[:, seg_mask] += mask_features[idx].unsqueeze(1)  # * scores[idx]
        pixel_features = pixel_features / (pixel_features.norm(dim=0, keepdim=True) + 1e-8)
        print(f'{pixel_features.shape=}')

        # pixel_feature = model_2d.extract_image_feature('../project/data/lerf_ovs/waldo_kitchen/images/00066.jpg')
        if False: # clip
            labels = labels
            text = clip.tokenize(labels)
            text = text.cuda()
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else: # open_clip
            labels = labels
            text_features = clip_model.encode_text(labels)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(f'{text_features.shape=}')

        sim = text_features@mask_features.T
        indice = sim.argmax(dim=-1)
        output_path = f'temp/open_clip/{os.path.basename(image_path)}'
        os.makedirs(output_path, exist_ok=True)
        for i,t in zip(indice.tolist(), labels):
            image = cv2.imread(str(img_dir))
            mask=sorted_masks[i]['segmentation']
            image[mask] = image[mask]*0.4+np.array([255,0,0], dtype=np.uint8)*0.6
            cv2.imwrite(os.path.join(output_path, f'{t}.jpg'), image)
    
    query('../project/data/lerf_ovs/waldo_kitchen/images/00066.jpg', ['plastic ladle', 'pot', 'refrigerator', 'spatula'])
    query('../project/data/lerf_ovs/waldo_kitchen/images/00053.jpg', ['knife', 'ottolenghi', 'pour-over vessel', 'Stainless steel pots', 'toaster', 'yellow desk'])
    query('../project/data/lerf_ovs/waldo_kitchen/images/00089.jpg', ['cabinet', 'ketchup', 'knife'])
    query('../project/data/lerf_ovs/waldo_kitchen/images/00140.jpg', ['dark cup', 'frog cup', 'plate', 'sink', 'spoon'])
    query('../project/data/lerf_ovs/waldo_kitchen/images/00154.jpg', ['knife', 'plate', 'red cup', 'sink'])

def make_input():
    dataset='lerf_ovs/waldo_kitchen'
    input_path=os.path.join('data', dataset, 'images')
    output_path=os.path.join('data', dataset, 'input')
    inputs=os.listdir(input_path)
    os.makedirs(output_path, exist_ok=True)
    for input in inputs:
        with open(os.path.join(input_path, input), 'rb') as fin, open(os.path.join(output_path, input[6:]), 'wb') as fout:
            fout.write(fin.read())
def main()->None:
    # make_input()
    test()

if __name__ == '__main__':
    main()