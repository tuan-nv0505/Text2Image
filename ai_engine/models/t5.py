# -*- coding: utf-8 -*-
import os
import re
import html
import urllib.parse as ul
import ftfy
import torch
from bs4 import BeautifulSoup
from transformers import T5EncoderModel, AutoTokenizer


class T5Embedder:
    bad_punct_regex = re.compile(r'[#®•©™&@·º½¾¿¡§~\)\(\]\[\}\{\|\\\/\*]{1,}')  # noqa

    def __init__(self, device, dir_or_name='google/t5-v1_1-large', *, cache_dir=None, hf_token=None,
                 use_text_preprocessing=True, torch_dtype=None, model_max_length=120):

        self.device = torch.device(device)
        self.use_text_preprocessing = use_text_preprocessing
        self.model_max_length = model_max_length
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/huggingface/hub')

        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            dir_or_name,
            cache_dir=self.cache_dir,
            token=hf_token
        )

        self.model = T5EncoderModel.from_pretrained(
            dir_or_name,
            cache_dir=self.cache_dir,
            token=hf_token,
            torch_dtype=self.torch_dtype
        ).to(self.device).eval()

    def get_text_embeddings(self, texts):
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        input_ids = text_tokens_and_mask['input_ids'].to(self.device)
        attention_mask = text_tokens_and_mask['attention_mask'].to(self.device)

        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )['last_hidden_state'].detach()

        return text_encoder_embs, attention_mask

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)

        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            '', caption)
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            '', caption)

        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)

        # CJK Characters
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)

        # Dash standard
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',
            '-', caption)

        # Quotes standard
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)
        caption = re.sub(r'&quot;?', '', caption)
        caption = re.sub(r'&amp', '', caption)

        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)
        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)
        # \n
        caption = re.sub(r'\\n', ' ', caption)
        # tags
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        caption = re.sub(r'#\d{5,}\b', '', caption)
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        caption = re.sub(r'[\"\']{2,}', r'"', caption)
        caption = re.sub(r'[\.]{2,}', r' ', caption)
        caption = re.sub(self.bad_punct_regex, r' ', caption)
        caption = re.sub(r'\s+\.\s+', r' ', caption)

        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)
        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)
        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)
        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()

    def to(self, device):
        self.model.to(device)


if __name__ == "__main__":
    embedder = T5Embedder(device='cpu')
    sample_texts = [
        "Mamba cho bài toán text to image"
    ]
    for i, text in enumerate(sample_texts):
        cleaned = embedder.text_preprocessing(text)
    embeddings, attention_mask = embedder.get_text_embeddings(sample_texts)
    print(f"embed: {embeddings.shape}")
    print(f"attention_mask: {attention_mask.shape}")