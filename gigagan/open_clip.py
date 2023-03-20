import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List
from torch import nn

import open_clip


def l2norm(t):
    return F.normalize(t, dim=-1)


@beartype
class OpenClipAdapter(nn.Module):
    def __init__(self,
                 name='ViT-B/32',
                 pretrained='laion400m_e32',
                 tokenizer_name='ViT-B-32-quickgelu',
                 eos_id=49407):
        super().__init__()

        clip, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        self.clip = clip
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]

        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self, layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def image_size(self):
        image_size = self.clip.visual.image_size
        if isinstance(image_size, tuple):
            return max(image_size)
        return image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.positional_embedding.shape[0]

    def embed_texts(self, texts: List[str]):
        ids = self.tokenizer(texts)
        ids = ids[..., :self.max_text_len]

        is_eos_id = (ids == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim=-1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value=True)
        text_mask = text_mask & (ids != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(ids)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return l2norm(text_embed.float()), text_encodings.float()

    def embed_image(self, image):
        assert not self.cleared
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return l2norm(image_embed.float()), None
