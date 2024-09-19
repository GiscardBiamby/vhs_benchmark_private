import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base_solver import Solver


class Phi3Solver(Solver):
    def __init__(self, image_root, debug_mode, **config):
        super().__init__(image_root, debug_mode)
        self.solver_name = config["name"]
        self.config = config
        self.huggingface_model_id = config["huggingface_model_id"]
        self.processor = AutoProcessor.from_pretrained(
            self.huggingface_model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_model_id,
            device_map="cuda",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt, image_paths):
        # image_paths = image_paths[:3]

        # Phi3
        user_prompt = "<|user|>\n"
        assistant_prompt = "<|assistant|>\n"
        prompt_suffix = "<|end|>\n"
        images = []
        image_tokens = ""
        for img_num, image_path in enumerate(image_paths):
            image_tokens += f"<|image_{img_num+1}|>\n"
            images.append(Image.open(image_path))
        question = (
            f"{user_prompt}{image_tokens}{prompt}{prompt_suffix}{assistant_prompt}"
        )
        # print(image_tokens)
        # print("Num images: ", len(images))

        inputs = self.processor(text=question, images=images, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print("Generated text: ", generated_text)
        # Hack: remove the prefix question
        # generated_text = generated_text.split("ASSISTANT:")[1]
        return generated_text, None
