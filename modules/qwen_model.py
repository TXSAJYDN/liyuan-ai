"""
千问多模态模型调用模块
- 封装 Qwen3-omni 模型的加载与推理接口
- 仅配置，按需加载（调用 load_model() 时才真正加载模型到 GPU）
- 支持纯文本对话和多模态（图像+文本）输入
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

from configs.settings import QWEN_CONFIG

logger = logging.getLogger(__name__)


class QwenModel:
    """千问多模态大模型封装"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.config = QWEN_CONFIG.copy()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self):
        """加载千问模型到显存（仅在需要使用时调用）"""
        if self._loaded:
            logger.info("千问模型已加载，跳过")
            return
        import torch
        from transformers import AutoProcessor
        from transformers import Qwen3OmniMoeForConditionalGeneration
        model_path = self.config["model_path"]
        logger.info(f"开始加载千问模型: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if self.config["torch_dtype"] == "auto" else self.config["torch_dtype"],
            device_map=self.config["device"],
            attn_implementation="sdpa",
        )
        self.model.eval()
        self._loaded = True
        logger.info("千问模型加载完成")

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """纯文本对话"""
        if not self._loaded:
            raise RuntimeError("千问模型未加载，请先调用 load_model()")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            return_audio=False,
            thinker_max_new_tokens=self.config["max_new_tokens"],
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def chat_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """纯文本对话（流式输出）"""
        if not self._loaded:
            raise RuntimeError("千问模型未加载，请先调用 load_model()")
        from transformers import TextIteratorStreamer
        import threading
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(inputs)
        gen_kwargs.update({
            "return_audio": False,
            "thinker_max_new_tokens": self.config["max_new_tokens"],
            "streamer": streamer,
        })
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
        thread.join()

    def chat_with_images(self, prompt: str, image_paths: List[str],
                         system_prompt: Optional[str] = None) -> str:
        """多模态对话（图像 + 文本）"""
        if not self._loaded:
            raise RuntimeError("千问模型未加载，请先调用 load_model()")
        from PIL import Image
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        inputs = self.processor(
            text=text, images=images if images else None,
            return_tensors="pt"
        ).to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            return_audio=False,
            thinker_max_new_tokens=self.config["max_new_tokens"],
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def analyze_opera_frames(self, image_paths: List[str],
                             knowledge_context: str = "") -> str:
        """戏曲视频帧结构化分析（核心功能）"""
        system_prompt = (
            "你是一位精通中国戏曲的专家，拥有丰富的戏曲表演艺术知识，"
            "熟悉京剧、昆曲、越剧、梆子、高腔等各大剧种的行当、程式、板式和服饰特征。"
            "你的任务是分析戏曲视频的关键帧图像，并给出专业的结构化分析。"
        )
        prompt_parts = ["请分析以下戏曲视频的关键帧图像序列。"]
        if knowledge_context:
            prompt_parts.append(f"\n参考以下专业知识：\n{knowledge_context}\n")
        prompt_parts.append(
            '请生成一段专业的结构化总结，以 JSON 格式输出，必须包含以下字段：\n'
            '{\n'
            '  "剧种剧目推测": "推测的剧种和剧目名称",\n'
            '  "核心行当": "涉及的行当（生、旦、净、丑等）",\n'
            '  "关键动作程式": "识别到的动作程式（如云手、起霸、亮相等）",\n'
            '  "唱腔板式推测": "推测的唱腔和板式",\n'
            '  "情感表达": "表演传达的情感",\n'
            '  "道具服饰细节": "识别到的道具和服饰特征",\n'
            '  "综合描述": "一段流畅的自然语言综合描述"\n'
            '}\n'
            '请确保分析专业、准确，如果无法确定某个字段，请标注为"待定"。'
        )
        prompt = "\n".join(prompt_parts)
        return self.chat_with_images(prompt, image_paths, system_prompt)

    def answer_opera_question(self, question: str,
                              knowledge_context: str = "",
                              is_rag: bool = True) -> str:
        """戏曲专业知识问答"""
        system_prompt = (
            "你是一位精通中国戏曲的专家，拥有丰富的戏曲知识。"
            "请根据提供的参考资料（如有）回答用户关于戏曲的问题。"
            "回答要准确、专业、通俗易懂。"
        )
        if is_rag and knowledge_context:
            prompt = (
                f"参考资料：\n{knowledge_context}\n\n"
                f"用户问题：{question}\n\n"
                "请结合以上参考资料和你的专业知识，直接回答用户的问题。"
                "回答应融会贯通，无需区分内容来源。"
            )
        else:
            prompt = (
                f"用户问题：{question}\n\n"
                "请根据你的戏曲专业知识回答。"
            )
        return self.chat(prompt, system_prompt)

    def explain_keyframe_match(self, image_path: str, user_query: str) -> str:
        """解释关键帧为何匹配用户查询"""
        system_prompt = "你是一位戏曲表演艺术专家。"
        prompt = (
            f"请用一两句话简要说明这张图片与用户描述「{user_query}」的关联。"
        )
        return self.chat_with_images(prompt, [image_path], system_prompt)


qwen_model = QwenModel()
