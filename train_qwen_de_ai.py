import os

# 强制要求：在脚本最开头（导入其他包前）设置 HF_ENDPOINT 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 针对 datasets 库增加超时相关的环境变量设置和禁用部分检查，以防国内网络连接超时
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HTTPX_TIMEOUT"] = "300" 
# 开启 PyTorch 的显存碎片优化，有助于缓解 OOM 问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from modelscope import snapshot_download

def main():
    # ==========================================
    # 1. 模型与 Tokenizer 基础配置
    # ==========================================
    # model_id = "qwen/Qwen2.5-7B-Instruct" # 注意：ModelScope上的命名格式可能稍有不同，如果找不到之前的ID，可以替换为ModelScope实际对应的ID
    model_id = "Qwen/Qwen3-4B-Instruct-2507" 
    
    print(f"正在从 ModelScope 下载或加载基础模型: {model_id}...")
    model_dir = snapshot_download(model_id)

    # 4bit 量化配置 (nf4, 双重量化, bfloat16 计算)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 强制要求：pad_token = eos_token
    tokenizer.pad_token = tokenizer.eos_token

    print("正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 准备 kbit 训练 (开启梯度检查点等)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ==========================================
    # 2. LoRA 配置 (PEFT)
    # ==========================================
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    # 获取 PEFT 模型
    model = get_peft_model(model, lora_config)
    
    # 强制要求：训练前 print 模型 trainable parameters
    print("模型可训练参数统计：")
    model.print_trainable_parameters()

    # ==========================================
    # 3. 数据集处理
    # ==========================================
    print("正在加载并处理数据集...")
    # 强制要求：优先使用 category == "C3" 或 "C4" 的样本，以及 Human 样本增强
    raw_dataset = load_dataset("AnxForever/chinese-ai-detection-dataset")
    
    def process_example(example):
        category = example.get("category", "")
        text = example.get("text", "")
        messages = []
        
        # 处理带有 AI 痕迹的文本 (C3, C4)
        if category in ["C3", "C4"]:
            parts = text.split("[SEP]")
            if len(parts) >= 2:
                human_part = parts[0].strip()
                ai_part = parts[1].strip()
                if human_part and ai_part:
                    messages = [
                        {
                            "role": "user", 
                            "content": f"请把下面这段带有明显AI生成痕迹的中文文本改写得更自然、更像真人随手写的风格。只去除AI味（模板化、啰嗦、过度礼貌、机械句式等），保留原意，不要添加新内容、不要改变事实、不要大幅改长度：\n\n{ai_part}"
                        },
                        {
                            "role": "assistant", 
                            "content": human_part
                        }
                    ]
        # 可选增强：加入 Human 样本
        elif category == "Human":
            if text.strip():
                messages = [
                    {
                        "role": "user", 
                        "content": "写一段自然、接地气的中文描述："
                    },
                    {
                        "role": "assistant", 
                        "content": text.strip()
                    }
                ]
        return {"messages": messages}

    # 处理训练集
    train_dataset = raw_dataset["train"].map(process_example, num_proc=4)
    # 过滤掉不符合条件的样本（即没有生成 messages 的）
    train_dataset = train_dataset.filter(lambda x: len(x["messages"]) > 0, num_proc=4)
    # 最终 dataset 只保留 "messages" 列
    columns_to_remove = [col for col in train_dataset.column_names if col != "messages"]
    train_dataset = train_dataset.remove_columns(columns_to_remove)

    # 如果数据集有验证集也一并处理，否则从训练集中切分一部分作为验证集
    if "validation" in raw_dataset:
        eval_dataset = raw_dataset["validation"].map(process_example, num_proc=4)
        eval_dataset = eval_dataset.filter(lambda x: len(x["messages"]) > 0, num_proc=4)
        eval_dataset = eval_dataset.remove_columns([col for col in eval_dataset.column_names if col != "messages"])
    else:
        print("未检测到单独的 validation 集，正在从训练集随机划分 5% 用于验证...")
        split_dataset = train_dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    print(f"处理完成！训练集样本数: {len(train_dataset)}, 验证集样本数: {len(eval_dataset)}")

    # ==========================================
    # 4. SFTTrainer 格式化函数
    # ==========================================
    def formatting_func(example):
        # 兼容 batch 和单个 example 的情况
        if isinstance(example.get("messages", [])[0], list):
            return [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) 
                for msgs in example["messages"]
            ]
        else:
            return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    # ==========================================
    # 5. 训练超参设置 (TrainingArguments / SFTConfig)
    # ==========================================
    output_dir = "./qwen-de-AI-flavor-lora"
    
    # 针对非常新版的 trl (>= 0.12.0)
    # SFTConfig 的参数列表发生过多次变化。
    # 最新版中，max_seq_length, packing 等必须通过 SFTConfig 传递，但旧版中没有这些参数。
    # 这里使用一个小 trick 动态适配：
    config_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,      # 减小单卡 batch size 以防止 OOM
        "per_device_eval_batch_size": 2,       # 减小单卡 batch size 以防止 OOM
        "gradient_accumulation_steps": 16,     # 2 * 16 = 32 (保持总的有效 batch size 约等于 32 不变)
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "max_grad_norm": 0.3,
        "weight_decay": 0.01,
        "dataset_text_field": None,
    }
    
    import inspect
    if "max_seq_length" in inspect.signature(SFTConfig.__init__).parameters:
        config_kwargs["max_seq_length"] = 1024
    if "packing" in inspect.signature(SFTConfig.__init__).parameters:
        config_kwargs["packing"] = True
        
    training_args = SFTConfig(**config_kwargs)

    # ==========================================
    # 6. SFTTrainer 实例化
    # ==========================================
    print("初始化 SFTTrainer...")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "formatting_func": formatting_func,
    }
    
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        
    if "max_seq_length" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["max_seq_length"] = 1024
    if "packing" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["packing"] = True

    trainer = SFTTrainer(**trainer_kwargs)

    # ==========================================
    # 7. 开始训练与模型保存
    # ==========================================
    print("开始训练...")
    trainer.train()

    print("训练完成，正在保存模型和 Tokenizer...")
    # 强制要求：训练完后保存
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"微调后的模型已成功保存至 {output_dir}")

if __name__ == "__main__":
    main()
