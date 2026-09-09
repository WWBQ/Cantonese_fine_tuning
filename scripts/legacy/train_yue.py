import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported

# ================= 1. 基础配置 =================
model_name = "unsloth/llama-3-8b-instruct-bnb-4bit"  # 推荐 4bit 节省显存
max_seq_length = 2048  # 根据显存调整，3090/4090 建议 2048
load_in_4bit = True  # 使用 4bit 量化加载

# 数据文件路径 (请确保文件已上传至此路径)
TRAIN_PATH = "/root/autodl-tmp/data/yue_train.jsonl"
EVAL_PATH = "/root/autodl-tmp/data/yue_val.jsonl"

# ================= 2. 加载模型与分词器 =================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

# 添加 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA Rank，越高能力越强但显存占用越多
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 覆盖更多层，语感更地道
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ================= 3. 数据预处理 =================
# 定义 Llama-3 的 Prompt 模板
prompt_style = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个精通粤语和普通话的助手，请根据要求完成翻译或对话任务。<|eot_id|><|start_header_id|>user<|end_header_id|>

{}
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt_style.format(instruction, input, output)
        texts.append(text)
    return {"text": texts, }


# 加载数据集
dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "eval": EVAL_PATH})
train_dataset = dataset["train"].map(formatting_prompts_func, batched=True)
eval_dataset = dataset["eval"].map(formatting_prompts_func, batched=True)

# ================= 4. 配置训练参数 =================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 传入验证集
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,  # 多核处理数据

    args=TrainingArguments(
        per_device_train_batch_size=4,  # 如果显存不够就改 2
        gradient_accumulation_steps=4,  # 等效 Batch Size = 4*4 = 16
        warmup_steps=100,
        max_steps=3000,  # 244万条不需要跑完，先跑3000步观察
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,  # 每 10 步打印一次训练 Loss

        # 验证与监控配置
        eval_strategy="steps",  # 按步数评估
        eval_steps=200,  # 每 200 步进行一次模拟考
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,  # 训练完自动加载验证集表现最好的模型
        metric_for_best_model="eval_loss",

        output_dir="outputs_yue",
        report_to="none",  # 可改为 "wandb" 开启在线监控
    ),

    # 添加早停机制：如果连续 3 次评估 Loss 都不降，提前收工
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ================= 5. 开始执行 =================
print("🚀 正在点火，开始粤语微调...")
trainer.train()

# 保存训练好的模型
print("💾 训练完成，正在保存模型...")
model.save_pretrained("yue_model_lora")
tokenizer.save_pretrained("yue_model_lora")