import os
import sys
import textwrap
from typing import List

# import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, default_data_collator, AutoTokenizer,
                          get_linear_schedule_with_warmup, LlamaTokenizer, LlamaForCausalLM)
from datasets import load_dataset, Dataset, DatasetDict
from peft import (get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig,
                  TaskType, PeftType, prepare_model_for_int8_training, LoraConfig, 
                  get_peft_model_state_dict)
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
import textwrap

device = "cuda" if torch.cuda.is_available() else "cpu"
max_length = 64
num_epochs = 5

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 16
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 30
OUTPUT_DIR = "/home/work/data00/result/experiences"

DEVICE = 'cuda'

BASE_MODEL = "beomi/llama-2-ko-7b"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="cuda:0", 
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

# training
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=num_epochs,
    # fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    # report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)


def create_prompt(data_point):
    return f"""아래는 작업을 설명하는 명령어로, 함께 입력되는 입력 문맥 및 답변과 쌍을 이룹니다. 주어진 문맥에 맞게 적절한 응답을 작성하세요.
### 명령어:
{data_point["instruction"]}
### 입력 문맥:
{data_point["input"]}
### Response:
"""

def generate_response(prompt: str, model: model):
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.1,
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=16,
        )

def format_response(response) -> str:
    decoded_output = tokenizer.decode(response.sequences[0])
    response = decoded_output.split("### Response:")[1].strip()
    return "\n".join(textwrap.wrap(response))

def ask_alpaca(prompt, model: model) -> str:
    prompt = create_prompt(prompt)
    response = generate_response(prompt, model)
    response = format_response(response)
    return response


if __name__ == '__main__':
    import pickle
    from tqdm import tqdm
    import pandas as pd
    import evaluate

    # open train / validation dataset
    with open('/home/work/data00/dataset/llama2/train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open('/home/work/data00/dataset/llama2/validation.pickle', 'rb') as f:
        val_data = pickle.load(f)

    # ready for trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    model = torch.compile(model)

    print('ready for training')
    # train
    trainer.train()
    OUTPUT_DIR = "/home/work/data00/result/experiences"
    model.save_pretrained(OUTPUT_DIR)    

    # check result
    with open('/home/work/data00/dataset/llama2/test.pickle', 'rb') as f:
        test_data = pickle.load(f)
    DEVICE = 'cuda'
    a = []
    i = 0
    for inst in tqdm(test_data):
        # print(inst)

        response = ask_alpaca(inst, model)
        # print(response)
        a.append((i, response))

        i += 1
    tmp_result = pd.DataFrame(a)

    with open(file='/home/work/data00/result/finetune.pickle', mode = 'wb') as f:
        pickle.dump(tmp_result, f)
    print('finish dump finetuned test data')

    metric = evaluate.load('rouge')
    result = metric.compute(predictions=a, references=test_data['output'])

    print(result)


        