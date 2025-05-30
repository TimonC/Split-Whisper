from datasets import load_dataset, DatasetDict, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import argparse
from datetime import datetime
import os
from load_data_custom_cslu import load_data_custom_cslu

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    parser.add_argument("--base_model", type=str, default="openai/whisper-small.en")
    parser.add_argument("--finetuned_model", type=str, default="aadel4/kid-whisper-small-en-myst")
    parser.add_argument("--using_base_whisper", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, default="./data_cslu_splits")
    parser.add_argument("--json_option", type=str, default="all")
    parser.add_argument("--cslu_option", type=str, default="scripted")
    parser.add_argument("--data_split", type=str, default="all_ages_all_genders")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-whisper")

    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    return parser


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    padding_vector: torch.Tensor = None  # padding vector to detect padded frames

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # If padding_vector is not set, compute it as min values per feature dimension over the batch
        if self.padding_vector is None:
            self.padding_vector = batch["input_features"].min(dim=1).values.min(dim=0).values
            # batch["input_features"].shape: (batch_size, seq_len, feature_dim)
            # min(dim=1) => min across seq_len -> (batch_size, feature_dim)
            # then min across batch_size -> (feature_dim,)

        # Create attention mask by checking if each vector equals padding vector
        # shape: (batch_size, seq_len, feature_dim)
        is_padding = (batch["input_features"] == self.padding_vector).all(dim=-1)  # (batch_size, seq_len)
        batch["attention_mask"] = (~is_padding).long()

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch




def train(args):
    metric = evaluate.load("wer")

    if "en" in args.base_model:
        tokenizer = WhisperTokenizer.from_pretrained(args.base_model, language="english", task="transcribe")
        processor = WhisperProcessor.from_pretrained(args.base_model, language="english", task="transcribe")
    else:
        tokenizer = WhisperTokenizer.from_pretrained(args.base_model, task="transcribe")
        processor = WhisperProcessor.from_pretrained(args.base_model, task="transcribe")
    if args.using_base_whisper:
        finetuned_model = args.base_model
    else:
        finetuned_model = args.finetuned_model
    dataset_path = os.path.join(args.data_path, args.json_option, "data", args.cslu_option, args.data_split)
    custom_dataset = load_data_custom_cslu(dataset_path, mode="train")

    normalizer = tokenizer._normalize
    model = WhisperForConditionalGeneration.from_pretrained(finetuned_model)
    model.resize_token_embeddings(len(tokenizer))

    # Create the padding vector by computing global min per feature across train dataset if needed
    # For simplicity here, we do it on first batch in collator (lazy init)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    if "en" not in args.base_model:
        print("Not english!")
        model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.config.suppress_tokens = []

    if args.num_train_epochs > 0:
        args.max_steps = 0
        args.evaluation_strategy = "epoch"
        print(f"Training model for {args.num_train_epochs} epochs")
        print("max steps ignored")
    else:
        print(f"Training model for {args.max_steps} steps")
        print("num_train_epochs ignored")

    
    output_dir = os.path.join(
        args.output_dir,
        finetuned_model.replace("/", "-"),
        f"{args.json_option}_dataset_{args.data_split}",
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    print(f"Output directory: {output_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.max_learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs, 
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        evaluation_strategy=args.evaluation_strategy,
        logging_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        max_grad_norm=args.max_grad_norm,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=custom_dataset["train"],
        eval_dataset=custom_dataset["development"],
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric, normalizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()


def compute_metrics(pred, tokenizer, metric, normalizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [normalizer(text) for text in pred_str]
    label_str = [normalizer(text) for text in label_str]
    pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
    label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)