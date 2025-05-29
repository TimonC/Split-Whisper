import json
from datasets import load_dataset, DatasetDict, Audio, Dataset, concatenate_datasets
import sys
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from whisper_normalizer.english import EnglishTextNormalizer
from collections import defaultdict
import gc
import os
import argparse
import shutil
import whisper

FRAME_DURATION = 160 / 16000.0  # 0.01s per frame

_whisper_model = None

def get_whisper_model(model_name="small.en"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def seconds_to_frame(ts_s: float) -> int:
    return int(ts_s / FRAME_DURATION)

def get_word_frame_ranges(wav_path, model_name="small.en"):
    model = get_whisper_model(model_name)
    result = model.transcribe(wav_path, word_timestamps=True)
    frame_ranges = []
    for seg in result["segments"]:
        for word in seg.get("words", []):
            start_f = seconds_to_frame(word["start"])
            end_f = seconds_to_frame(word["end"])
            frame_ranges.append((start_f, end_f))
    return frame_ranges

def create_dataset(args):
    base_model = args.base_model
    if args.language=="english":
        base_model += ".en"

    print(f"Base model: {base_model}")
    hf_base_model = "openai/whisper-" + base_model
    feature_extractor = WhisperFeatureExtractor.from_pretrained(hf_base_model)
    tokenizer = WhisperTokenizer.from_pretrained(hf_base_model, language="english", task="transcribe")
    normalizer = EnglishTextNormalizer()

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["sentence"] = normalizer(batch["sentence"])
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        batch["audio_path"] = batch["audio"]["path"]
        batch["frame_ranges"] = get_word_frame_ranges(batch["audio"]["path"], base_model)
        return batch

    def create_empty_entry():
        return {
            "audio": [],
            "sentence": [],
            "dataset": []
        }

    split_dict = {}
    if args.split_by_grade:
        for age_group in ["younger", "older"]:
            if args.split_by_gender:
                for gender in ["Girl", "Boy"]:
                    split_dict[f"{age_group}_{gender}"] =  create_empty_entry()
            else:
                split_dict[f"{age_group}_all_genders"] =  create_empty_entry()
    else:
        if args.split_by_gender:
                for gender in ["Girl", "Boy"]:
                    split_dict[f"all_ages_{gender}"] =  create_empty_entry()
        else:
                split_dict[f"all_ages_all_genders"] =  create_empty_entry()

    json_path = os.path.join(args.root_dir, args.json_version, "splits", args.cslu_option, args.split + ".json")
    print(f"Retrieving json data splits from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            entry = json.loads(line)
            if entry["verification"].isdigit():
                if int(entry["verification"]) not in args.verification_to_include:
                    continue

            grade = int(entry["grade"])
            if  grade <= args.max_grade_to_include:
                if args.split_by_grade:
                    age_group = "younger" if grade<args.split_grade else "older"
                else:
                    age_group = "all_ages"
                if args.split_by_gender:
                    gender = entry["gender"]
                else:
                    gender = "all_genders"
                split_dict[f"{age_group}_{gender}"]["audio"].append(entry["audio_path"])
                split_dict[f"{age_group}_{gender}"]["sentence"].append(entry["text"])
                split_dict[f"{age_group}_{gender}"]["dataset"].append(entry["dataset"])

    print(split_dict.keys())

    datasets = {}
    for key in split_dict:
        count = len(split_dict[key]["audio"])
        print(f"{key}: {count} samples")
        datasets[key] = Dataset.from_dict(split_dict[key]).cast_column("audio", Audio())
    del split_dict
    print("Dataset loaded")

    custom_dataset = DatasetDict()
    print(datasets.items())
    for dataset_name, dataset in datasets.items():
        print(dataset_name, dataset)
        dataset = dataset.map(
            prepare_dataset,
            load_from_cache_file=False,
            remove_columns=["audio"],
            num_proc=4,
            batched=False,
        )

        outdir = os.path.join(args.root_dir, args.json_version, "data-timeframe", args.cslu_option, dataset_name, args.split)
        if os.path.exists(outdir):
            print(f"Removing existing directory: {outdir}")
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)
        dataset.save_to_disk(outdir)
        custom_dataset[dataset_name] = dataset

    return custom_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data_cslu_splits')
    parser.add_argument('--cslu_option', type=str, default="scripted")
    parser.add_argument('--json_version', type=str, default="all")
    parser.add_argument('--split', type=str, default="development")
    parser.add_argument('--base_model', type=str, default="openai/whisper-small")
    parser.add_argument('--split_grade', type=int, default=4)
    parser.add_argument('--split_by_grade', action='store_true')
    parser.add_argument('--no_split_by_grade', action='store_false', dest='split_by_grade')
    parser.add_argument('--split_by_gender', action='store_true')
    parser.add_argument('--no_split_by_gender', action='store_false', dest='split_by_gender')
    parser.set_defaults(split_by_grade=True, split_by_gender=False)
    parser.add_argument('--max_grade_to_include', type=int, default=7)
    parser.add_argument('--verification_to_include', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--language', type=str, default="english")
    args = parser.parse_args()

    print(f"Preparing dataset {args.split}")
    print(f"Language: {args.language}")
    print(f"Grade range: 0-{args.max_grade_to_include}")
    if args.split_by_grade:
        print(f"Split dataset into 'younger' and 'older' with split_grade: {args.split_grade}")
    else:
        print("Save dataset with all grades in the range as 'all_ages'")
    ds = create_dataset(args)
    print(f"Dataset {args.split} prepared")
    print(ds)
