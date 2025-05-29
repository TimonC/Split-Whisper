from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer
from transformers import WhisperFeatureExtractor
import os
from datasets import Dataset, load_dataset, Audio
import soundfile as sf
import json
# from whisper_normalizer.english import EnglishTextNormalizer
from collections import defaultdict
from evaluate import load
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from Rishabh_norm import RishabhTextNormalizer

from load_data_custom_cslu import load_data_custom_cslu
import argparse


def transcribe(args):
    print(f"Base model: {args.base_model}")
    if "rishabh" in args.base_model.lower():
        normalizer = RishabhTextNormalizer
        tokenizer = WhisperTokenizer.from_pretrained(args.base_model, language="english", task="transcribe")
    else:
        tokenizer = WhisperTokenizer.from_pretrained(args.base_model, language="english", task="transcribe")
        normalizer = tokenizer._normalize
        
    metric = load("wer")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    pipe = pipeline(task = "automatic-speech-recognition", model=model, tokenizer=args.base_model, feature_extractor=WhisperFeatureExtractor.from_pretrained(args.base_model), device="cuda", chunk_length_s=30,)
    
    test_path = os.path.join(args.data_path, args.json_option, "data", args.cslu_option, "test")
    print(f"Base model: {args.base_model}")
    print(f"Finetuned Model: {args.finetuned_model}")
    print(f"Test dataset: {test_path}")
    
    testsets = load_data_custom_cslu(test_path)
    transcription_dir = args.finetuned_model + "/transcriptions"
    if "fine-tuned-whisper" not in transcription_dir:
        transcription_dir = "huggingface_models_transcription/" + transcription_dir
    os.makedirs(transcription_dir, exist_ok=True)
    for testset in testsets:
        datasets = {"ground_truths": [], "hypotheses": []}
        print(f"Transcribing {testset.name}")
        transcription_file = transcription_dir + "/" + testset.name + ".txt"
        transcription_file = open(transcription_file, "w")
        for out, line in tqdm(zip(pipe(KeyDataset(testset, "audio")), testset), desc=f"Transcribing {testset.name}", total=len(testset)):

            transcription = out["text"]
            ground_truth = line["sentence"]
            path = line["audio"]["path"]
            datasets["ground_truths"].append(normalizer(ground_truth))
            datasets["hypotheses"].append(normalizer(transcription))
            transcription_file.write(path + "\t" + transcription + "\n")
        wer = metric.compute(predictions=datasets["hypotheses"], references=datasets["ground_truths"]) * 100
        print("Dataset: {} WER: {}".format(testset.name, wer))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openai/whisper-small.en")
    parser.add_argument("--finetuned_model", type=str, default="openai/whisper-small.en")
    parser.add_argument("--data_path", type=str, default="./cslu_data_splits")
    parser.add_argument("--json_option", type=str, default="all")
    parser.add_argument("--cslu_option", type=str, default="scripted")
    parser.add_argument("--data_split", type=str, default="all_genders_all_ages")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-whisper")
    args = parser.parse_args()
    transcribe(args)
