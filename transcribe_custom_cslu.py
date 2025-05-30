from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer
from transformers import WhisperFeatureExtractor
import os
from evaluate import load
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from load_data_custom_cslu import load_data_custom_cslu
import argparse
from datasets import Dataset, load_dataset, Audio
import json 
from datetime import datetime
def transcribe(args):
    # Construct model names based on args
    base_model = f"{args.base_model}-{args.whisper_size}"
    finetuned_model = f"{args.finetuned_model}-{args.whisper_size}"
    if args.whisper_language == "english":
        base_model += ".en"
        finetuned_model += "-en"
    finetuned_model += "-myst"

    if args.using_base_whisper:
        finetuned_model = base_model
        
    if args.split_whisper_path is not None:
        finetuned_model = args.split_whisper_path
    
    finetuned_model_name = os.path.basename(finetuned_model)
    


    print(f"Base model: {base_model}")
    print(f"Finetuned Model: {finetuned_model}")

    # Set up tokenizer and normalizer
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language=args.whisper_language, task="transcribe")
    normalizer = tokenizer._normalize


    # Load model and pipeline
    metric = load("wer")
    model = WhisperForConditionalGeneration.from_pretrained(finetuned_model)
    model.config.forced_decoder_ids = None ##i dont love doing this but i have to to avoid error
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=WhisperFeatureExtractor.from_pretrained(base_model),
        device="cuda",
        chunk_length_s=30,
    )
    if args.json_option=="all":
        data_splits = ["all_ages_all_genders", "older_all_genders", "younger_all_genders"]
    else:
        data_splits = ["all_ages_all_genders", "all_ages_Girl", "all_ages_Boy", "older_Girl", "younger_Girl", "older_Boy", "younger_Boy"]
    for cslu_option in args.cslu_options:
        # Load in test data
        data_path = os.path.join(args.data_path, args.json_option, "data", cslu_option)
        print(f"Dataset: {data_path}")
        print(f"Data splits: {data_splits}")
        testsets = {}
        for ds in data_splits:
            testset = load_data_custom_cslu(os.path.join(data_path, ds), mode="test")

            testset = testset.cast_column("audio_path", Audio())
            testset = testset.rename_column("audio_path", "audio")

            testset = testset['test']
            testsets[ds] = testset
        # Prepare output directory
        transcription_dir = os.path.join("huggingface_models_transcription", finetuned_model, "transcriptions")
        os.makedirs(transcription_dir, exist_ok=True)
        print(testsets)

        # Transcription loop
        results_summary = {}
        for testset_name, testset in testsets.items():
            datasets = {"ground_truths": [], "hypotheses": []}
            print(f"Transcribing {testset_name}")
            transcription_file_path = os.path.join(transcription_dir, f"{testset_name}.txt")

            with open(transcription_file_path, "w") as transcription_file:

                for out, line in tqdm(
                    zip(pipe(KeyDataset(testset, "audio")), testset),
                    desc=f"Transcribing {testset_name}",
                    total=len(testset)
                ):
                    transcription = out["text"]
                    ground_truth = line["sentence"]
                    path = line["audio"]["path"]
                    datasets["ground_truths"].append(normalizer(ground_truth))
                    datasets["hypotheses"].append(normalizer(transcription))
                    transcription_file.write(path + "\t" + transcription + "\n")

                # Compute and print WER
            wer = metric.compute(predictions=datasets["hypotheses"], references=datasets["ground_truths"]) * 100
            print(f"CSLU Option: {cslu_option} Dataset: {testset_name} WER: {wer:.2f}")

            # Store summary
            results_summary[testset_name] = {
                "wer": wer,
                "model": finetuned_model_name,
                "num_samples": len(testset),
                "cslu_option": cslu_option,
                "ground_truths": datasets["ground_truths"],
                "hypotheses": datasets["hypotheses"]
            }

        # Save final summary
        results_dir = os.path.join(args.results_dir, args.json_option, cslu_option) 
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(results_dir, f"{finetuned_model_name}.json")
        with open(summary_path, "w") as summary_file:
            json.dump(results_summary, summary_file, indent=2)

        print(f"Saved transcription summary to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_size", type=str, default="small")
    parser.add_argument("--whisper_language", type=str, default="english")
    parser.add_argument("--using_base_whisper", action="store_true", default=False)
    parser.add_argument("--base_model", type=str, default="openai/whisper")
    parser.add_argument("--finetuned_model", type=str, default="aadel4/kid-whisper")
    parser.add_argument("--data_path", type=str, default="./data_cslu_splits")
    parser.add_argument("--json_option", type=str, default="all")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--cslu_options", type=str, nargs='+', default=["spontaneous", "scripted"])
    parser.add_argument("--split_whisper_path", type=str, default=None)
    args = parser.parse_args()

    transcribe(args)