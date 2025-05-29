from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer
from transformers import WhisperFeatureExtractor
import os
from evaluate import load
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from load_data_custom_cslu import load_data_custom_cslu
import argparse


def transcribe(args):
    # Construct model names based on args
    base_model = f"{args.base_model}-{args.whisper_size}"
    finetuned_model = f"{args.finetuned_model}-{args.whisper_size}"
    if args.whisper_language == "english":
        base_model += ".en"
        finetuned_model += "-en"
    finetuned_model += "-myst"

    print(f"Base model: {base_model}")
    print(f"Finetuned Model: {finetuned_model}")

    # Set up tokenizer and normalizer
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language=args.whisper_language, task="transcribe")
    normalizer = tokenizer._normalize

    # Load model and pipeline
    metric = load("wer")
    model = WhisperForConditionalGeneration.from_pretrained(finetuned_model)
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=WhisperFeatureExtractor.from_pretrained(base_model),
        device="cuda",
        chunk_length_s=30,
    )

    # Prepare test data path
    test_path = os.path.join(args.data_path, args.json_option, "data", args.cslu_option, "test")
    print(f"Test dataset: {test_path}")

    # Load test data
    testsets = load_data_custom_cslu(test_path, mode = "test")

    # Prepare output directory
    transcription_dir = os.path.join("huggingface_models_transcription", finetuned_model, "transcriptions")
    os.makedirs(transcription_dir, exist_ok=True)

    # Transcription loop
    for testset in testsets:
        datasets = {"ground_truths": [], "hypotheses": []}
        print(f"Transcribing {testset}")
        transcription_file_path = os.path.join(transcription_dir, f"{testset}.txt")

        with open(transcription_file_path, "w") as f:
            for out, line in tqdm(zip(pipe(KeyDataset(testsets[testset], "audio")), testsets[testset]), desc=f"Transcribing {testset}", total=len(testsets[testset])):
                transcription = out["text"]
                ground_truth = line["sentence"]
                path = line["audio"]["path"]

                # Normalize and record
                datasets["ground_truths"].append(normalizer(ground_truth))
                datasets["hypotheses"].append(normalizer(transcription))
                f.write(path + "\t" + transcription + "\n")

        # Compute and print WER
        wer = metric.compute(predictions=datasets["hypotheses"], references=datasets["ground_truths"]) * 100
        print(f"Dataset: {testset} WER: {wer:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_size", type=str, default="small")
    parser.add_argument("--whisper_language", type=str, default="english")
    parser.add_argument("--base_model", type=str, default="openai/whisper")
    parser.add_argument("--finetuned_model", type=str, default="aadel4/kid-whisper")
    parser.add_argument("--data_path", type=str, default="./cslu_data_splits")
    parser.add_argument("--json_option", type=str, default="all")
    parser.add_argument("--cslu_option", type=str, default="scripted")
    parser.add_argument("--data_split", type=str, default="all_genders_all_ages")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-whisper")
    args = parser.parse_args()

    transcribe(args)