import os
import json
import csv
import os
from tqdm import tqdm
from collections import defaultdict
BASE_DIR = "data_cslu"
OUTPUT_DIR = os.path.join("data_cslu_splits", "gender")
os.makedirs(OUTPUT_DIR, exist_ok=True)


###This limits only to samples from 'https://github.com/vpspeech/ChildAugment/blob/main/CSLU_Trial_Finetune_Metadata/trial_list_combined_with_age_gender_abs'
gender_lookup = {}
valid_IDs = []
input_file = 'trial_list_combined_with_age_gender_abs.txt'
with open(input_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
                SpeakerID = os.path.basename(os.path.dirname(row['utterance-2']))
                gender_lookup[SpeakerID] = row['gender-2']
                if SpeakerID not in valid_IDs:
                    valid_IDs.append(SpeakerID)
             
                

print("starting!")

# Gather entries
for cslu_option in ["scripted", "spontaneous"]:
    AUDIO_ROOT = os.path.join(BASE_DIR, "speech", cslu_option)
    TRANS_ROOT = os.path.join(BASE_DIR, "trans", cslu_option)
    VERIFICATION_ROOT = os.path.join(BASE_DIR, "verify", "scripted")
    dataset_entries = []
    for root, _, files in os.walk(AUDIO_ROOT):
        for filename in files:
            if filename.endswith(".wav"):

                base_name = os.path.splitext(filename)[0]
                rel_path = os.path.relpath(root, AUDIO_ROOT)
                audio_path = os.path.join(root, filename)

                
                
                transcript_path = os.path.join(TRANS_ROOT, rel_path, base_name + ".txt")
                if(os.path.exists(transcript_path)):
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                    path_parts = os.path.normpath(audio_path).split(os.sep)
                    try:

                        grade = path_parts[-4]
                        SpeakerID = path_parts[-2]
                        if SpeakerID not in valid_IDs:
                            continue
                        else:
                            gender = gender_lookup[SpeakerID]
                            verification_path = os.path.join(VERIFICATION_ROOT, rel_path, base_name + ".txt")
                            if cslu_option=="scripted" and os.path.exists(verification_path):
                                with open(verification_path, "r", encoding="utf-8") as f:
                                    verification = f.read().strip()
                            else:
                                verification = "N/A"
                            dataset_entries.append({
                                    "dataset": "cslu_" + cslu_option,
                                    "audio_path": audio_path,
                                    "SpeakerID": SpeakerID,
                                    "text": text,
                                    "grade": grade,
                                    "verification": verification,
                                    "gender": gender
                                })
                    
                    except IndexError:
                        print(f"Warning: unexpected path format for {audio_path}")
                        continue


                        # Extract grade and verification from path
            
            

                            
                                
                

    if cslu_option == "scripted":
        ###make a custom train,dev,test split with no speaker overlap and equal grade & gender distribution
        unique_grades = list(set(entry["grade"] for entry in dataset_entries))
        unique_genders = list(set(entry["gender"] for entry in dataset_entries))
        print(f"Unique grades: {unique_grades}")
        print(f"Unique genders: {unique_genders}")
        train_idx = []; dev_idx = []; test_idx = []
        for grade in unique_grades:
            for gender in unique_genders:

                grade_gender_idx = []
                ID_to_indices = defaultdict(list)
                for i, entry in enumerate(dataset_entries):
                    if entry.get("grade") == grade and entry.get("gender")==gender:
                        grade_gender_idx.append(i)
                        ID_to_indices[entry["SpeakerID"]].append(i)
                print(grade, gender, len(grade_gender_idx))

                unique_IDs = list(ID_to_indices.keys())
                    # Split into train/dev/test
                n = len(unique_IDs)
                train_IDs = unique_IDs[:int(0.8 * n)]
                dev_IDs = unique_IDs[int(0.8 * n):int(0.9 * n)]
                test_IDs = unique_IDs[int(0.9 * n):]
                print(n)
                    # Collect corresponding indices
                for SpeakerID in train_IDs:
                    train_idx.extend(ID_to_indices[SpeakerID])
                for SpeakerID in dev_IDs:
                    dev_idx.extend(ID_to_indices[SpeakerID])
                for SpeakerID in test_IDs:
                    test_idx.extend(ID_to_indices[SpeakerID])

                print(f"nr of train:  {len(train_idx)}")
                print(f"nr of test:  {len(test_idx)}")
                print(f"nr of dev:  {len(dev_idx)}")
                    
        splits = {
            "train": [dataset_entries[i] for i in train_idx],
            "development": [dataset_entries[i] for i in dev_idx],
            "test": [dataset_entries[i] for i in test_idx]
        }
    
    else:
        splits = {"test": dataset_entries}


    # Write splits to separate files
    for split_name, entries in splits.items():
        output_path = os.path.join(OUTPUT_DIR, "splits", cslu_option, f"{split_name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in tqdm(entries, desc=f"Saving {split_name}, {cslu_option}", dynamic_ncols=True, leave=True):
                f.write(json.dumps(entry) + "\n")


    print(f"Done! {len(dataset_entries)} total samples split into {OUTPUT_DIR}")