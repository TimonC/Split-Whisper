import os
import json

results_dir = "results"

# Models of interest and their display names
model_name_map = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "all_dataset_younger_all_genders": "YoungerWhisper",
    "all_dataset_older_all_genders": "OlderWhisper"
}

cslu_options = ["scripted", "spontaneous"]

# Mapping testset keys to display names used in summary files
testsets = {
    "all_ages_all_genders": "All Ages",
    "younger_all_genders": "Younger",
    "older_all_genders": "Older"
}

# Initialize results dictionary: results[model][style][age_group] = WER
results = {name: {style: {} for style in cslu_options} for name in model_name_map.values()}

for cslu_option in cslu_options:
    style_dir = os.path.join(results_dir, cslu_option)
    if not os.path.isdir(style_dir):
        continue
    
    for model_folder in os.listdir(style_dir):
        if model_folder not in model_name_map:
            continue
        
        model_display = model_name_map[model_folder]
        summary_path = os.path.join(style_dir, model_folder, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        # summary keys might be testset names (e.g. all_ages_all_genders)
        for testset_key, entry in summary.items():
            if testset_key not in testsets:
                continue
            wer = entry.get("wer", 0.0)
            results[model_display][cslu_option][testsets[testset_key]] = f"{wer:.2f}"

# Now build LaTeX table rows
rows = []
for model in model_name_map.values():
    row = [model]
    for cslu_option in cslu_options:
        for age_group in ["All Ages", "Younger", "Older"]:
            wer = results.get(model, {}).get(cslu_option, {}).get(age_group, "0.00")
            row.append(wer)
    rows.append(" & ".join(row) + r" \\" + "\n\\hline")

# LaTeX tabular snippet (without document environment)
latex_table = r"""
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
 & \multicolumn{3}{c|}{\textbf{Scripted}} & \multicolumn{3}{c|}{\textbf{Spontaneous}} \\
\hline
\textbf{Model} & All Ages & Younger & Older & All Ages & Younger & Older \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
"""

# Save LaTeX table snippet to a file
output_path = os.path.join(results_dir, "wer_summary_table.tex")
with open(output_path, "w") as f:
    f.write(latex_table.strip())

print(f"LaTeX table snippet saved to: {output_path}")