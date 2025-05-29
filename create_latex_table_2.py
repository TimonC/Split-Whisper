import os
import json

results_dir = "results"

# Display names for each model folder
model_name_map = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "girl-whisper": "GirlWhisper",
    "boy-whisper": "BoyWhisper",
    "younger_girl-whisper": "YoungerGirlWhisper",
    "younger_boy-whisper": "YoungerBoyWhisper"
}

# Define gender+age group keys for table layout
testsets = [
    ("Girl", "Younger", "gender_dataset_younger_Girl"),
    ("Girl", "Older", "gender_dataset_older_Girl"),
    ("Boy", "Younger", "gender_dataset_younger_Boy"),
    ("Boy", "Older", "gender_dataset_older_Boy")
]

# Choose which CSLU option to use
cslu_option = "scripted"  # or "spontaneous"

# Initialize results dictionary
results = {display_name: {} for display_name in model_name_map.values()}

for folder, display_name in model_name_map.items():
    summary_path = os.path.join(results_dir, cslu_option, folder, "summary.json")
    if not os.path.isfile(summary_path):
        continue
    with open(summary_path) as f:
        summary = json.load(f)

    for gender, age, testset_key in testsets:
        wer = summary.get(testset_key, {}).get("wer", 0.0)
        results[display_name][(gender, age)] = f"{wer:.2f}"

# Build LaTeX rows
rows = []
for model in results:
    row = [model]
    for gender in ["Girl", "Boy"]:
        for age in ["Younger", "Older"]:
            val = results[model].get((gender, age), "0.00")
            row.append(val)
    rows.append(" & ".join(row) + r" \\" + "\n\\hline")

# Final LaTeX tabular (without preamble or document)
latex_table = r"""
\begin{tabular}{|l|c|c|c|c|}
\hline
 & \multicolumn{2}{c|}{\textbf{Girl}} & \multicolumn{2}{c|}{\textbf{Boy}} \\
\hline
\textbf{Model} & Younger & Older & Younger & Older \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
"""

# Save LaTeX table to file
output_path = os.path.join(results_dir, f"wer_gender_table_{cslu_option}.tex")
with open(output_path, "w") as f:
    f.write(latex_table.strip())

print(f"LaTeX gender table saved to: {output_path}")