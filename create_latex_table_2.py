import os
import json

results_dir = "results"
cslu_option = "scripted"

# Models and how they should appear
models = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "girl-whisper": "GirlWhisper",
    "boy-whisper": "BoyWhisper",
    "younger_girl-whisper": "YoungerGirlWhisper",
    "younger_boy-whisper": "YoungerBoyWhisper"
}

# Mapping of testset keys to table structure
testsets = {
    ("All", "", "gender_dataset_all_ages_all_genders"),
    ("Girl", "Younger", "gender_dataset_younger_Girl"),
    ("Girl", "Older", "gender_dataset_older_Girl"),
    ("Boy", "Younger", "gender_dataset_younger_Boy"),
    ("Boy", "Older", "gender_dataset_older_Boy"),
}

# Initialize results
results = {display: {} for display in models.values()}

# Load data
for folder, display in models.items():
    path = os.path.join(results_dir, cslu_option, folder, "summary.json")
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        summary = json.load(f)
    for gender, age, key in testsets:
        wer = summary.get(key, {}).get("wer", 0.0)
        results[display][(gender, age)] = f"{wer:.2f}"

# Build LaTeX table rows
rows = []
for model in results:
    row = [model]
    # All (no subcategories)
    row.append(results[model].get(("All", ""), "0.00"))
    # Girl: Younger and Older
    for age in ["Younger", "Older"]:
        row.append(results[model].get(("Girl", age), "0.00"))
    # Boy: Younger and Older
    for age in ["Younger", "Older"]:
        row.append(results[model].get(("Boy", age), "0.00"))
    rows.append(" & ".join(row) + r" \\ \hline")

# Create LaTeX code
latex = r"""
\begin{table}[ht]
\centering
\resizebox{0.5\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|}
\hline
 & \textbf{All} & \multicolumn{2}{c|}{\textbf{Girl}} & \multicolumn{2}{c|}{\textbf{Boy}} \\
\hline
\textbf{Model} &  & Younger & Older & Younger & Older \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}%
}
\caption{WER Results by Model, Gender, and Age Group (Including All)}
\end{table}
"""

# Save to file
with open("wer_table_gender_age_all.tex", "w") as f:
    f.write(latex.strip())

print("Saved to wer_table_gender_age_all.tex")