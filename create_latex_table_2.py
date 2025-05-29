import os
import json

results_dir = "results"
cslu_option = "scripted"

# Models and how they should appear (added OlderGirlWhisper and OlderBoyWhisper)
models = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "girl-whisper": "GirlWhisper",
    "boy-whisper": "BoyWhisper",
    "younger_girl-whisper": "YoungerGirlWhisper",
    "younger_boy-whisper": "YoungerBoyWhisper",
    "older_girl-whisper": "OlderGirlWhisper",
    "older_boy-whisper": "OlderBoyWhisper"
}

# Mapping of testset keys to table structure, unchanged
testsets = {
    ("Girl", "All", "gender_dataset_all_ages_Girl"),
    ("Girl", "Younger", "gender_dataset_younger_Girl"),
    ("Girl", "Older", "gender_dataset_older_Girl"),
    ("Boy", "All", "gender_dataset_all_ages_Boy"),
    ("Boy", "Younger", "gender_dataset_younger_Boy"),
    ("Boy", "Older", "gender_dataset_older_Boy"),
}

# Initialize results dictionary
results = {display: {} for display in models.values()}

# Load data from JSON files
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
    # Girl: All, Younger, Older
    for age in ["All", "Younger", "Older"]:
        row.append(results[model].get(("Girl", age), "0.00"))
    # Boy: All, Younger, Older
    for age in ["All", "Younger", "Older"]:
        row.append(results[model].get(("Boy", age), "0.00"))
    rows.append(" & ".join(row) + r" \\ \hline")

# Create LaTeX code
latex = r"""
\begin{table}[ht]
\centering
\resizebox{0.5\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
 & \multicolumn{3}{c|}{\textbf{Girl}} & \multicolumn{3}{c|}{\textbf{Boy}} \\
\hline
\textbf{Model} & All & Younger & Older & All & Younger & Older \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}%
}
\caption{WER Results by Model, Gender, and Age Group}
\end{table}
"""

# Save LaTeX to file
with open("wer_table_gender_age.tex", "w") as f:
    f.write(latex.strip())

print("Saved to wer_table_gender_age.tex")