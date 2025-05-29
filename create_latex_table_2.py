import os
import json

results_dir = "results"
cslu_option = "scripted"

models = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "girl-whisper": "GirlWhisper",
    "boy-whisper": "BoyWhisper",
    "younger_girl-whisper": "YoungerGirlWhisper",
    "older_girl-whisper": "OlderGirlWhisper",
    "younger_boy-whisper": "YoungerBoyWhisper",
    "older_boy-whisper": "OlderBoyWhisper",
}

testsets = {
    ("Girl", "All", "gender_dataset_all_ages_Girl"),
    ("Girl", "Younger", "gender_dataset_younger_Girl"),
    ("Girl", "Older", "gender_dataset_older_Girl"),
    ("Boy", "All", "gender_dataset_all_ages_Boy"),
    ("Boy", "Younger", "gender_dataset_younger_Boy"),
    ("Boy", "Older", "gender_dataset_older_Boy"),
}

results = {name: {} for name in models.values()}
for folder, display in models.items():
    path = os.path.join(results_dir, cslu_option, folder, "summary.json")
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        summary = json.load(f)
    for gender, age, key in testsets:
        wer = summary.get(key, {}).get("wer", 0.0)
        results[display][(gender, age)] = wer

# Bold best per row
def format_row(model_name, wer_dict):
    cells = []
    for g in ["Girl", "Boy"]:
        for a in ["All", "Younger", "Older"]:
            cells.append(wer_dict.get((g, a), 0.0))
    min_val = min(cells)
    formatted = [f"\\textbf{{{v:.2f}}}" if v == min_val else f"{v:.2f}" for v in cells]
    return model_name + " & " + " & ".join(formatted) + r" \\"

group1 = ["Whisper", "KidWhisper"]
group2 = ["GirlWhisper", "BoyWhisper"]
group3 = ["YoungerGirlWhisper", "OlderGirlWhisper", "YoungerBoyWhisper", "OlderBoyWhisper"]

rows = []
for group in [group1, group2, group3]:
    for model in group:
        rows.append(format_row(model, results[model]))
    rows.append(r"\hline")

latex = r"""
\begin{table}[ht]
\centering
\resizebox{0.95\textwidth}{!}{%
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

with open("wer_table_gender_age.tex", "w") as f:
    f.write(latex.strip())

print("Saved to wer_table_gender_age.tex")