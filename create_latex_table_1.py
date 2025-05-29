import os
import json

# Define paths
results_dir = "results"
cslu_options = ["scripted", "spontaneous"]
models = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "younger-whisper": "YoungerWhisper",
    "older-whisper": "OlderWhisper"
}
testsets = ["all_ages_all_genders", "younger_all_genders", "older_all_genders"]

# Initialize data
results = {model_display: {} for model_display in models.values()}
for cslu in cslu_options:
    for model_folder, model_display in models.items():
        path = os.path.join(results_dir, cslu, model_folder, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        for test in testsets:
            wer = summary.get(test, {}).get("wer", 0.0)
            results[model_display][(cslu, test)] = f"{wer:.2f}"

# Build LaTeX rows
rows = []
for model in results:
    row = [model]
    for cslu in cslu_options:
        for test in testsets:
            row.append(results[model].get((cslu, test), "0.00"))
    rows.append(" & ".join(row) + r" \\ \hline")

# LaTeX table
latex = r"""
\begin{table}[ht]
\centering
\resizebox{0.5\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
 & \multicolumn{3}{c|}{\textbf{Scripted}} & \multicolumn{3}{c|}{\textbf{Spontaneous}} \\
\hline
\textbf{Model} & All & Younger & Older & All & Younger & Older \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}%
}
\caption{WER Results by Model, Style, and Age Group}
\end{table}
"""

with open("wer_table_style_age.tex", "w") as f:
    f.write(latex.strip())
print("Saved to wer_table_style_age.tex")