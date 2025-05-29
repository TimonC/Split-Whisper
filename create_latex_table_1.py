import os
import json

# Paths and setup
results_dir = "results"
cslu_options = ["scripted", "spontaneous"]
models = {
    "whisper": "Whisper",
    "kid-whisper": "KidWhisper",
    "younger-whisper": "YoungerWhisper",
    "older-whisper": "OlderWhisper"
}
testsets = ["all_ages_all_genders", "younger_all_genders", "older_all_genders"]

# Load WERs
results = {name: {} for name in models.values()}
for cslu in cslu_options:
    for folder, display in models.items():
        path = os.path.join(results_dir, cslu, folder, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        for test in testsets:
            wer = summary.get(test, {}).get("wer", 0.0)
            results[display][(cslu, test)] = wer

# Build LaTeX rows with bold for best per row
def format_row(model, scores):
    vals = []
    for cslu in cslu_options:
        for test in testsets:
            vals.append(scores.get((cslu, test), 0.0))
    min_val = min(vals)
    row_cells = [f"\\textbf{{{v:.2f}}}" if v == min_val else f"{v:.2f}" for v in vals]
    return model + " & " + " & ".join(row_cells) + r" \\"

# Grouping
group1 = ["Whisper", "KidWhisper"]
group2 = ["YoungerWhisper", "OlderWhisper"]

rows = []
for group in [group1, group2]:
    for model in group:
        rows.append(format_row(model, results[model]))
    rows.append(r"\hline")

# LaTeX code
latex = r"""
\begin{table}[ht]
\centering
\resizebox{0.9\textwidth}{!}{%
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