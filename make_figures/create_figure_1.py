import json
import os
import matplotlib.pyplot as plt

# Folder containing JSON result files
results_folder = "results/classifier"

# Files to load
tasks = ['age', 'gender', 'both']

# Data containers
loss_data = {}
accuracy_data = {}

for task in tasks:
    json_path = os.path.join(results_folder, f"{task}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    loss_data[task] = data['loss']
    
    # Select accuracy depending on task
    if task == 'age':
        accuracy_data[task] = data['accuracy_age']
    elif task == 'gender':
        accuracy_data[task] = data['accuracy_gender']
    else:  # both
        accuracy_data[task] = data['accuracy_joint']

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Loss
for task in tasks:
    axs[0].plot(loss_data[task], label=task)
axs[0].set_ylabel('𝐿')
axs[0].set_title('Training Loss per Epoch')
axs[0].legend()
axs[0].grid(True)

# Plot Accuracy
for task in tasks:
    axs[1].plot(accuracy_data[task], label=task)
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_title('Accuracy per Epoch')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()