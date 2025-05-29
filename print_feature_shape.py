from load_data_custom_cslu import load_data_custom_cslu
import numpy as np
data = load_data_custom_cslu("data_cslu_splits/all/data/spontaneous/all_ages_all_genders", mode="test")
print(data)
test_data = data['test']
entry = np.array(test_data[0]['input_features'])
print("done printing feature shapes of all spontaneaus data entries")

def print_ascii_heatmap(arr, rows=20, cols=60):
    import numpy as np

    # Downsample array to desired rows x cols for visibility
    arr_small = arr.reshape(rows, arr.shape[0]//rows, cols, arr.shape[1]//cols).mean(axis=(1,3))

    # Normalize to 0-9 scale
    norm = (arr_small - arr_small.min()) / (arr_small.ptp() + 1e-6)
    scaled = (norm * 9).astype(int)

    chars = " .:-=+*#%@"
    for row in scaled:
        print("".join(chars[val] for val in row))

# Example usage
print_ascii_heatmap(entry)