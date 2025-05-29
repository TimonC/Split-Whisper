from load_data_custom_cslu import load_data_custom_cslu
import numpy as np
data = load_data_custom_cslu("data_cslu_splits/all/data/spontaneous/all_ages_all_genders", mode="test")
print(data)
test_data = data['test']
for entry_idx in range(len(test_data)):
    print(np.array(test_data[entry_idx]['input_features']).shape)
print("done printing feature shapes of all spontaneaus data entries")