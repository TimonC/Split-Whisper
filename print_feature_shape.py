from load_data_custom_cslu import load_data_custom_cslu
import numpy as np
data = load_data_custom_cslu("data_cslu_splits/all/data/spontaneous/all_ages_all_genders", mode="test")
print(data)
test_data = data['test']
print(np.array(test_data[0]['input_features']).shape)