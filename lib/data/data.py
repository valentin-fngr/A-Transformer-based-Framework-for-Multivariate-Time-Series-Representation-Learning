import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset
         

def get_data_and_preprocess(
        csv_path, 
        target, 
        timesteps, 
        train_split, 
        val_split
): 
    data = pd.read_csv(csv_path)
    data.shape[1] - 1

    # placeholders
    X = np.zeros((len(data), timesteps, data.shape[1]-1))
    y = np.zeros((len(data), timesteps, 1))

    # fill X : 
    # for each time serie
    for i, name in enumerate(list(data.columns[:-1])):
        # for each timestep
        for j in range(timesteps):
            X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

    
    for j in range(timesteps):
       y[:, j, 0] = data[target].shift(timesteps - j - 1).fillna(method="bfill")

    prediction_horizon = 1
    # The prediction horizon is everything that comes after the timestamps. 
    # If you are using [t=1, ..., t=10] as input, your target will be t=11. 
    target = data[target].shift(-prediction_horizon).fillna(method="ffill").values # (num_sample, 1)
    
    train_length = int(len(data) * train_split)
    val_length = int(len(data) * val_split)

    X_train = X[:train_length] # (num_sample, T, n)
    y_his_train = y[:train_length] # (num_sample, T, 1)
    X_val = X[train_length:train_length+val_length]
    y_his_val = y[train_length:train_length+val_length]
    X_test = X[train_length + val_length:]
    y_his_test = y[train_length + val_length:]
    target_train = target[:train_length]
    target_val = target[train_length:train_length+val_length]
    target_test = target[train_length + val_length:]

    # min max scaling 
    X_train_mean= X_train.mean(axis=(0, 1))[None, None, :] # (1, 1, n)
    X_train_std = X_train.std(axis=(0, 1))[None, None, :] # (1, 1, n)
    y_his_train_mean = y_his_train.mean(axis=(0, 1))[None, :] # (T, 1)
    y_his_train_std = y_his_train.std(axis=(0, 1))[None, :] # (T, 1)
    target_train_mean = target_train.mean(axis=0) # scalar
    target_train_std = target_train.std(axis=0) # scalar

    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    y_his_train = (y_his_train - y_his_train_mean) / y_his_train_std
    y_his_val = (y_his_val - y_his_train_mean) / y_his_train_std
    y_his_test = (y_his_test - y_his_train_mean) / y_his_train_std

    target_train = (target_train - target_train_mean) / target_train_std
    target_val = (target_val - target_train_mean) / target_train_std
    target_test = (target_test - target_train_mean) / target_train_std

    X_train_t = X_train
    X_val_t = X_val
    X_test_t = X_test
    y_his_train_t = y_his_train
    y_his_val_t = y_his_val
    y_his_test_t = y_his_test
    target_train_t = target_train
    target_val_t = target_val
    target_test_t = target_test

    return [
        X_train_t,
        X_val_t,
        X_test_t,
        y_his_train_t,
        y_his_val_t,
        y_his_test_t,
        target_train_t,
        target_val_t,
        target_test_t
    ]


def get_data_and_preprocess_unsupervised(
        csv_path, 
        target, 
        timesteps, 
        train_split, 
        val_split
): 
    data = pd.read_csv(csv_path)
    data.shape[1] - 1

    # placeholders
    X = np.zeros((len(data), timesteps, data.shape[1]-1))
    y = np.zeros((len(data), timesteps, 1))

    # fill X : 
    # for each time serie
    for i, name in enumerate(list(data.columns[:-1])):
        # for each timestep
        for j in range(timesteps):
            X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

    
    for j in range(timesteps):
       y[:, j, 0] = data[target].shift(timesteps - j - 1).fillna(method="bfill")

    
    train_length = int(len(data) * train_split)
    val_length = int(len(data) * val_split)

    # "To select hyperparameters, for each dataset we randomly
    # split the training set in two parts, 80%-20%, and used the 20% as
    # a validation set for hyperparameter tuning"

    shuffle_idx  = torch.randperm(train_length + val_length)
    X_train_val = X[:train_length + val_length][shuffle_idx]
    y_train_val = y[:train_length + val_length][shuffle_idx] 

    X_train = X_train_val[:train_length] # (num_sample, T, n)
    y_his_train = y_train_val[:train_length] # (num_sample, T, 1)
    X_val = X_train_val[train_length:train_length+val_length]
    y_his_val = y_train_val[train_length:train_length+val_length]
    X_test = X[train_length+val_length:]
    y_his_test = y[train_length+val_length:]

    # min max scaling 
    X_train_mean= X_train.mean(axis=(0, 1))[None, None, :] # (1, 1, n)
    X_train_std = X_train.std(axis=(0, 1))[None, None, :] # (1, 1, n)
    y_his_train_mean = y_his_train.mean(axis=(0, 1))[None, :] # (T, 1)
    y_his_train_std = y_his_train.std(axis=(0, 1))[None, :] # (T, 1)

    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    y_his_train = (y_his_train - y_his_train_mean) / y_his_train_std
    y_his_val = (y_his_val - y_his_train_mean) / y_his_train_std
    y_his_test = (y_his_test - y_his_train_mean) / y_his_train_std

    X_train_t = X_train
    X_val_t = X_val
    X_test_t = X_test
    y_his_train_t = y_his_train
    y_his_val_t = y_his_val
    y_his_test_t = y_his_test

    return [
        X_train_t,
        X_val_t,
        X_test_t,
        y_his_train_t,
        y_his_val_t,
        y_his_test_t
    ]





class DatasetUnsupervised(Dataset): 
    """
    A basic torch dataset for unsupervised training. 
    Each sample will be composed of exogenous series x and a mask of these features over time. 
    See the Unsupervisd section of the reference for detailed masking explanations.

    References 
    -----
        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning

    """

    def __init__(
        self, 
        X, 
        timesteps,
        y_known=None, 
        mask_r = 0.15, 
        lm = 3
    ):
        self.X = X  # (num_samples, w, m)
        self.lm = lm 
        self.lu = ((1-mask_r) / mask_r) * lm
        self.prob_mask_stop = 1/lm 
        self.prob_unmask_stop = (self.prob_mask_stop * mask_r) / (1 - mask_r)
        self.mask_r = mask_r
        self.timesteps = timesteps

        print(X.shape)

        if y_known: 
            self.X = np.concat([self.X, y_known], dim=2) # (num_samples, w, m + 1)

        
    def __getitem__(self, idx): 
        
        x = self.X[idx] #  (w, m)
        mask_idx = np.empty(x.shape) # (w, m)
        # apply mask for each feature over time  
        for n in range(x.shape[1]): 
            mask = self._get_mask(x[:, n]) # (w, )
            mask_idx[:, n] = mask

        masked_input = x * (1 - mask_idx)
        target = x.copy()
        return masked_input, target, mask_idx


    def _get_mask(self, x): 

        mask = np.ones_like(x) # (w, ) 
        
        is_mask = np.random.binomial(1, self.mask_r)
        for t in range(self.timesteps): 
            mask[t] = is_mask 

            prob_stop = self.prob_mask_stop if is_mask else self.prob_unmask_stop
            is_stop = np.random.binomial(1, prob_stop) 
            if is_stop: 
                is_mask = 1 - is_mask
                
        return mask
    
    def __len__(self): 
        # return 3000
        return len(self.X)

# X = np.random.rand(1, 30, 10) 
# target = np.random.rand(1,)

# dataset = DatasetUnsupervised(X, target, timesteps=30)

# for data in dataset: 
#     break