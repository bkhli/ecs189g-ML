'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test(setting):
    fold = 3

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        
        rng = np.random.default_rng()
        # rng.permutation(np.arange(9).reshape((3,3)), axis=0)
        
        # https://stackoverflow.com/questions/77881247/pytorch-conv2d-error-expected-3-channels-got-128-for-1-128-128-3-help-n
        # (N, C, H, W) or (C, H, W)
        
        print(np.shape(loaded_data['train'][0]['image']))
        print(np.transpose(loaded_data['train'][0]['image'], (2, 0, 1)).shape)
        # get channel out

        # print(loaded_data['train'][0]['image'])
        train_data = rng.permutation(loaded_data['train'], axis=0)
        test_data = rng.permutation(loaded_data['test'], axis=0)
        # print(train_data[0]['image'])

        X_train = []
        y_train = []
        for instance in train_data:
            image = np.transpose(instance['image'], (2, 0, 1))
            X_train.append(image)
            y_train.append(instance['label'])

        X_test = []
        y_test = []
        for instance in test_data:
            image = np.transpose(instance['image'], (2, 0, 1))
            X_test.append(image)
            y_test.append(instance['label'])
        
        print("X_train shape: ", np.shape(X_train))
        print("X_test shape: ", np.shape(X_test))

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

