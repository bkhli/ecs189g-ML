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
        
        # https://stackoverflow.com/questions/77881247/pytorch-conv2d-error-expected-3-channels-got-128-for-1-128-128-3-help-n
        # (N, C, H, W) or (C, H, W)
        
        X_train = []
        y_train = []
        for instance in loaded_data['train']:
            X_train.append([instance['image']])
            y_train.append(instance['label'])

        X_test = []
        y_test = []
        for instance in loaded_data['test']:
            X_test.append([instance['image']])
            y_test.append(instance['label'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

