'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test(setting):
    fold = 3
    
    def prepare(self, sTrainDataset, sTestDatatset, sMethod, sResult, sEvaluate):
        self.dataset = sTrainDataset
        self.testdataset = sTestDatatset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        X_train, y_train = loaded_data['X'], loaded_data['y']
        
        loaded_data = self.testdataset.load()
        X_test, y_test = loaded_data['X'], loaded_data['y']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        