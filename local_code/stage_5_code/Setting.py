'''
Concrete SettingModule class for a specific experimental SettingModule
'''



from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class Setting_Train_Test(setting):
    fold = 3

    def prepare(self, sDataset, sMethod, sResult, sEvaluateAcc
                , sEvaluateF1None, sEvaluateF1Macro, sEvaluateF1Micro, sEvaluateF1Weighted
                ):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluateAcc
        self.evaluate_f1_none = sEvaluateF1None
        self.evaluate_f1_macro = sEvaluateF1Macro
        self.evaluate_f1_micro = sEvaluateF1Micro
        self.evaluate_f1_weighted = sEvaluateF1Weighted
    
    def load_run_save_evaluate(self):

        
        
        # load dataset
        loaded_data = self.dataset.load()


        # just give it access to literally everything, it seems like it needs literally everything
        self.method.data = loaded_data
        learned_result = self.method.run()


        self.result.data = learned_result
        self.result.save()
            

        self.evaluate.data = learned_result
        self.evaluate_f1_none.data = learned_result
        self.evaluate_f1_macro.data = learned_result
        self.evaluate_f1_micro.data = learned_result
        self.evaluate_f1_weighted.data = learned_result


        return self.evaluate.evaluate(), self.evaluate_f1_none.evaluate(), self.evaluate_f1_macro.evaluate(), self.evaluate_f1_micro.evaluate(), self.evaluate_f1_weighted.evaluate()

    