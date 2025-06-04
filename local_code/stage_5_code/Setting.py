'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Note - copied from stage 3, still need to modify

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class Setting_Train_Test(setting):
    fold = 3

    def prepare(self, sDataset, sMethod, sResult, sEvaluateAcc
                #, sEvaluateF1None, sEvaluateF1Macro, sEvaluateF1Micro, sEvaluateF1Weighted
                ):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluateAcc
        # self.evaluate_f1_none = sEvaluateF1None
        # self.evaluate_f1_macro = sEvaluateF1Macro
        # self.evaluate_f1_micro = sEvaluateF1Micro
        # self.evaluate_f1_weighted = sEvaluateF1Weighted
    
    def load_run_save_evaluate(self):

        # commented out is what was left from stage 3
        
        # load dataset
        loaded_data = self.dataset.load()

        # In class, make sure to ask whether or not we use validation
        # Especially since paper seems to use it???
        # idx_train = range(20)
        
        # idx_test = range(20)
        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        # if self.dataset.dataset_name == 'cora':





        #     #idx_val = range(1200, 1500)
        # elif self.dataset.dataset_name == 'citeseer':
        #     idx_train = range(120)
        #     idx_test = range(120, 1500)
        #     #idx_val = range(1200, 1500)
        # elif self.dataset.dataset_name == 'pubmed':
        #     idx_train = range(60)
        #     idx_test = range(60, 7300)
        #     #idx_val = range(6000, 6300)
        # #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        # elif self.dataset.dataset_name == 'cora-small':
        #     idx_train = range(5)
        #     #idx_val = range(5, 10)
        #     idx_test = range(5, 10)

        
        # # https://stackoverflow.com/questions/77881247/pytorch-conv2d-error-expected-3-channels-got-128-for-1-128-128-3-help-n
        # # (N, C, H, W) or (C, H, W)
        
        # X_train = []
        # y_train = []
        # for instance in loaded_data['train']:
        #     X_train.append([instance['image']])
        #     y_train.append(instance['label'])

        # X_test = []
        # y_test = []
        # for instance in loaded_data['test']:
        #     X_test.append([instance['image']])
        #     y_test.append(instance['label'])

       

        # # get the training nodes/testing nodes
        # train_x = loaded_data['X'][idx_train]
        # #val_x = loaded_data[features[idx_val]
        # test_x = loaded_data['X'][idx_test]
        # train_y = loaded_data['y'][idx_train]
        # test_y = loaded_data['y'][idx_test]

        # print(train_x, train_y, test_x, test_y)


        #print("PRITNING OUTPUT")
        #output = self.method(loaded_data['X'], loaded_data['utility']['A'])
        #print(output)

        # run MethodModule - not entirely sure what this is for?
        # note - may need to modify this
        # first see what train/test inside method needs access to since that's how the train funciton is called
        # it needs all features, 
        # self.method.data = {'train': {'X': train_x, 'y': train_y}, 'test': {'X': test_x, 'y': test_y}, 'adj': loaded_data['utility']['A'], 'features': loaded_data['X']
        #                     , 'idx_train': loaded_data}

        # acutally just give it access to literally everything, it seems like it needs literally everything
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule - also not entirely sure what this code is doing, come back to it
        self.result.data = learned_result
        self.result.save()
            

        # come back to this evaluate stuff
        self.evaluate.data = learned_result
        # self.evaluate_f1_none.data = learned_result
        # self.evaluate_f1_macro.data = learned_result
        # self.evaluate_f1_micro.data = learned_result
        # self.evaluate_f1_weighted.data = learned_result


        return self.evaluate.evaluate(), None
        # return self.evaluate.evaluate(), self.evaluate_f1_none.evaluate(), self.evaluate_f1_macro.evaluate(), self.evaluate_f1_micro.evaluate(), self.evaluate_f1_weighted.evaluate()


       

        # maybe have run MethodModule and raw ResultModule (like in prev stages here???)
        # Come back to this


        # or instead return accuracy parameters?
        # remove this line, not sure why it was here (or added)
        return train_test_val
    