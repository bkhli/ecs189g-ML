"""
Concrete SettingModule class for a specific experimental SettingModule
"""

import numpy as np
from sklearn.model_selection import train_test_split

from local_code.base_class.setting import setting
from local_code.stage_4_code.Method_Generation import Method_RNN

# from local_code.stage_4_code.Method_Generation_GRU import Method_GRU
from local_code.stage_4_code.Method_Generation_LSTM import Method_LSTM


class Setting_Train_Test(setting):
    fold = 3

    # def prepare(self, sDataset, sResult, sEvaluateAcc, sEvaluateF1None, sEvaluateF1Macro, sEvaluateF1Micro, sEvaluateF1Weighted):
    def prepare(self, sDataset, sResult, sEvaluateAcc):
        self.dataset = sDataset
        self.result = sResult
        self.evaluate = sEvaluateAcc
        # self.evaluate_f1_none = sEvaluateF1None
        # self.evaluate_f1_macro = sEvaluateF1Macro
        # self.evaluate_f1_micro = sEvaluateF1Micro
        # self.evaluate_f1_weighted = sEvaluateF1Weighted

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data, vocab = self.dataset.load()
        # print("[DEBUG] vocab from dataset.load():", vocab)
        # print("[DEBUG] vocab keys sample:", list(vocab.get_stoi().items())[:5] if vocab else "N/A")

        # Uncomment below to change with method is used

        method = "LSTM"
        match method:
            case "RNN":
                method = Method_RNN("RNN", "RNN sentiment classification", vocab)
            # case "GRU":
            #     method = Method_GRU("GRU", "GRU sentiment classification", vocab)
            case "LSTM":
                method = Method_LSTM("LSTM", "LSTM sentiment classification", vocab)
            case _:
                raise Exception("Invalid model passed in Setting_Classifciation.py")

        X_train = loaded_data["X"]
        y_train = loaded_data["y"]

        # run MethodModule
        method.data = {
            "X": X_train,
            "y": y_train,
            # "train": {"X": X_train, "y": y_train},
            # "test": {"X": X_test, "y": y_test},
        }
        generations = method.run()

        # save raw ResultModule
        # self.result.data = learned_result
        # self.result.save()
        #
        # self.evaluate.data = learned_result

        # self.evaluate_f1_none.data = learned_result
        # self.evaluate_f1_macro.data = learned_result
        # self.evaluate_f1_micro.data = learned_result
        # self.evaluate_f1_weighted.data = learned_result

        return generations
        # return self.evaluate.evaluate()
        # , self.evaluate_f1_none.evaluate(), self.evaluate_f1_macro.evaluate(), self.evaluate_f1_micro.evaluate(), self.evaluate_f1_weighted.evaluate()

    def print_setup_summary(self):
        print(
            "dataset:",
            self.dataset.dataset_name,
            ", method: RNN",
            ", setting:",
            self.setting_name,
            ", result:",
            self.result.result_name,
            ", evaluation:",
            self.evaluate.evaluate_name,
        )
