from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Classification import Setting_Train_Test
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Evaluate_F1 import Evaluate_F1_None, Evaluate_F1_Macro, Evaluate_F1_Micro, Evaluate_F1_Weighted
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('Movie Reviews and Ratings', '')
    data_obj.dataset_source_folder_path = 'data/stage_4_data/text_classification/'

    # MOVED METHOD OBJECT INTO SETTING OBJECT - method_obj = Method_MLP('RNN', '')
    # Needed to load data before initializing model (embeddings related, vocab size)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_4_result/classification'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train and test', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj_acc = Evaluate_Accuracy('accuracy', '')
    # evaluate_obj_f1_none = Evaluate_F1_None('multilabel classification', '')
    # evaluate_obj_f1_macro = Evaluate_F1_Macro('f1 - macro', '')
    # evaluate_obj_f1_micro = Evaluate_F1_Micro('f1 - micro', '')
    # evaluate_obj_f1_weighted = Evaluate_F1_Weighted('f1 - weighted', '')
    # ------------------------------------------------------

    
    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, result_obj, evaluate_obj_acc)
    # setting_obj.prepare(data_obj, result_obj, evaluate_obj_acc, evaluate_obj_f1_none,
    #                     evaluate_obj_f1_macro, evaluate_obj_f1_micro, evaluate_obj_f1_weighted)
    setting_obj.print_setup_summary()
    mean_score, f1_none, f1_macro, f1_micro, f1_weighted = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(
        f"\nMLP Test Accuracy: {mean_score:.4f}\n"
        # f"F1 Scores:\n"
        # f"\tIndividual: {[f'{score:.4f}' for score in f1_none]}\n"
        # f"\tMacro:     {f1_macro:.4f}\n"
        # f"\tMicro:     {f1_micro:.4f}\n"
        # f"\tWeighted:  {f1_weighted:.4f}"
    )
    print('************ Finish ************')
    # ------------------------------------------------------
    

