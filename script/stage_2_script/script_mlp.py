from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP

# to test different (earlier models), but had to change names so that it works
# from local_code.stage_2_code.Method_MLP_first_working import Method_MLP_first_working
# from local_code.stage_2_code.Method_MLP_four_layer import Method_MLP_four_layer
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Evaluate_F1 import Evaluate_F1_None, Evaluate_F1_Macro, Evaluate_F1_Micro, Evaluate_F1_Weighted
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    path_var = "" # "../../" or ""
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('mlp_train', '')
    data_obj.dataset_source_folder_path = path_var + 'data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'
    
    testdata_obj = Dataset_Loader('mlp_test', '')
    testdata_obj.dataset_source_folder_path = path_var + 'data/stage_2_data/'
    testdata_obj.dataset_source_file_name = 'test.csv'

    #method_obj = Method_MLP_four_layer('multi-layer perceptron', '')
    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = path_var + 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('Full Batch', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    #evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # maybe remove this later??
    evaluate_obj_acc = Evaluate_Accuracy('accuracy', '')
    evaluate_obj_f1_none = Evaluate_F1_None('multilabel classification', '')
    evaluate_obj_f1_macro = Evaluate_F1_Macro('f1 - macro', '')
    evaluate_obj_f1_micro = Evaluate_F1_Micro('f1 - micro', '')
    evaluate_obj_f1_weighted = Evaluate_F1_Weighted('f1 - weighted', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, testdata_obj, method_obj, result_obj, evaluate_obj_acc, evaluate_obj_f1_none,
                        evaluate_obj_f1_macro, evaluate_obj_f1_micro, evaluate_obj_f1_weighted)
    # print('Should be done training here???')
    setting_obj.print_setup_summary()
    mean_score, f1_none, f1_macro, f1_micro, f1_weighted = setting_obj.load_run_save_evaluate()
    # mean_score,
    print('************ Overall Performance ************')
    print(
        f"\nMLP Test Accuracy: {mean_score:.4f}\n"
        f"F1 Scores:\n"
        f"\tIndividual: {[f'{score:.4f}' for score in f1_none]}\n"
        f"\tMacro:     {f1_macro:.4f}\n"
        f"\tMicro:     {f1_micro:.4f}\n"
        f"\tWeighted:  {f1_weighted:.4f}"
    )
    # setting_obj.prepare(data_obj, testdata_obj, method_obj, result_obj, evaluate_obj_f1)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ F1 Performance ************')
    # print('F1 Score: ' + str(mean_score) + ' +/- ' + str(std_score))

    print('************ Finish ************')
    # ------------------------------------------------------
    

    
