from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_MLP import GCN
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting import Setting_Train_Test
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy



import numpy as np
import torch
import torch.onnx

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('', '')
    # Note - might need to change paths, but it seems like it should be this?
    data_obj.dataset_source_folder_path = 'data/stage_5_data/pubmed'
    data_obj.dataset_name = 'pubmed'

    # Come back to this
    # Oh, this is how it passes into the method function
    method_obj = GCN(nfeat=500,
            nhid=16,
            nclass=3,
            dropout=0.5)

    # probably don't need to change anything?
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_5_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train and test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    