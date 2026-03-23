def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
b_Linear=[]
lable_Linear = []
RMSE_list=[]
MAE_list=[]
SMAPE_list=[]
para_list = []
my_net.eval()
with torch.no_grad():
    for data in test_loader:
        predict_value,param= my_net(data)  # [B, N, 1, D]
        b_Linear.append(predict_value)
        lable_Linear.append(data["flow_y"])
        para_list.append(param)
        loss_RMSE = criterion(predict_value*norm, data["flow_y"]*norm)
        loss_MAE = nn.L1Loss()(predict_value*norm, data["flow_y"]*norm)
        loss_Smape = smape(predict_value.cpu().detach().numpy()*norm, data["flow_y"].cpu().detach().numpy()*norm)
        RMSE_list.append(loss_RMSE)
        MAE_list.append(loss_MAE)
        SMAPE_list.append(loss_Smape)
    RMSE = sum(RMSE_list)/len(RMSE_list)
    MAE = sum(MAE_list)/len(MAE_list)
    Smape = sum(SMAPE_list)/len(SMAPE_list)
    print("Test RMSE: {:02.8f},Test MAE: {:02.8f},Test SMAPE: {:02.8f}".format(RMSE.sqrt(),MAE,Smape)) 
