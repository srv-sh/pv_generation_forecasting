from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math, random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def flip_from_probability(p):
    return True if random.random() < p else False

def transformer(dataloader, EPOCH, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):
    EPOCH =10
    device = torch.device(device)

    model = Transformer().float().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        for _input, target in dataloader:
        
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            print(_input.shape,'___input shape____')
            print(target.shape,'-----target-----')

            optimizer.zero_grad()
            src =torch.squeeze(_input,0)  # torch.Size([24, 1, 7])
            target = torch.squeeze(target,0) # src shifted by 1.
            sampled_src = src[:1, :, :] #t0 torch.Size([1, 1, 7])
            # print(src.shape,"____src shape____")
            # print(target.shape,"____target shape____")
            # print(sampled_src.shape,"____sample shape___")
            # # input("!!!!!!!!!")
            print(len(target))
            for i in range(len(target)):
                print(sampled_src.shape)
                prediction = model(sampled_src.float(), device) # torch.Size([1xw, 1, 1])
                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                if i < 24: # One day, enough data to make inferences about cycles
                    prob_true_val = True
                else:
                    ## coin flip
                    v = k/(k+math.exp(epoch/k)) # probability of heads/tails depends on the epoch, evolves with time.
                    prob_true_val = flip_from_probability(v) # starts with over 95 % probability of true val for each flip in epoch 0.
                    ## if using true value as new value

                if prob_true_val: # Using true value as next value
                    sampled_src = torch.cat((sampled_src.detach(), src[i+1, :, :].unsqueeze(0).detach()))
                else: ## using prediction as new value
                    positional_encodings_new_val = src[i+1,:,1:].unsqueeze(0)
                    predicted_humidity = torch.cat((prediction[-1,:,:].unsqueeze(0), positional_encodings_new_val), dim=2)
                    sampled_src = torch.cat((sampled_src.detach(), predicted_humidity.detach()))
            
            """To update model after each sequence"""
            loss = criterion(target, prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            mse = mean_squared_error(target[:,0,0].numpy(), prediction[:,0,0].numpy())
            print("Mean square error : " + str(mse))


            error = mae(target[:,0,0].numpy(), prediction[:,0,0].numpy())
            print("Mean absolute error : " + str(error))


            r2 = r2_score(target[:,0,0].numpy(), prediction[:,0,0].numpy())
            print('r2 score for perfect model is', r2)



        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"


        if epoch % 2 == 0: # Plot 1-Step Predictions

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler_feature = load('scalar_feature.joblib')
            scaler_target = load('scalar_target.joblib')

            sampled_src_inverter_hr_mean = scaler_feature.inverse_transform(sampled_src[:,:,-1].cpu()) #torch.Size([35, 1, 7])
            src_inverter_hr_mean = scaler_feature.inverse_transform(src[:,:,-1].cpu()) #torch.Size([35, 1, 7])
            target_inverter_hr_mean = scaler_target.inverse_transform(target.cpu()) #torch.Size([35, 1, 7])
            prediction_inverter_hr_mean = scaler_target.inverse_transform(prediction.detach().cpu().numpy()) #torch.Size([35, 1, 7])
            plot_training_3(epoch, path_to_save_predictions, src_inverter_hr_mean, sampled_src_inverter_hr_mean, prediction_inverter_hr_mean)

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    plot_loss(path_to_save_loss, train=True)
    return best_model