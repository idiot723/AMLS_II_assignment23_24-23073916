from task.functions import get_train, train_model, train_vit_model,train_snapmix_model, plot_accloss, plot_2accloss
import pandas as pd
from matplotlib import pyplot as plt

def main():
    #load data
    train = get_train()
    
    #train and validation
    # train_model(train)
    # train_vit_model(train)
    # train_snapmix_model(train)

    #plot using saved csv files
    #resnext with mixup plot
    plot_accloss()
    #resnext vs vit plot
    plot_2accloss('./task/nomix10_results.csv','./task/vit_results.csv','ResNeXt','Vit','./task/vs_vit')
    #resnext vs with mixup plot
    plot_2accloss('./task/nomix_results.csv','./task/validation_results2.csv','ResNeXt','ResNeXt with Mixup','./task/vs_nomix')
    #resnext vs with snapmix plot
    plot_2accloss("./task/nomix10_results.csv",'./task/resnext_snapmix2.csv','ResNeXt','ResNeXt with Snapmix','./task/vs_snapmix')
    
    train_model(train)
    train_vit_model(train)
    train_snapmix_model(train)

main()