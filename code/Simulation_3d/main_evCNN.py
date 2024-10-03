# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:33:47 2023

@author: andres cremades botella

File for evaluating the results of the CNN
"""
import ann_config as ann
import get_data_fun as gd


start = 7000 
end = 7010
step = 1   
dy = 1
dz = 1
dx = 1

# Load the trained CNN model
CNN = ann.convolutional_residual()
CNN.load_model()

# Make a prediction using the trained model
CNN.pred_rms(start,end,step=step,down_y=dy,down_z=dz,down_x=dx)

# Get the original data for reference
normdata = gd.get_data_norm() 
normdata.geom_param(start,dy,dz,dx)
normdata.read_Urms()

# Save the predicted rms values
CNN.saverms()

# Plot the wall-normalized value of the rms in a log and linear axis
CNN.plotrms_sim(normdata)
CNN.plotrms_simlin(normdata)

# Calculate the Mean Relative Error of the velocities respect to the simulation
# u_error = sum(vol_i* abs(u_pred-u_sim)/max(u_sim)) / vol_total
CNN.mre_pred(normdata,start,end,step)

# Save mre values
CNN.savemre()

print('End: main_evCNN.py')