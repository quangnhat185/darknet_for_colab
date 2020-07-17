# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:34:58 2020

@author: Quang Nguyen
"""

import os
import shutil
from yolov4_config import *

filters=(classes+5)*3
 
VARIABLE_LIST = [classes, max_batches, batch, subdivisions, width, 
                 height, channels, momentum, decay, learning_rate,
                 steps,scales, filters]

DICT_ORIGIN = {"classes":80, 
               "max_batches":8000,
               "batch":64,
               "subdivisions":16,
               "width":608,
               "height":608,
               "channels":3,
               "momentum":0.949,
               "decay":0.0005,
               "learning_rate":0.001,
               "steps": (6400,7200),
               "scales":(.1,.1),
               "filters":255}

def removefile(path):
    if os.path.isfile(path):
        os.remove(path)

def generate_custom_train(CFG_ORIGIN_PATH, CFG_FOLDER_PATH, CFG_TRAIN_FILE):
    file = open(CFG_ORIGIN_PATH + CFG_TRAIN_FILE, "rt")
    yolov4_train = file.read()
    for i, (key, value) in enumerate(DICT_ORIGIN.items()):
        if key=="steps" or key=="scales":
            processed_variable = str(VARIABLE_LIST[i]).replace("(","").replace(")","").replace(" ","")
            processed_value = str(value).replace("(","").replace(")","").replace(" ","")            
            #print(key+'='+ processed_value,key+"={:s}".format(processed_variable))
            yolov4_train = yolov4_train.replace(key+'='+ processed_value,key+"={}".format(processed_variable))
            
        else:
            yolov4_train = yolov4_train.replace(key+'='+str(value),key+"={}".format(VARIABLE_LIST[i]))
            #print(key+'='+str(value),key+"={}".format(VARIABLE_LIST[i]))    
    file.close()

    new_file = open(CFG_FOLDER_PATH + CFG_TRAIN_FILE, "wt")
    new_file.write(yolov4_train)
    new_file.close()
    print("[INFO] Generating {} successfully...".format(CFG_TRAIN_FILE))
    
def generate_custom_test(CFG_ORIGIN_PATH, CFG_FOLDER_PATH, CFG_TEST_FILE):
    file = open(CFG_ORIGIN_PATH + CFG_TEST_FILE, "rt")
    yolov4_test = file.read()
    for i, (key, value) in enumerate(DICT_ORIGIN.items()):
        if key=="batch" or key=="subdivisions":
            continue
        elif key=="steps" or key=="scales":
            processed_variable = str(VARIABLE_LIST[i]).replace("(","").replace(")","").replace(" ","")
            processed_value = str(value).replace("(","").replace(")","").replace(" ","")            
            #print(key+'='+ processed_value,key+"={:s}".format(processed_variable))
            yolov4_test = yolov4_test.replace(key+'='+ processed_value,key+"={}".format(processed_variable))
            
        else:
            yolov4_test = yolov4_test.replace(key+'='+str(value),key+"={}".format(VARIABLE_LIST[i]))
            #print(key+'='+str(value),key+"={}".format(VARIABLE_LIST[i]))    
    file.close()

    new_file = open(CFG_FOLDER_PATH + CFG_TEST_FILE, "wt")
    new_file.write(yolov4_test)
    new_file.close()
    print("[INFO] Generating {} successfully...".format(CFG_TEST_FILE))      
        
if __name__=="__main__" :
    CFG_ORIGIN_PATH='./cfg/origin/'
    CFG_TRAIN_FILE='yolov4_custom_train.cfg'
    CFG_FOLDER_PATH='./cfg/'
    CFG_TEST_FILE='yolov4_custom_test.cfg'
    
    removefile(CFG_FOLDER_PATH + CFG_TRAIN_FILE)
    removefile(CFG_FOLDER_PATH + CFG_TEST_FILE)
    
    generate_custom_train(CFG_ORIGIN_PATH,CFG_FOLDER_PATH,CFG_TRAIN_FILE)
    generate_custom_test(CFG_ORIGIN_PATH,CFG_FOLDER_PATH,CFG_TEST_FILE)
    
