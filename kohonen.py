# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:19:45 2017

@author: alef1


rede kohonen
"""

#entradas com n dimensões
# v1, v2, v3 ... vn

#cada no com um peso
# w1, w2, w3 ... wn

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.misc import toimage

def Eucli_dists(MAP,x):
    x = x.reshape((1,1,-1))
    Eucli_MAP = MAP - x
    Eucli_MAP = Eucli_MAP**2
    Eucli_MAP = np.sqrt(np.sum(Eucli_MAP,2))
    return Eucli_MAP


def main():
    
    input_dimensions = 13
    map_width = 7
    map_height = 5
    radius0 = max(map_width,map_height)/2
    learning_rate0 = 0.1
    
    epochs = 5000
    radius=radius0
    learning_rate = learning_rate0
    BMU = np.zeros([2],dtype=np.int32)
    timestep=1
    e=0.001 
    flag=0
    epoch=0

    patterns = []
    classes = []
    #carregando o arquivo vizinhos.txt
    file = open('vizinhos.txt','r')
    for line in file.readlines():
        row = line.strip().split(',')
        patterns.append(row[1:14])
        classes.append(row[0])
    file.close
    
    patterns = np.asarray(patterns,dtype=np.float32)
    max_iterations = epochs*len(patterns)
    too_many_iterations = 10*max_iterations
    
    MAP = np.random.uniform(size=(map_height,map_width,input_dimensions))
    prev_MAP = np.zeros((map_height,map_width,input_dimensions))

    result_map = np.zeros([map_height,map_width,3],dtype=np.float32)

    coordinate_map = np.zeros([map_height,map_width,2],dtype=np.int32)

    for i in range(map_height):
        for j in range(map_width):
            coordinate_map[i][j] = [i,j]
            
    while (epoch <= epochs):
    
        shuffle = random.sample(list(np.arange(0,len(patterns),1,'int')),len(patterns))
        
        for i in range(len(patterns)):
        
            J = np.sqrt(np.sum(np.sum((prev_MAP-MAP)**2,2)))
            if  J<= e: 
                flag=1
                break         
            else:
            
                if timestep == max_iterations and timestep != too_many_iterations:
                    epochs += 1
                    max_iterations = epochs*len(patterns)
            
                pattern = patterns[shuffle[i]]
                Eucli_MAP = Eucli_dists(MAP,pattern)
        
                BMU[0] = np.argmin(np.amin(Eucli_MAP,1),0)
                BMU[1] = np.argmin(Eucli_MAP,1)[int(BMU[0])]
    
                Eucli_from_BMU = Eucli_dists(coordinate_map,BMU)  
        
                prev_MAP = np.copy(MAP)
            
                for i in range(map_height):
                    for j in range(map_width):
                        distance = Eucli_from_BMU[i][j]
                        if distance <= radius:
                            theta = math.exp(-(distance**2)/(2*(radius**2)))
                            MAP[i][j] = MAP[i][j] + theta*learning_rate*(pattern-MAP[i][j])
            
                learning_rate = learning_rate0*math.exp(-(timestep)/max_iterations)
                time_constant = max_iterations/math.log(radius) 
                radius = radius0*math.exp(-(timestep)/time_constant)
                timestep+=1
            
        if flag==1:
            break
        epoch+=1    
     
    #visualização
    i=0
    for pattern in patterns:
    
        Eucli_MAP = Eucli_dists(MAP,pattern)
    
        BMU[0] = np.argmin(np.amin(Eucli_MAP,1),0)
        BMU[1] = np.argmin(Eucli_MAP,1)[int(BMU[0])]
        x = BMU[0]
        y = BMU[1]
    
        if classes[i] == '1':
            if result_map[x][y][0] <= 0.5:
                result_map[x][y] += np.asarray([0.5,0,0])
        elif classes[i] == '2':
            if result_map[x][y][1] <= 0.5:
                result_map[x][y] += np.asarray([0,0.5,0])
        elif classes[i] == '3':
            if result_map[x][y][2] <= 0.5:
                result_map[x][y] += np.asarray([0,0,0.5])
        i+=1
    result_map = np.flip(result_map,0)

    print ("\nRed = Iris-Setosa")
    print ("Green = Iris-Virginica")
    print ("Blue = Iris-Versicolor\n")

    plt.imshow(toimage(result_map),interpolation='nearest')
    
if __name__ == "__main__":
    main()