# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:25:58 2017

@author: alef1

PCA -- Adaptativa

X0 -------  O -- Y0
            |
X1 -------  O -- Y1
            |
Xp -------  O -- Ym

yi (n) = somatorio de i=0 ate p de { wij(n)*Xi(n) + somatorio de l<j Ulj(n)*Y1(n) }

aprendizado de HEBB --> delta wij  = n*Xi(n)*Yj(n) i de 0 a p e j de 0 a m

delta Ulj(n) = -u*Yl(n)*Yj(n) l < j
"""
import random
import numpy as np
from random import randint
from random import randrange, uniform

def main():
    #taxa de aprendizado
    eta = 0.1
    eta2 = 0.3
    
    ref_arquivo = open("iris.txt","r")
    dados     = []
    mat_saida = []
    for linha in ref_arquivo:
       valores = linha.split(",")
       vet_amostra = []
       vet_saida   = []
       for am in range(len(valores)-1):
           vet_amostra.append(float(valores[am]))
       if(valores[len(valores)-1] == 'Iris-setosa\n'):
           vet_saida.append(0)
           vet_saida.append(0)
           vet_saida.append(1)
       if(valores[len(valores)-1] == 'Iris-versicolor\n'):
           vet_saida.append(0)
           vet_saida.append(1)
           vet_saida.append(0)
       if(valores[len(valores)-1] == 'Iris-virginica\n' or len(valores[len(valores)-1]) == 14):
           vet_saida.append(1)
           vet_saida.append(0)
           vet_saida.append(0)
       dados.append(vet_amostra)
       mat_saida.append(vet_saida)
    ref_arquivo.close()
    mat_entrada = []
    mat_saida_2 = []
    lista = random.sample(range(0,len(dados)), len(dados))
    for i in range(len(dados)):
        mat_entrada.append(dados[lista[i]])
        mat_saida_2.append(mat_saida[lista[i]]) 
    mat_entrada = []
    mat_saida_2 = []
    lista = random.sample(range(0,len(dados)), len(dados))
    for i in range(len(dados)):
        mat_entrada.append(dados[lista[i]])
        mat_saida_2.append(mat_saida[lista[i]]) 
    w = []
    for i in range(len(mat_entrada)):
        c = []
        for j in range(len(mat_entrada[i])):
            c.append(uniform(0, 1))
        w.append(c)
        
    u = []
    for i in range(len(mat_entrada)):
        c = []
        for j in range(len(mat_entrada[i])):
            c.append(uniform(0, 1))
        u.append(c)
        
    Y = []
    for j in range(len(mat_entrada)):
        Y.append(0)
    n = 0
    while(n < 2):
        for i in range(len(mat_entrada)):
            #print("epoca: ",i)
            for j in range(len(mat_entrada[i])): 
                somat = 0
                result = 0
                l = 0
                while(l < j):
                    result = result + u[l][j]*Y[l]
                    l = l + 1
                somat = somat +  w[i][j]*mat_entrada[i][j] + result
            Y[i] = somat
            for j in range(len(mat_entrada[i])): 
                w[i][j] = eta*mat_entrada[i][j]*Y[j]
                l = 0
                while(l < j):
                    u[l][j] = -eta2*Y[l]*Y[j]
                    l = l + 1
        n = n + 1
        

    print(Y)
            
            
   #aprendizado de HEBB --> delta wij  = n*Xi(n)*Yj(n) i de 0 a p e j de 0 a m

   #delta Ulj(n) = -u*Yl(n)*Yj(n) l < j         
            
        
    
if __name__ == "__main__":
    main()