# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:27:48 2017

@author: alef1
"""
#bibliotecas utilizadas
import math as m
import random as r
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import randrange, uniform

#implementação da mlp completa
#somatorio
def somatorio(entradas, pesos):
    soma = 0
    for i in range(len(pesos)):
            soma += entradas[i]*pesos[i]
           
    return soma
#f de net
def f_net(net):
    return (1/(1 + m.exp(-net)))
#derivada pronta
def derivada_f_net(net):
    return (net * (1 - net))
#função mat aleatoria
def mat_aleatoria(vmin = -1,vmax = 1,linhas = 2, colunas = 2):
    matriz = []
    for i in range(linhas):
        l = []
        for x in range(colunas):
            l.append(r.uniform(vmin,vmax))
        matriz.append(l)
    return matriz

def arquitetura(vet_arquitetura):
    mat_final = []
    for j in range(len(vet_arquitetura)-1):
        mat= mat_aleatoria(-1,1,vet_arquitetura[j+1],vet_arquitetura[j])
        mat_final.append(mat)
    return mat_final
    
    #forward
        #calcular e retornar todos os net e fnets
def forward(matriz_pesos, vet_arquitetura, vet_entrada, vet_saida):
    #arquitetura = 1ª camada 1 a N camadas ocultas camada saida
    #criar a arquitetura de net e fnets mat [[6 nets][3 nets][1 net]]
    mat_net = []
    mat_f_net = []
    #primeira interligação de camada
    #calculando os nets
    net = []
    for i in range(len(matriz_pesos[0])):
        net.append(somatorio(vet_entrada,matriz_pesos[0][i]))
    mat_net.append(net)
    #calculando os f de net
    vet_fnet = []
    for i in range(len(mat_net[0])):
        vet_fnet.append(f_net(mat_net[0][i]))
    mat_f_net.append(vet_fnet)
    #camadas posteriores
    for j in range(len(matriz_pesos)-1):
        net = []
        for k in range(len(matriz_pesos[j+1])):
            net.append(somatorio(mat_f_net[j],matriz_pesos[j+1][k]))
        mat_net.append(net)
        vet_fnet = []
        for f in range(len(mat_net[j+1])):
            vet_fnet.append(f_net(mat_net[j+1][f]))
        mat_f_net.append(vet_fnet)
        
    return [mat_net,mat_f_net]

def pcaAdaptativa(x):
   
    X = x
    
    # Padronização dos valores com media 0 e desvio padrão 1
    entrada = StandardScaler().fit_transform(X)
    
    #matriz de covariancia, poderia ser usada uma matriz de correlação, podem daria o mesmo valor 
    mat_covariancia = np.mean(entrada, axis=0)
    #calculo da covariancia 
    cov_mat = (entrada - mat_covariancia).T.dot((entrada - mat_covariancia)) / (entrada.shape[0]-1)
    cov_mat = np.cov(entrada.T)
    #auto valores e auto vetores
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    u,s,v = np.linalg.svd(entrada.T)
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    #aleatorizando os auto valores e auto vetores
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

   
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))

    Y = entrada.dot(matrix_w)
    print(Y)
    eta = 0.1
    eta2 = 0.3     
    #pesos entre as conexões    
    w = []
    for i in range(len(Y)):
        c = []
        for j in range(len(Y[i])):
            c.append(uniform(0, 1))
        w.append(c)
    #pesos laterais   das saidas ligadas as outras saídas
    u = []
    for i in range(len(Y)):
        c = []
        for j in range(len(Y[i])):
            c.append(uniform(0, 1))
        u.append(c)
        
    for i in range(len(Y)):
            #print("epoca: ",i)
            for j in range(len(Y[i])): 
                somat = 0
                result = 0
                l = 0
                while(l < j):
                    result = result + u[l][j]*Y[l]
                    l = l + 1
                somat = somat +  w[i][j]*Y[i][j] + result
                Y[j] = somat
            for j in range(len(Y[i])): 
                w[i][j] = w[i][j] + eta*Y[i][j]*Y[j]
                l = 0
                while(l < j):
                    u[l][j] = u[l][j] -eta2*Y[l]*Y[j]
                    l = l + 1
    print(Y)
    return Y
    


def backpropagation(vet_arquitetura, matriz_pesos, mat_entrada, mat_saida, eta = 0.5, threshold = 0.01):
    erros_totais = 2 * threshold
    avg = 2 * threshold
    interacao = 1
    #lista = random.sample(range(0,len(mat_entrada)), len(mat_entrada))
    while(avg> threshold):
        erros_totais = 0
        #print(len(lista))
        for e in range(len(mat_entrada)):
          z = e
          frd = forward(matriz_pesos, vet_arquitetura, mat_entrada[z], mat_saida[z])
          f_net_n_camadas = frd[1]
          erro = [] 
          for i in range(len(f_net_n_camadas[1])):  
              #print(mat_saida[z][i])
              erro.append(mat_saida[z][i]-f_net_n_camadas[1][i])
          # print("esperado = ",saida[i],"\n obtido = ",matriz_respostas[3], "\n erro = ",erro)
          somat_erro = 0
          for s in range(len(f_net_n_camadas[1])):
                somat_erro += erro[s]*erro[s]/2
          erros_totais += somat_erro
          #camada de saida atualizando os pesos
          pesos_out = matriz_pesos[len(matriz_pesos)-1]
          delta_apos = []
          for d in range(vet_arquitetura[len(vet_arquitetura)-1]):
              delta_apos.append(erro[d] * derivada_f_net(f_net_n_camadas[len(f_net_n_camadas)-1][d]))
          for k in range(vet_arquitetura[len(vet_arquitetura)-1]):
              for n in range(vet_arquitetura[len(vet_arquitetura)-2]):
                  pesos_out[k][n] += eta * delta_apos[k] * f_net_n_camadas[len(f_net_n_camadas)-2][n]
          matriz_pesos[len(matriz_pesos)-1] = pesos_out
          
          #camadas ocultas intermediarias entre entrada e saida
          cond = (len(vet_arquitetura)-1) - 2
          while(cond >= 1):
              #print("camada: ", cond)
              pesos_out = matriz_pesos[cond+1]
              delta_ant = delta_apos
              delta_apos = []
              for k in range(vet_arquitetura[cond+1]):
                  result = 0
                  for n in range(vet_arquitetura[cond+2]):
                      result += delta_ant[n] * pesos_out[n][k]
                  delta_apos.append(result)
              pesos_out = matriz_pesos[cond]
              for l in range(vet_arquitetura[cond+1]):
                  for j in range(vet_arquitetura[cond]):
                      pesos_out[l][j] += eta * delta_apos[l] * f_net_n_camadas[cond-1][j]
              matriz_pesos[cond] = pesos_out
              cond = cond -1
          #camada de entrada atualizando
          delta_h = []
          pesos_out = matriz_pesos[1]
          for k in range(vet_arquitetura[1]):
              result = 0
              #colocar a camada anterior 
              for n in range(vet_arquitetura[2]):
                  result += delta_apos[n] *pesos_out[n][k]
              delta_h.append(derivada_f_net(f_net_n_camadas[0][k]) * result )
          
          pesos_entrada = matriz_pesos[0]    
          for l in range(vet_arquitetura[1]):
              for j in range(vet_arquitetura[0]):
                  pesos_entrada[l][j] += eta * delta_h[l] * mat_entrada[z][j]
          matriz_pesos[0] = pesos_entrada
        avg = erros_totais / interacao
        print ("erro: ", erros_totais," ,erro medio: ",avg)
        print("interações = ", interacao)
        interacao = interacao+1
    return matriz_pesos 
            
#calcular os erros
def calc_erro(vet_saida,vet_saida_obtido):
    erros = []
    for i in range(len(vet_saida)):
        erros.append(vet_saida[i] - vet_saida_obtido[i])
    return erros
            
def main():
    vet_arquitetura = [1,1,3]
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
      
    nova_mat_entrada = pcaAdaptativa(mat_entrada)
    mat_entrada_treino = nova_mat_entrada[0:int(len(nova_mat_entrada)*0.90)]
    mat_saida_treino = mat_saida_2[0:int(len(mat_saida_2)*0.90)]
    mat_entrada_teste = nova_mat_entrada[int(len(nova_mat_entrada)*0.90):len(nova_mat_entrada)]
    mat_saida_teste = mat_saida_2[int(len(mat_saida_2)*0.90):len(mat_saida_2)]   

    
    matriz_pesos = arquitetura(vet_arquitetura)
    pesos_obtidos = backpropagation(vet_arquitetura, matriz_pesos, mat_entrada_treino, mat_saida_treino, eta = 1, threshold = 0.01)
    for i in range(len(mat_entrada_teste)):
        frd = forward(pesos_obtidos, vet_arquitetura, mat_entrada_teste[i], mat_saida_teste[i])
        mat_saida = frd[1]
        erro = []
        for j in range(vet_arquitetura[len(vet_arquitetura)-1]):               
            erro.append(mat_saida_teste[i][j]-mat_saida[len(mat_saida)-1][j])
        somat_erro = 0
        for s in range(vet_arquitetura[len(vet_arquitetura)-1]):
            somat_erro += erro[s]*erro[s]
        print("esperado: ,",mat_saida_teste[i],",obtido: ,",mat_saida[len(vet_arquitetura)-2],",erro: ,", somat_erro, ",",(somat_erro<0.24))
   

if __name__ == "__main__":
    main()