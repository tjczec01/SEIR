# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:18:51 2020

@author: Travis Czechorski tjczec01@gmail.com

Model: 
       
COVID-19: Development of A Robust Mathematical
Model and Simulation Package with Consideration for
Ageing Population and Time Delay for Control Action
and Resusceptibility


https://arxiv.org/abs/2004.01974


"""

import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
import numpy as np
import math
import pandas as pd
import os

clear = lambda: os.system('cls')
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
path_fol = "{}\MSEIRS Model for Spread of Disease".format(dir_path)

try:
       os.mkdir(path_fol)
except:
       pass

def roundup(x, places):
    return int(math.ceil(x / int(places))) * int(places)

def R0(α, β, μ, γ):
       R_0 = (α/(μ + α))*(β/(μ + γ))
       return R_0

def R_t(σ, R_0):
       return (1 - σ)*R_0

def R_s(t, *args):
       τ_ξ, ξ, times, R = args
       te = t - τ_ξ
       teindex = min(range(len(times)), key=lambda i: abs(times[i]- te)) - 1
       return ξ*R[teindex]

def R_c(t, τ_ξ, ξ):
       tt = t - τ_ξ
       return ξ*tt

def τ_σf(τ_pre, τ_post):
       return τ_pre + τ_post

def τ_in(α):
       return α**-1

def τ_re(γ):
       return γ**-1

def τ_ho(σ):
       return σ**-1

def alpha(τ_inc):
       return τ_inc**-1

def beta(R_0, γ):
       return R_0*γ

def betaf(R_0, α, μ, γ):
       val1 = R_0*(μ + α)*(μ + γ)
       val2 = val1/α
       return val2

def gamma(τ_rec):
       return τ_rec**-1


# print(betaf(2.8, 0.48, 0.0205, 3**-1))
# Λ is the birth rate in the overall population 
# µ is the death rate due to conditions other than the COVID-19
# β is the rate of transmission per S-I contact
# α is the rate of which an exposed person becomes infected
# γ is the recovery rate
# σ is the efficiency of the control action
# ξ is the percentage of the recovered population who are resusceptible
# κ represents the percentage of non-elderly who recovered
# κ_old represents the percentage of non-elderly and elderly who recovered 
# τ_(pre - σ) represents the time to initiate the control action after the first confirmed case at t = 0
# τ_(post - σ) represents the time after the control action has been initiated but before the effects are evidenced in the outputs of the system
# τ_inc incubation time
# τ_rec recovery time
# τ_hos represents the time spent hospitalised
# τ_ξ represents the duration of temporary immunity
# N represents the stock population
# N_old represents the percentage of elderly population (above 65 years of age) 
# R_0 represents the basic reproduction number 
# t represents the time

def SEIRS(t, y, *args):
       σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old, time, Is, Ss, Rs  = args
       args2 = [τ_ξ, ξ, time, Rs]
       S = y[0]
       E = y[1]
       I = y[2]
       R = y[3]
       D = y[4]
       tt = t - τ_σ
       val1 = (1 - κ_old)*N_old
       val2 = (1 - κ)*(1 - N_old)
       val3 = val1 + val2
       val4 = val3*γ
       if t >= τ_σ and t >= τ_ξ:
              tindex = min(range(len(time)), key=lambda i: abs(time[i]- tt)) - 1  # time.index(tt)
              It = Is[tindex]
              St = Ss[tindex]
              dsdt = Λ - μ*S - (β*I*S)/N + (σ*β*It*St)/N + R_s(t, *args2)
              drdt = (γ + μ)*I - μ*R - R_s(t, *args2)
       if t >= τ_σ and t < τ_ξ:
               tindex = min(range(len(time)), key=lambda i: abs(time[i]- tt)) # time.index(tt)
               It = Is[tindex]
               St = Ss[tindex]
               dsdt = Λ - μ*S - (β*I*S)/N + (σ*β*It*St)/N 
               drdt = (γ + μ)*I - μ*R 
       elif t < τ_σ:
              It = 0
              St = 0
              dsdt = Λ - μ*S - (β*I*S)/N  #+ R_s(t, *args2)
              drdt = (γ + μ)*I - μ*R
       dedt = (β*I*S)/N - (σ*β*It*St)/N - (μ + α)*E
       didt = (μ + α)*E - (γ + μ)*I - val4*I
       dDdt = val4*I
       
       return [dsdt, dedt, didt, drdt, dDdt]
       
Init_inf = 4

days = 200
intval = 100
tint = days/intval
time_list = [round(i*tint, 2) for i in range(intval+1)]

σ = 0.0076
α = 0.196
β = 0.277
γ = 0.053 
Λ = 0.0 # Birth rate
μ = 0.0 # Death rate
ξ = 0.00 
κ = 0.98    
κ_old = 0.96
τ_ξ = 0
τ_pre = 30
τ_post = 13
τ_σ = τ_σf(τ_pre, τ_post)
N = 51.5 * 10**6 #329436928  
N_old = 0.15   
its = roundup(days/τ_σ, 1)
S = [N]
E = [20*Init_inf]
I = [Init_inf]
R = [0]
D = [0]
R_0i = 2.8
τ_inc = 5.1
τ_rec = 18.8
argslist = [σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old, time_list, I[:], S[:], R[:]]
 

for i in range(intval):
       t_start = time_list[i]
       t_end = time_list[i+1]
       Y0 = [S[-1], E[-1], I[-1], R[-1], D[-1]]
       α = alpha(τ_inc)
       γ = gamma(τ_rec)
       b1 = beta(R_0i, γ)
       R_0 = (1 - σ)*R_0i
       # β = 0.5944
       answer = solve_ivp(SEIRS, [t_start, t_end], Y0, t_eval=[t_start, t_end], method = 'Radau', args=(σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old, time_list, I[:], S[:], R[:])) #, rtol=1E-10, atol=1E-10) # jac=jacobian,
       Sn = answer.y[0][-1]
       En = answer.y[1][-1]
       In = answer.y[2][-1]
       Rn = answer.y[3][-1]
       Dn = answer.y[4][-1]
       S.append(Sn)
       E.append(En)
       I.append(In)
       R.append(Rn)
       D.append(Dn)

Sp = [(i/N)*100.0 for i in S]
Ep = [(i/N)*100.0 for i in E]
Ip = [(i/N)*100.0 for i in I]
Rp = [(i/N)*100.0 for i in R]
Dp = [(i/N)*100.0 for i in D]

Ip1 = (I).index(max(I))
peakn = int(days*(Ip1/intval))
Ip2 = (Ip).index(max(Ip))
peakn2 = int(days*(Ip2/intval))

fig = plt.figure()
plt.plot(time_list, S, 'b-', label=r'$\it{Susceptible}$')
plt.plot(time_list, E, 'c-', label=r'$\it{Exposed}$')
plt.plot(time_list, I, 'r-', label=r'$\it{Infected}$')
plt.plot(time_list, R, 'g-', label=r'$\it{Recovered}$')
plt.axvline(x=int(time_list[Ip1]), color='k', linestyle='--')
plt.legend([r'$\it{Susceptible}$', r'$\it{Exposed}$', r'$\it{Infected}$', r'$\it{Recovered}$', r'$\it{}$'.format("{}{}{}".format('{',"Peak \ = {}{}{} \ Days".format('{',r'\ {}'.format(peakn),'}'),'}'))],loc="best", fontsize=15)
plt.xlim((0,days))
plt.ylim((0,N))
plt.yticks([roundup(i*(N/10.0), intval) for i in range(11)])
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.gca().set_yticklabels([r'{:,}'.format(int(x)) for x in plt.gca().get_yticks()])
plt.xlabel(r'$\bf{Time \ [Days]}$', fontsize=15)
plt.ylabel(r'$\bf{Number \ of \ people}$', fontsize=15)
plt.title(r'$\bf{SEIR \ Method  \ for \ Spread \ of \ Disease}$', fontsize=18)
plt.grid()

fig = plt.figure()
plt.plot(time_list, Sp, 'b-', label=r'$\it{Susceptible}$')
plt.plot(time_list, Ep, 'c-', label=r'$\it{Exposed}$')
plt.plot(time_list, Ip, 'r-', label=r'$\it{Infected}$')
plt.plot(time_list, Rp, 'g-', label=r'$\it{Recovered}$')
plt.axvline(x=int(time_list[Ip2]), color='k', linestyle='--')
plt.legend([r'$\it{Susceptible}$', r'$\it{Exposed}$', r'$\it{Infected}$', r'$\it{Recovered}$', r'$\it{}$'.format("{}{}{}".format('{',"Peak \ = {}{}{} \ Days".format('{',r'\ {}'.format(peakn2),'}'),'}'))],loc="best", fontsize=15)
plt.xlim((0,days))
plt.ylim((0,100))
plt.yticks([i*10 for i in range(11)])
plt.gca().set_yticklabels([f'{int(y)}%' for y in plt.gca().get_yticks()])
plt.gca().set_xticklabels([f'{int(x)}' for x in plt.gca().get_xticks()])
plt.xlabel(r'$\bf{Time \ [Days]}$', fontsize=15)
plt.ylabel(r'$\bf{Number \ of \ people}$', fontsize=15)
plt.title(r'$\bf{SEIR \ Method  \ for \ Spread \ of \ Disease}$', fontsize=18)
plt.grid()
       
       
       
       