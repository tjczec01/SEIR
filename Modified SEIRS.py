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
import math
import os
import time
from tqdm import tqdm

start = time.time() #Real time when the program starts to run

clear = lambda: os.system('cls')
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
path_fol = "{}\Modified SEIRS Model for the Spread of Disease".format(dir_path)

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
       if teindex == len(R):
              teindex -= 1
       elif teindex >= len(R):
              while teindex >= len(R):
                     teindex -= 1
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

def betaf(R_0, α, γ, μ):
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
       σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old, time, Is, Ss, Rs, λ, d  = args
       args2 = [τ_ξ, ξ, time, Rs]
       S = y[0]
       E = y[1]
       I = y[2]
       R = y[3]
       D = y[4]
       if t >= τ_σ and t >= τ_ξ:
              tindex = min(range(len(time)), key=lambda i: abs(time[i]- t)) - 1  # time.index(tt)
              if tindex == len(Is):
                     tindex -= 1
              elif tindex >= len(Is):
                     tindex = len(Is) - 1
              It = Is[tindex]
              St = Ss[tindex]
              dsdt = Λ - μ*S - (β*I*S)/N + (σ*β*It*St)/N + R_s(t, *args2)
              drdt = (γ + μ)*I - μ*R - R_s(t, *args2)
       if t >= τ_σ and t < τ_ξ:
               tindex = min(range(len(time)), key=lambda i: abs(time[i]- t)) - 1 # time.index(tt)
               if tindex == len(Is):
                     tindex -= 1
               elif tindex >= len(Is):
                      tindex = len(Is) - 1
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
       didt = (μ + α)*E - (γ + μ)*I - (γ*((1 - κ_old)*N_old + (1 - κ)*(1 - N_old)))*I 
       dDdt = d*γ*((1 - κ_old)*N_old + (1 - κ)*(1 - N_old))*I - λ*D
       
       return [dsdt, dedt, didt, drdt, dDdt]
       
Init_inf = 10

days = 1200
intval = 1000
tint = days/intval
time_list = [i*tint for i in range(intval+1)]
zhi_list = [0, 30, 60, 120, 180, 240, 360]
# zhi_list = [0]
τ_inc = 5.1
τ_rec = 18.8
τ_lat = 11.2
R_0i = 5.2

for z in zhi_list:
       σ = 0.0
       β = beta(5.2, 1.0/18.8)
       γ = gamma(τ_rec)
       α = alpha(τ_inc)
       Λ = 0 # Birth rate
       μ = 0 # Death rate
       ξ = 0.02
       κ = 0.98  
       κ_old = 0.92
       τ_ξ = z
       τ_pre = 30
       τ_post = 13
       τ_σ = τ_σf(τ_pre, τ_post)
       N = 329436928 #51.5 * 10**6   
       N_old = 0.15
       λ = τ_lat**-1
       d = 0.034
       S = [N]
       E = [20*Init_inf]
       I = [Init_inf]
       R = [0]
       D = [0]
       
       argslist = (σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old, time_list, I[:], S[:], R[:], λ, d)

       for i in tqdm(range(intval)):
              t_start = time_list[i]
              t_end = time_list[i+1]
              Y0 = [S[-1], E[-1], I[-1], R[-1], D[-1]]
              answer = solve_ivp(SEIRS, [t_start, t_end], Y0, t_eval=[t_start, t_end], method = 'Radau', args=(σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old, time_list, I[:], S[:], R[:], λ, d), rtol=1E-6, atol=1E-9) 
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
              
       Imax = max(I)*1.05
       
       fig = plt.figure()
       plt.plot(time_list, I, 'b-', label=r'$\it{Infected}$')
       plt.plot(time_list, D, '-', color='orange', label=r'$\it{Dead}$')
       plt.legend([r'$\it{Infected}$', r'$\it{Dead}$'], loc="best", fontsize=15)
       plt.xlim((0,days))
       plt.ylim((0,Imax))
       plt.yticks([roundup(i*(Imax/10), intval) for i in range(11)])
       plt.xticks([int(i*100) for i in range(13)])
       plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
       plt.gca().set_yticklabels([r'{:,}'.format(int(x)) for x in plt.gca().get_yticks()])
       plt.xlabel(r'$\bf{Time \ [Days]}$', fontsize=15)
       plt.ylabel(r'$\bf{Number \ of \ people}$', fontsize=15)
       plt.title(r'$\bf{SEIRS \ Method  \ for \ Spread \ of \ Disease}$', fontsize=18)
       plt.grid()
       fig.savefig(r"{}\SEIRS-{} Dead vs Infected.pdf".format(path_fol, τ_ξ), bbox_inches='tight')
       fig.savefig(r"{}\SEIRS-{} Dead vs Infected.svg".format(path_fol, τ_ξ), bbox_inches='tight')
       plt.show()

       Sp = [(i/N)*100.0 for i in S]
       Ep = [(i/N)*100.0 for i in E]
       Ip = [(i/N)*100.0 for i in I]
       Rp = [(i/N)*100.0 for i in R]
       Dp = [(i/N)*100.0 for i in D]
       
       Ip1 = (I).index(max(I))
       peakn = int(days*(Ip1/intval))
       Ip2 = (Ip).index(max(Ip))
       peakn2 = int(days*(Ip2/intval))
       Imax = max(I)*1.05
       
       fig = plt.figure()
       plt.plot(time_list, S, 'b-', label=r'$\it{Susceptible}$')
       plt.plot(time_list, E, 'c-', label=r'$\it{Exposed}$')
       plt.plot(time_list, I, 'r-', label=r'$\it{Infected}$')
       plt.plot(time_list, R, 'g-', label=r'$\it{Recovered}$')
       plt.axvline(x=int(time_list[Ip1]), color='k', linestyle='--')
       plt.legend([r'$\it{Susceptible}$', r'$\it{Exposed}$', r'$\it{Infected}$', r'$\it{Recovered}$', r'$\it{}$'.format("{}{}{}".format('{',"Peak \ = {}{}{} \ Days".format('{',r'\ {}'.format(peakn),'}'),'}'))], loc="best", fontsize=15)
       plt.xlim((0,days))
       plt.ylim((0,N))
       plt.yticks([roundup(i*(N/10.0), intval) for i in range(11)])
       plt.xticks([int(i*100) for i in range(13)])
       plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
       plt.gca().set_yticklabels([r'{:,}'.format(int(x)) for x in plt.gca().get_yticks()])
       plt.xlabel(r'$\bf{Time \ [Days]}$', fontsize=15)
       plt.ylabel(r'$\bf{Number \ of \ people}$', fontsize=15)
       plt.title(r'$\bf{SEIRS \ Method  \ for \ Spread \ of \ Disease}$', fontsize=18)
       plt.grid()
       fig.savefig(r"{}\SEIRS Population.pdf".format(path_fol), bbox_inches='tight')
       fig.savefig(r"{}\SEIRS Population.svg".format(path_fol), bbox_inches='tight')
       plt.show()
       
       fig = plt.figure()
       plt.plot(time_list, Sp, 'b-', label=r'$\it{Susceptible}$')
       plt.plot(time_list, Ep, 'c-', label=r'$\it{Exposed}$')
       plt.plot(time_list, Ip, 'r-', label=r'$\it{Infected}$')
       plt.plot(time_list, Rp, 'g-', label=r'$\it{Recovered}$')
       plt.axvline(x=int(time_list[Ip2]), color='k', linestyle='--')
       plt.legend([r'$\it{Susceptible}$', r'$\it{Exposed}$', r'$\it{Infected}$', r'$\it{Recovered}$', r'$\it{}$'.format("{}{}{}".format('{',"Peak \ = {}{}{} \ Days".format('{',r'\ {}'.format(peakn2),'}'),'}'))], loc="best", fontsize=15)
       plt.xlim((0,days))
       plt.ylim((0,100))
       plt.yticks([i*10 for i in range(11)])
       plt.gca().set_yticklabels([f'{int(y)}%' for y in plt.gca().get_yticks()])
       plt.gca().set_xticklabels([f'{int(x)}' for x in plt.gca().get_xticks()])
       plt.xlabel(r'$\bf{Time \ [Days]}$', fontsize=15)
       plt.ylabel(r'$\bf{Number \ of \ people}$', fontsize=15)
       plt.title(r'$\bf{SEIRS \ Method  \ for \ Spread \ of \ Disease}$', fontsize=18)
       plt.grid()
       fig.savefig(r"{}\SEIRS Percent.pdf".format(path_fol), bbox_inches='tight')
       fig.savefig(r"{}\SEIRS Percent.svg".format(path_fol), bbox_inches='tight')
              
       fig = plt.figure()
       plt.plot(time_list, D, 'b-', label=r'$\it{Deaths}$')
       # plt.axvline(x=int(time_list[Ip1]), color='k', linestyle='--')
       plt.legend([r'$\it{Deaths}$',  r'$\it{}$'.format("{}{}{}".format('{',"Peak \ = {}{}{} \ Days".format('{',r'\ {}'.format(peakn),'}'),'}'))], loc="best", fontsize=15)
       plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
       plt.gca().set_yticklabels([r'{:,}'.format(int(x)) for x in plt.gca().get_yticks()])
       plt.xlabel(r'$\bf{Time \ [Days]}$', fontsize=15)
       plt.ylabel(r'$\bf{Number \ of \ people}$', fontsize=15)
       plt.title(r'$\bf{SEIRS \ Method  \ for \ Spread \ of \ Disease}$', fontsize=18)
       plt.grid()
       fig.savefig(r"{}\SEIRS Deaths.pdf".format(path_fol), bbox_inches='tight')
       fig.savefig(r"{}\SEIRS Deaths.svg".format(path_fol), bbox_inches='tight')
       plt.show()
       
       S.clear()
       E.clear()
       I.clear()
       R.clear()
       D.clear()
       Sp.clear()
       Ep.clear()
       Ip.clear()
       Rp.clear()
       Dp.clear()

     
       
end = time.time() #Time when it finishes, this is real time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Completion Time: {} Hours {} Minutes {} Seconds".format(int(hours),int(minutes),int(seconds)))

timer(start,end) # Prints the amount of time passed since starting the program
