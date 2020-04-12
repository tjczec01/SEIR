# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:34:43 2020

@author: Travis Czechorski tjcze01@gmail.com
"""

import numpy as np
import sympy as sp
import pprint 
from sympy import Heaviside

S, E, I, R, N, D, C, α, β, γ, μ, Λ, F, σ, D, κ, λ, d, t, β_t = sp.symbols('S E I R N D C α, β, γ, μ, Λ, F, σ, D, κ, λ d t β_t')


t, σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old = sp.symbols('t σ β γ α Λ μ ξ κ κ_old τ_ξ τ_σ N N_old')
pr = pprint.pprint


def SEIRS(t, y, *args):
       σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old  = args
       # args2 = [τ_ξ, ξ, time, Rs]
       S = y[0]
       E = y[1]
       I = y[2]
       R = y[3]
       D = y[4]
       # tt = t - τ_σ
       # val1 = γ*((1 - κ_old)*N_old + (1 - κ)*(1 - N_old))
       # val2 = (1 - κ)*(1 - N_old)
       # val3 = val1 + val2
       # val4 = val3*γ
       
       # def R_s(t, *args):
       #        τ_ξ, ξ, times, R = args
       #        te = t - τ_ξ
       #        teindex = min(range(len(times)), key=lambda i: abs(times[i]- te)) - 1
       #        return ξ*R[teindex]
       
       # print(val4)
       # if t >= τ_σ and t >= τ_ξ:
       #        tindex = min(range(len(time)), key=lambda i: abs(time[i]- tt)) - 1  # time.index(tt)
       #        It = Is[tindex]
       #        St = Ss[tindex]
              
       # if t >= τ_σ and t < τ_ξ:
       #         tindex = min(range(len(time)), key=lambda i: abs(time[i]- tt)) - 1 # time.index(tt)
       #         It = Is[tindex]
       #         St = Ss[tindex]
       #         dsdt = Λ - μ*S - (β*I*S)/N + (σ*β*It*St)/N 
       #         drdt = (γ + μ)*I - μ*R 
       # elif t < τ_σ:
       #        It = 0
       #        St = 0
       #        dsdt = Λ - μ*S - (β*I*S)/N  #+ R_s(t, *args2)
       #        drdt = (γ + μ)*I - μ*R
       dsdt = Λ - μ*S - (β*I*S)/N + (σ*β*I*S)/N + ξ*Heaviside(R - τ_ξ)
       dedt = (β*I*S)/N - (σ*β*I*S)/N - (μ + α)*E
       didt = (μ + α)*E - (γ + μ)*I - γ*((1 - κ_old)*N_old + (1 - κ)*(1 - N_old))*I
       drdt = (γ + μ)*I - μ*R - ξ*Heaviside(R - τ_ξ)
       dDdt = γ*((1 - κ_old)*N_old + (1 - κ)*(1 - N_old))*I
       
       return [dsdt, dedt, didt, drdt, dDdt]

def Jac2(t, y, *args):
       σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old  = args
       S = y[0]
       E = y[1]
       I = y[2]
       R = y[3]
       D = y[4]
       return [[I*β*σ/N - I*β/N - μ,      0,                                      S*β*σ/N - S*β/N,      ξ*sp.DiracDelta(R - τ_ξ), 0],
               [   -I*β*σ/N + I*β/N, -α - μ,                                     -S*β*σ/N + S*β/N,                          0, 0],
               [                  0,  α + μ, -γ*(N_old*(1 - κ_old) + (1 - N_old)*(1 - κ)) - γ - μ,                          0, 0],
               [                  0,      0,                                                γ + μ, -μ - ξ*sp.DiracDelta(R - τ_ξ), 0],
               [                  0,      0,          γ*(N_old*(1 - κ_old) + (1 - N_old)*(1 - κ)),                          0, 0]]

ydot = sp.Matrix(SEIRS(t, [S, E, I, R, D], σ, β, γ, α, Λ, μ, ξ, κ, κ_old, τ_ξ, τ_σ, N, N_old))
C = sp.Matrix([S, E, I, R, D])
J = ydot.jacobian(C)
pr(J)
print(J)
def Beta(a, b, D, N, k):
       B = b*(1 - a)*((1 - D/N)**k)
       return B
b_1, b_2, r_1, r_2, v, p, b, q = sp.symbols('b_1 b_2 r_1 r_2 v p b q')
Eeq = b_1*(S*F/N) + b_2*(S*I/N) - (σ + μ)*E 
Ieq = σ*E - (γ + μ)*I
Seq = Λ - b_1*(S*F/N) - b_2*(S*I/N) - μ*S
Req =  μ*I  - d*γ*R  


FM = sp.Matrix(([[ b_2*(S*I/N)], [0.0], [0.0], [0.0]]))
VM = sp.Matrix([[(σ + μ)*E],  [-σ*E + (γ + μ)*I],   [ b_2*(S*I/N) + μ*S], [-μ*I  + d*γ*R]])
vm = VM.subs({E : 0, I : 0, S : 1, R : 0})
# F0 = FM.subs({S : 1}) #{E : 0, I : 0, S : 1, R : 0})
# V0 = VM.subs({S : 1})
FJ = FM.jacobian(sp.Matrix((E, I, S, R)))
VJ = VM.jacobian(sp.Matrix([E, I, S, R]))
V0 = VJ.subs({E : 0, I : 0, S : 1, R : 0, N : 1})
F0 = FJ.subs({E : 0, I : 0, S : 1, R : 0, N : 1})
VV = sp.Matrix(V0)
FF = sp.Matrix(F0)
# VJf = V0.subs({E : 0, I : 0, S : 1, R : 0})
# FJf = FJ.subs({E : 0, I : 0, S : 1, R : 0})
# VI = V0.eigenvals(simplify=True)
FI = F0.eigenvals(simplify=True, multiple=True)
# VIe = V0.eigenvects(simplify=True)
# FIe = F0.eigenvects(simplify=True)
# B = sp.Matrix([0, 0, 1, 0])
# pr(FM.LDLsolve(B))
# print(sol)
# print(params)
# pr(vj)
# pr(fj) #simplify=True, multiple=True
# pr(F0)
# pr(V0)
# pr(FI)
# pr(VV)
V = VV[:2,:2]
F = FF[:2, :2]
# pr(F)
# pr(V)
Vinv = V.inv()
Fin = F*Vinv
# pr(Fin)
# pr(Fin.eigenvals(simplify=True, multiple=True))
R0 = Fin[0]
pr(R0)
def R0(b_2, γ, σ, μ):
       return (b_2*σ)/((γ + μ)*(μ + σ))
# pr(b_1*v/(-p*r_2*v + (d + r_2)*(d + r_1 + v)))
# print(sp.simplify(Fin, ratio=2.0)) #.norm(2))
# pr(V0.det())
# pr(F0)
# pr(VIe[:])
# pr(FIe)
# pr(VIe)
# pr(F0)
# pr(V0)
# pr(FM.jacobian(sp.Matrix([S, E, I, R])))
# print(VM.jacobian(sp.Matrix([S, E, I, R])))
# pr(list(FI.keys()))
# print(list(FI.values()))
# pr(FI[0][2][0])
# pr(FI[0][2][1])
# pr(FI[0][2][2])
# print(list(VI.keys()))
# print(list(VI.values()))
# pr(VI[0][2][0])
# pr(VI[0][2][1])
# pr(VI[0][2][2])
# pr(VI)
# VJJ = sp.Matrix(VJ).inv()
# print(VJJ.subs({E : 0, I : 0, S : 1, R : 0}))
# pr(V0.transpose())
# pr(VJf)
def SEIR(t, y, *args):
       σ, β, γ, μ, Λ, F, α, d, κ, λ, β_t = args
       # β_t = Beta(α, β, y[5], y[4], κ)
       dsdt = Λ - μ*y[0] - ((β*F)/y[4])*y[0] - (β_t/y[4])*y[2]*y[0] 
       dedt = ((β*F)/y[4])*y[0] + (β_t/y[4])*y[2]*y[0] - (μ + σ)*y[1]
       didt = σ*y[1] - (μ + γ)*y[2]
       drdt = γ*y[2] - μ*y[3]
       dndt = -μ*y[4]
       dDdt = d*γ*y[2] - λ*y[5]
       dcdt = σ*y[1]
       return [dsdt, dedt, didt, drdt, dndt, dDdt, dcdt]

def jacobian(t, y, *args):
        σ, β, γ, μ, Λ, F, α, d, κ, λ, β_t = args
        return [[-F*β/y[4]- y[2]*β_t/y[4]- μ,      0, -y[0]*β_t/y[4],  0,  F*y[0]*β/y[4]**2 + y[2]*y[0]*β_t/y[4]**2,  0, 0],
                [     F*β/y[4]+ y[2]*β_t/y[4], -μ - σ,  y[0]*β_t/y[4],  0, -F*y[0]*β/y[4]**2 - y[2]*y[0]*β_t/y[4]**2,  0, 0],
                [                   0,      σ,   -γ - μ,  0,                          0,  0, 0],
                [                   0,      0,        γ, -μ,                          0,  0, 0],
                [                   0,      0,        0,  0,                         -μ,  0, 0],
                [                   0,      0,      d*γ,  0,                          0, -λ, 0],
                [                   0,      σ,        0,  0,                          0,  0, 0]]

# Y = [S, E, I, R, N, D, C]
# arg = [σ, β, γ, μ, Λ, F, α, d, κ, λ, β_t]
# ydot = SEIR(t, Y, *arg)
# J = sp.Matrix(ydot).jacobian(Y)
# pr(J)

# JAC = [[-I*b,    -S*b, 0],
#        [ I*b, S*b - k, 0],
#        [   0,       k, 0]]

# f = {\[CapitalLambda] - \[Mu]*S[t] - (\[Beta]*F[t]*S[t])/Np + (\[Sigma]*\[Beta]*F[t - \[Tau]]*S[t - \[Tau]])/Np +\[Xi]*R[t - te], (\[Beta]*F[t]*S[t])/Np - (\[Sigma]*\[Beta]*F[t - \[Tau]]*S[t - \[Tau]])/Np - (\[Mu] + \[Alpha])*X[t], (\[Mu] + \[Alpha])*X[t] - (\[Gamma] + \[Mu])*F[t] - \[Sigma]*((1 - kold)*Npold + (1 - k)*(1 - Npold))*F[t], (\[Gamma] + \[Mu])*F[t] - \[Mu]*R[t] - \[Xi]*R[t - te],  \[Sigma]*((1 - kold)*Npold + (1 - k)*(1 - Npold))*F[t]}
#    pars = {\[Sigma] ->0.76, \[Beta] -> 0.277, \[Gamma] -> 0.053, \[Alpha] -> 0.196, \[CapitalLambda] -> 0, \[Mu] -> 0, \[Xi] -> 0.01, k -> 0.98, kold -> 0.96, te -> 30, \[Tau] -> 43, Np -> 10000000, Npold -> 0.15}
#    reactor = NonlinearStateSpaceModel[{f,{S[t], X[t], F[t],}},{S[t], X[t], F[t], R[t], M[t]},   t]  /. pars