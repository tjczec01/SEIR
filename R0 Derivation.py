# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:52:07 2020

@author: Travis Czechorski tjczec01@gmail.com
"""

import numpy as np
import sympy as sp
import pprint 
from IPython.display import display, Latex, Math, Image, DisplayObject

S, E, I, R, N, D, C, α, β, γ, μ, Λ, F, σ, D, κ, λ, d, t, β_t = sp.symbols('S E I R N D C α, β, γ, μ, Λ, F, σ, D, κ, λ d t β_t')

pr = pprint.pprint

def Beta(a, b, D, N, k):
       B = b*(1 - a)*((1 - D/N)**k)
       return B

β_1, β_2, r_1, r_2, v, p, b, q = sp.symbols('β_1 β_2 r_1 r_2 v p b q')
Eeq = β_1*(S*F/N) + β_2*(S*I/N) - (σ + μ)*E 
Ieq = σ*E - (γ + μ)*I
Seq = Λ - β_1*(S*F/N) - β_2*(S*I/N) - μ*S
Req =  μ*I  - d*γ*R  
FM = sp.Matrix(([[ β_2*(S*I/N)], [0.0], [0.0], [0.0]]))
VM = sp.Matrix([[(σ + μ)*E],  [-σ*E + (γ + μ)*I],   [ β_2*(S*I/N) + μ*S], [-μ*I  + d*γ*R]])
vm = VM.subs({E : 0, I : 0, S : 1, R : 0})
FJ = FM.jacobian(sp.Matrix((E, I, S, R)))
VJ = VM.jacobian(sp.Matrix([E, I, S, R]))
V0 = VJ.subs({E : 0, I : 0, S : 1, R : 0, N : 1})
F0 = FJ.subs({E : 0, I : 0, S : 1, R : 0, N : 1})
VV = sp.Matrix(V0)
FF = sp.Matrix(F0)
FI = F0.eigenvals(simplify=True, multiple=True)
V = VV[:2,:2]
F = FF[:2, :2]
Vinv = V.inv()
Fin = F*Vinv
R0 = r'$R_{0} =' + r'{}$'.format(sp.latex(Fin[0]))
display(Latex(R0))