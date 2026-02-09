#!/usr/bin/env python3
"""
Purpose
-------
This script evaluates a simple kinematical *structure-factor proxy* as a function of an
interlayer translation parameter b (0 → 1). It was used to compare how diffraction
intensity changes for:
  - a "perfect" translation path (single shift direction)
  - a "partial + partial" path (two sequential partial shifts)

The output is a set of intensity-vs-b curves for several reciprocal vectors g.

Model (kinematical proxy)
-------------------------
For each b, we compute an amplitude-like quantity:
  F(b; g) = 2*f_Mo*cos(2π <t(b) - r_Mo, g>) + 4*f_S*cos(2π <t(b) - r_Mo - phase, g>)
and use I(b; g) = |F|^2.

Notes
-----
- This is a *toy model* for qualitative trend checking. It does not include dynamical
  diffraction, thickness effects, or full electron scattering factors.
- Coordinates are fractional (in-plane) unless you explicitly map them to Cartesian.


Dependencies
------------
pip install numpy matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Physical / model parameters
# ----------------------------

# Atomic form factor (approximated as Z-number)
fmo=41.956
fs=15.998

# Internal phase shift used in the S sublattice term (fractional coordinates)
AB_unit = np.array([[0,0,0], [1/3,1/3,0], [1/3,1/3,1/2], [2/3,2/3,1/2]])
BA_unit = np.array([[1/3,1/3,1/2], [1/3,1/3,1/2], [0,0,0], [1/3,1/3,0]])

phase = np.array([2/3, 1/3,0])

# Reference Mo position (fractional coordinates)
r_Mo = np.array([[2/3, 1/3, 0.5]]) 

# Default b grids:
b = np.linspace(0,1,100)
b2 = np.linspace(0,1,200)


def burgers_vector_perfect(g):
    """
    "Perfect" translation path
    """

    intensity_list = []
    
    p = np.array([0,1,0])

    
    for i in range(0, int(len(b2))):
 
        F = 2*fmo*np.cos(2*np.pi*np.dot(b2[i]*p/2 - r_Mo ,g)) + 4*fs*np.cos(2*np.pi*np.dot(b2[i]*p/2 - r_Mo - phase,g))
    
        intensity_list.append(F*F.conjugate())

        
    return intensity_list
  
    
def burgers_vector_partial(g):
    """
    Two-step path ("partial + partial") implemented as two segments concatenated.
    """

    intensity_list = []
    
    p1 = np.array([-1/3, 1/3, 0])
    p2 = np.array([1/3, 2/3, 0])

    
    for i in range(0, int(len(b))):
 
        F = 2*fmo*np.cos(2*np.pi*np.dot(b[i]*p1/2 - r_Mo,g)) + 4*fs*np.cos(2*np.pi*np.dot(b[i]*p1/2 - r_Mo - phase ,g))
    
        intensity_list.append(F*F.conjugate())

    for i in range(0, int(len(b))):
 
        F = 2*fmo*np.cos(2*np.pi*np.dot(p1/2+b[i]*p2/2 - r_Mo ,g)) + 4*fs*np.cos(2*np.pi*np.dot(p1/2+b[i]*p2/2 - r_Mo - phase ,g))
        
        intensity_list.append(F*F.conjugate())

        
    return intensity_list
    

# ----------------------------
# Main pipeline
# ----------------------------


figsizenum = (6,8)
tickfontsize = 20
plt.rcParams['figure.constrained_layout.use'] = True

# ----------------------------
# Perfect: g = {100}
# ----------------------------

plt.figure(figsize=figsizenum)

plt.plot(b2, burgers_vector_perfect(np.array([1,0,0])), label = 'g=100', color = 'r', linewidth = 8)
plt.plot(b2, burgers_vector_perfect(np.array([0,1,0])), label = 'g=010', color = 'gray', linewidth = 8)
plt.plot(b2, burgers_vector_perfect(np.array([1,-1,0])), label = 'g=1-10', color = 'k', linewidth = 8)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(False)
plt.show()

# ----------------------------
# Perfect: g = {110}
# ----------------------------

plt.figure(figsize=figsizenum)

plt.plot(b2, burgers_vector_perfect(np.array([-1,2,0])), label = 'g=-120', color = 'r', linewidth = 8)
plt.plot(b2, burgers_vector_perfect(np.array([1,1,0])), label = 'g=110', color = 'gray', linewidth = 8)
plt.plot(b2, burgers_vector_perfect(np.array([2,-1,0])), label = 'g=-210', color = 'k', linewidth = 4, linestyle = '--')
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)


plt.grid(False)


plt.show()

# ----------------------------
# Partial-Partial: g = {100}
# ----------------------------

plt.figure(figsize=figsizenum)

plt.plot(b2, burgers_vector_partial(np.array([0,1,0])), label = 'g=010', color = 'gray', linewidth = 8)
plt.plot(b2, burgers_vector_partial(np.array([1,-1,0])), label = 'g=1-10', color = 'k', linewidth = 8)
plt.plot(b2, burgers_vector_partial(np.array([1,0,0])), label = 'g=100', color = 'r', linewidth = 4, linestyle = '--')
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)


plt.grid(False)


plt.show()

# ----------------------------
# Partial-Partial: g = {110}
# ----------------------------

plt.figure(figsize=figsizenum)

plt.plot(b2, burgers_vector_partial(np.array([1,1,0])), label = 'g=110', color = 'gray', linewidth = 8)
plt.plot(b2, burgers_vector_partial(np.array([2,-1,0])), label = 'g=-210', color = 'k', linewidth = 8)
plt.plot(b2, burgers_vector_partial(np.array([-1,2,0])), label = 'g=-120', color = 'r', linewidth = 4, linestyle = '--')
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)


plt.grid(False)

plt.show()