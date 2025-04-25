# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:44:30 2025

@author: Mayank
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq

global sid, natom, ntype, nntype, ntime, dynamics, sid, delt,slat,rij

f = open('trajectory.lammpstrj', 'r')

#sid = 2
ntime = 200
nomega = np.linspace(0, ntime)

norigin = 100
delt = 0.02
xf = fftfreq(ntime, delt)[:ntime]
time = range(ntime)
ntype = 3
nntype = (161, 54, 216)
natom = 161 + 54 + 216
pos = np.zeros((ntime, natom, 3),dtype=np.float64)
vel = np.zeros((ntime, natom, 3),dtype=np.float64)
dos = np.zeros((ntime, natom),dtype=np.float64)
slat=np.zeros((3,3),dtype=np.float64)
aslat=np.zeros((3,3),dtype=np.float64)
rij=np.zeros(3,dtype=np.float64)
msd = np.zeros((ntime, natom))
msdtype = np.zeros((ntime, ntype))
dostype = np.zeros((ntime, ntype))

dynamics = True

for it in range(ntime):
    f.readline()  # Comment
    f.readline()  # Time step
    f.readline()  # Natom
    f.readline()  # Comment box-bounds
    f.readline()  # Comment box-bounds

    amin, amax = (f.readline()).split()  # Comment box-bounds
    bmin, bmax = f.readline().split()  # Comment box-bounds
    cmin, cmax = f.readline().split()  # Comment box-bounds
    vals = f.readline().split()  # Comment
    slat[0,0]=float(amax)-float(amin)
    slat[1,1]=float(bmax)-float(bmin)
    slat[2,2]=float(cmax)-float(cmin)
    aslat=np.linalg.inv(slat)
    # print(len(vals))
    if len(vals) == 7:
        sid = 2
    for i in range(natom):
        pos[it, i, :] = f.readline().split()[sid:]
        if dynamics and it >1:
            vel[it, i, :] = (pos[it, i, :] - pos[it-1, i, :]) / delt
            # print(pos[it,i,:])
#vel[ntime-1, :, :]=vel[ntime-2, :, :]
#print((vel[:, 1, 0]).shape)

# Compute MSD
for it in range(ntime):
    if it + norigin > ntime:
        norigin = ntime - it
    for i in range(natom):
        dummy = pos[it:it + norigin, i, :] - pos[0:norigin, i, :]
        msd[it, i] = np.tensordot(dummy, dummy, axes=2) / norigin
        # if i==1:
        # print(it,msd[it,i])

for i in range(ntype):
    n1 = int(np.sum(nntype[:i]))
    n2 = int(np.sum(nntype[:i + 1]))
    # print(n1,n2)
    msdtype[:, i] = np.sum(msd[:, n1:n2], axis=1)
# plt.plot(time,msdtype[:,0])
# plt.plot(time,msdtype[:,1])
# plt.plot(time,msdtype[:,2])

# Compute DOS
if dynamics:
    for i in range(natom):
        fx = fft(vel[:, i, 0])
        dos[:, i] = fx * np.conj(fx)
# print(dos[:,1])
for i in range(ntype):
    n1 = int(np.sum(nntype[:i]))
    n2 = int(np.sum(nntype[:i + 1]))
    # print(n1,n2)
    dostype[:, i] = (4.0/ntime**2) * np.sum(dos[:, n1:n2], axis=1)
dostype[0, :] = 0
plt.plot(4.136*xf, np.sum(dostype[:, :],axis=1))
plt.xlim(-0, 80)
plt.ylim(0, )
#print(dos[:, 1])
plt.show()

# Compute PDF
delr=0.1
rmax=10
rmin=1.0

for i in range(natom):
    for j in range(natom):
        if i != j:
         for it in range(5):
            rij=pos[it, i, :] - pos[it, j, :]
            rij=np.matmul(rij,aslat)


            if (i==1) & (j==2):

             np.where(rij < 0.5, rij-1.0,rij )
             #np.where(rij < -0.5, rij+1, rij)
             #print(rij)
