# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:44:30 2025

@author: Mayank
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft,fftfreq
global sid,natom,ntype,nntype,ntime,dynamics,sid,delt



 
f=open('trajectory.lammpstrj','r')

sid=2
ntime=1024
nomega=np.linspace(0, ntime)

norigin=100
delt=0.01
xf = fftfreq(ntime, delt)[:ntime]
time=range(ntime)
ntype=3
nntype=(161, 54, 216)
natom=161+54+216
pos=np.zeros((ntime,natom,3))
vel=np.zeros((ntime,natom,3))
dos=np.zeros((ntime,natom))

msd=np.zeros((ntime,natom))
msdtype=np.zeros((ntime,ntype))

dynamics=True

for it in range(ntime):
 f.readline()   #Comment
 f.readline()#Time step
 f.readline()#Natom
 f.readline()#Comment box-bounds
 f.readline()#Comment box-bounds

 amin,amax=(f.readline()).split()  #Comment box-bounds
 bmin,bmax=f.readline().split()  #Comment box-bounds
 cmin,cmax=f.readline().split()  #Comment box-bounds
 vals=f.readline().split()  #Comment
 #print(len(vals))
 if len==7:
  sid=2
 for i in range(natom):
   pos[it,i,:]=f.readline().split()[sid:]
   if dynamics and it < ntime-1:
    vel[it,i,:]=(pos[it+1,i,:]-pos[it,i,:] )/delt   
   #print(pos[it,i,:])


print((vel[:,1,0]).shape)



#Compute MSD
for it in range(ntime):
    if it+norigin > ntime:
        norigin=ntime-it
    for i in range(natom):
     dummy=pos[it:it+norigin,i,:]-pos[0:norigin,i,:]
     msd[it,i]=np.tensordot(dummy, dummy, axes=2)/norigin
     #if i==1:
     # print(it,msd[it,i])

for i in range(ntype):
 n1=int(np.sum(nntype[:i]))
 n2=int(np.sum(nntype[:i+1]))
 #print(n1,n2)  
 msdtype[:,i] =np.sum(msd[:,n1:n2],axis=1)     
#plt.plot(time,msdtype[:,0])
#plt.plot(time,msdtype[:,1])
#plt.plot(time,msdtype[:,2])

#Compute DOS
if dynamics:
 for i in range(natom):
  fx=fft(vel[:,i,0])
  dos[:,i]=fx*np.conj(fx)
#print(dos[:,1])
dos[0,:]=0
plt.plot(xf,dos[:,1])
plt.xlim(0,100)
print(dos[:,1])
print(dos[:,1])
plt.show()     
      
