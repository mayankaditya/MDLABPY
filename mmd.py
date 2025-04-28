# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:44:30 2025

@author: Mayank
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import time
from datetime import datetime
global sid, natom, ntype, nntype, ntime, dynamics, sid, delt,slat,rij,delr,pdf,angleijk
global cal_msd, cal_pdf,cal_dos
cal_msd=0
cal_pdf=0
cal_dos=0
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)



#Reading trajectory
f = open('trajectory.lammpstrj', 'r')

#sid = 2
ntime = 2000
nomega = np.linspace(0, ntime)

norigin = 1000
delt = 0.02
delr=0.1
xf = fftfreq(ntime, delt)[:ntime]
time = range(ntime)
ntype = 3
atm=('Na','P','S')
nntype = (161, 54, 216)

natom = 161 + 54 + 216


ngrid=1000
pos = np.zeros((ntime, natom, 3),dtype=np.float64)
vel = np.zeros((ntime, natom, 3),dtype=np.float64)
dos = np.zeros((ntime, natom),dtype=np.float64)
pdf = np.zeros((ntype, ntype, ngrid),dtype=np.float64)
angleijk = np.zeros((ntype, ntype, ntype,360),dtype=np.float64)
slat=np.zeros((3,3),dtype=np.float64)
aslat=np.zeros((3,3),dtype=np.float64)
rij=np.zeros(3,dtype=np.float64)
msd = np.zeros((ntime, natom))
msdtype = np.zeros((ntime, ntype))
dostype = np.zeros((ntime, ntype))

dynamics = True

pdf[:,:,:]=0

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


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)













#Computing MSD 
from joblib import Parallel, delayed
import numpy as np

def compute_msd_for_it(it, pos, ntime, natom, norigin):
    if it + norigin > ntime:
        norigin_local = ntime - it
    else:
        norigin_local = norigin
    msd_it = np.zeros(natom)
    for i in range(natom):
        dummy = pos[it:it + norigin_local, i, :] - pos[0:norigin_local, i, :]
        msd_it[i] = np.tensordot(dummy, dummy, axes=2) / norigin_local
    return msd_it



# Parallel execution
if cal_msd:
    msd_list = Parallel(n_jobs=-1)(delayed(compute_msd_for_it)(it, pos, ntime, natom, norigin) for it in range(ntime))
    msd = np.array(msd_list)

# Accumulate per-type MSD
    for i in range(ntype):
        n1 = int(np.sum(nntype[:i]))
        n2 = int(np.sum(nntype[:i + 1]))
        msdtype[:, i] = np.sum(msd[:, n1:n2], axis=1)


    # Sequential calculation of MSD per type (this part might not benefit as much from simple parallelization
    # due to its nature of summing over atom types)
    msdtype = np.zeros((ntime, ntype))
    for i in range(ntype):
        n1 = int(np.sum(nntype[:i]))
        n2 = int(np.sum(nntype[:i + 1]))
        msdtype[:, i] = np.sum(msd[:, n1:n2], axis=1)

    # Now 'msd' and 'msdtype' contain the results calculated in a potentially faster way
    time_array = np.arange(ntime) * delt
    np.savetxt('msd.dat', np.c_[time_array, *[msdtype[:, i] for i in range(ntype)]])

    for i in range(ntype):
        plt.plot(time_array, msdtype[:, i], label=atm[i])
    plt.xlabel('Time(ps)')
    plt.ylabel('MSD')
    plt.title('Mean Squared Displacement per Type')
    plt.legend()
    plt.grid(True)
    plt.savefig('msd.png')
    plt.close()
    #plt.show()

if cal_dos:
    #Computing DOS
    from joblib import Parallel, delayed
    from numpy.fft import fft

    def compute_dos_per_atom(i):
        fx = fft(vel[:, i, 0])
        return fx * np.conj(fx)

    # Parallel FFT for each atom
    dos_list = Parallel(n_jobs=-1)(delayed(compute_dos_per_atom)(i) for i in range(natom))
    dos = np.stack(dos_list, axis=1)  # Shape: (ntime, natom)

    # Per-type DOS computation
    for i in range(ntype):
        n1 = int(np.sum(nntype[:i]))
        n2 = int(np.sum(nntype[:i + 1]))
        dostype[:, i] = (4.0 / ntime**2) * np.sum(dos[:, n1:n2], axis=1)
        norm=0.5*np.sum(dostype[:,i])*(xf[2]-xf[1])*4.136
        dostype[:,i]=dostype[:,i]/norm
    dostotal=np.dot(dostype,nntype)

    #dostype[0, :] = 0




    np.savetxt('dos.dat', np.c_[4.136*xf, *[dostype[:, i] for i in range(ntype)],dostotal])
    for i in range(ntype):
        plt.plot(4.136*xf[:], dostype[:, i], label=atm[i])
    plt.xlabel('E(meV)')
    plt.ylabel('DOS')
    plt.title('Phonon Density of States          ')
    plt.legend()
    plt.grid(True)
    plt.xlim(0,)
    plt.ylim(0,)
    plt.savefig('dos.png')
    plt.close()
    #plt.show()



#PDF
if cal_pdf:
    #Create Dicstionary
    delr=0.02
    rmax=10
    rmin=1.0
    stride=10
    dist=np.arange(0,delr*ngrid,delr)

    d={}
    for i in range(ntype):
        n1 = int(np.sum(nntype[:i]))
        n2 = int(np.sum(nntype[:i + 1]))
        for j in range(n1,n2):
         d.update({j:i})
    
    
    block_size = 100  # number of frames per block
    nblocks = ntime // block_size
    pdf = np.zeros((ntype, ntype, ngrid))
    
    
    def compute_pdf_block(block_idx, pos, delr, aslat, slat, d, ngrid, block_size):
        start = block_idx * block_size
        end = min(start + block_size, ntime)
        nsteps = end - start
        pdf_local = np.zeros((ntype, ntype, ngrid))
    
        for i in range(natom):
            i1 = d[i]
            for j in range(i + 1, natom):
                j1 = d[j]
                for it in range(start, end, stride):
                    rij = pos[it, i, :] - pos[it, j, :]
                    rij = np.matmul(aslat, rij)
                    rij = rij - np.round(rij)
                    rij = np.matmul(rij, slat)
                    rabs = np.linalg.norm(rij)
                    rbin = int(rabs / delr)
                    if 0 <= rbin < ngrid:
                        pdf_local[i1, j1, rbin] += 1.0
                    if i1 != j1:
                        pdf_local[j1, i1, rbin] += 1.0  # symmetric
        return pdf_local / nsteps
    results = Parallel(n_jobs=-1)(    delayed(compute_pdf_block)(b, pos, delr, aslat, slat, d, ngrid, block_size)    for b in range(nblocks))

    # Sum and average
    for r in results:
        pdf += r
    pdf /= nblocks
    
    fact = 4.0 * np.pi * dist * dist + 1e-12
    pdf /= fact
    
    # Output format — dynamically create columns
    cols_to_save = [pdf[i, j, :] for i in range(ntype) for j in range(i, ntype)]
    np.savetxt('pdf.data', np.c_[dist, *cols_to_save])

    for i in range(ntype):
        for j in range(i,ntype):
           plt.plot(dist,pdf[i, j, :],label=atm[i]+'-'+atm[j])
    plt.xlabel('Time')
    plt.ylabel('g(r)')
    #plt.title('Mean Squared Displacement per Type')
    plt.legend()
    plt.grid(True)
    
    plt.xlim(0,10)
    plt.ylim(0,)
    plt.savefig('pdf.png')
    plt.close()
    #plt.show()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)















#Bond_Angle

import numpy as np
import multiprocessing as mp
stride=10
# Assuming these variables are defined elsewhere in your script
# ntype: number of atom types
# nntype: array specifying the number of atoms of each type
# natom: total number of atoms
# pos: array of atomic positions (e.g., shape (time_steps, natom, 3))
# aslat: transformation matrix (likely for converting to fractional coordinates)
# slat: transformation matrix (likely for converting from fractional to Cartesian coordinates)
# ntime: number of time steps
# stride: step size for processing frames

# Create a dictionary to map atom index to its type
d = {}
start_index = 0
for i, count in enumerate(nntype):
    for j in range(start_index, start_index + count):
        d[j] = i
    start_index += count

def compute_bond_angle(atomA, atomB, atomC, aslat, slat, cutoff=4.0):
    """Computes the bond angle between three atoms.

    Args:
        atomA (np.ndarray): Coordinates of the first atom.
        atomB (np.ndarray): Coordinates of the central atom.
        atomC (np.ndarray): Coordinates of the third atom.
        aslat (np.ndarray): Transformation matrix for fractional coordinates.
        slat (np.ndarray): Transformation matrix for Cartesian coordinates.
        cutoff (float, optional): Maximum distance for considering a bond. Defaults to 3.0.

    Returns:
        float: The bond angle in degrees, or None if the bond distances exceed the cutoff.
    """
    BA = atomA - atomB
    BA = np.matmul(aslat, BA)
    BA = BA - np.round(BA)  # Apply periodic boundary conditions in fractional coordinates
    BA = np.matmul(BA, slat)

    BC = atomC - atomB
    BC = np.matmul(aslat, BC)
    BC = BC - np.round(BC)  # Apply periodic boundary conditions in fractional coordinates
    BC = np.matmul(BC, slat)

    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    if norm_BA < cutoff and norm_BC < cutoff:
        BA_unit = BA / norm_BA
        BC_unit = BC / norm_BC
        cos_angle = np.dot(BA_unit, BC_unit)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)
    return None

# Prepare angle histogram
angle_bins = 360
angleijk = np.zeros((ntype, ntype, ntype, angle_bins), dtype=np.float64)
theta = np.arange(angle_bins)

# Generate list of all valid (i, j, k) triplets based on initial frame
triplets = []
if pos.ndim > 1:  # Ensure 'pos' has at least one time frame
    for i in range(natom):
        for j in range(natom):
            for k in range(natom):
                if i !=j and i!=k and j!=k:
                 angle = compute_bond_angle(pos[0, i, :], pos[0, j, :], pos[0, k, :], aslat, slat)
                 if angle is not None :
                  triplets.append([i, j, k, d.get(i), d.get(j), d.get(k)])
'''










from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations

def process_triplet(triplet):
    i, j, k = triplet
    if i != j and i != k and j != k:
        angle = compute_bond_angle(pos[0, i, :], pos[0, j, :], pos[0, k, :], aslat, slat)
        if angle is not None:
            return [i, j, k, d.get(i), d.get(j), d.get(k)]
    return None

triplets = []

if pos.ndim > 1:
    all_triplets = permutations(range(natom), 3)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_triplet, triplet): triplet for triplet in all_triplets}
        for future in as_completed(futures):
            result = future.result()
            if result:
                triplets.append(result)
'''

# --- Worker Function ---
def process_frame(it, pos, aslat, slat, triplets, angle_bins):
    """Processes a single time frame to compute bond angle histograms.

    Args:
        it (int): The index of the current time frame.
        pos (np.ndarray): Array of atomic positions.
        aslat (np.ndarray): Transformation matrix for fractional coordinates.
        slat (np.ndarray): Transformation matrix for Cartesian coordinates.
        triplets (list): List of atom triplets to consider.
        angle_bins (int): Number of bins for the angle histogram.

    Returns:
        np.ndarray: A local histogram for the current time frame.
    """
    local_hist = np.zeros((ntype, ntype, ntype, angle_bins), dtype=np.float64)
    for i, j, k, i1, j1, k1 in triplets:
        angle = compute_bond_angle(pos[it, i, :], pos[it, j, :], pos[it, k, :], aslat, slat)
        if angle is not None:
            #print('angle',i,j,k,angle)
            index = int(round(angle)) % angle_bins
            #print(index)

            local_hist[i1, j1, k1, index] += 1
    return local_hist

# Run in parallel
if __name__ == "__main__":
    with mp.Pool() as pool:
        results = pool.starmap(process_frame, [(it, pos, aslat, slat, triplets, angle_bins) for it in range(0, ntime, stride)])

    # Sum all per-frame histograms
    for hist in results:
        angleijk += hist

    # Save to file
    cols_to_save = [angleijk[i1, j1, k1, :] for i1 in range(ntype) for j1 in range(i1, ntype) for k1 in range(j1, ntype)]
    np.savetxt('angle.dat', np.c_[theta, *cols_to_save])










#Bond Angle
'''
stride=10
d={}
for i in range(ntype):
    n1 = int(np.sum(nntype[:i]))
    n2 = int(np.sum(nntype[:i + 1]))
    for j in range(n1,n2):
     d.update({j:i})


import numpy as np
import multiprocessing as mp

def compute_bond_angle(atomA, atomB, atomC, cutoff=3.0):
    BA = atomA - atomB
    BA=np.matmul(aslat, BA)
    BA=BA-np.round(BA)
    BA=np.matmul(BA,slat)
    BC = atomC - atomB
    BC=np.matmul(aslat, BC)
    BC=BC-np.round(BC)
    BC=np.matmul(BC,slat)
    if np.linalg.norm(BA) < cutoff and np.linalg.norm(BC) < cutoff:
        BA_unit = BA / np.linalg.norm(BA)
        BC_unit = BC / np.linalg.norm(BC)
        cos_angle = np.dot(BA_unit, BC_unit)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)
    return None

# Prepare angle histogram
angle_bins = 360
angleijk = np.zeros((ntype, ntype, ntype, angle_bins), dtype=np.float64)
theta = np.arange(angle_bins)

# Generate list of all valid (i, j, k) triplets
triplets=[]
for i in range(natom):
        for j in range(natom):
            for k in range(natom):
                check=compute_bond_angle(pos[0,i,:],pos[0,j,:],pos[0,k,:])
                if check:
                    #a * a + b * b == c * c:
                 triplets.append([i,j,j,d[i],d[j],d[k]])

#print(len(triplets))
#print(triplets)
# --- Worker Function ---
def process_frame(it):
    local_hist = np.zeros((ntype, ntype, ntype, angle_bins), dtype=np.float64)
    for (i, j, k, i1, j1, k1) in triplets:
        angle = compute_bond_angle(pos[it, i], pos[it, j], pos[it, k])
        if angle is not None:
            print(angle)
            index = int(round(angle)) % angle_bins
            print(index)
            local_hist[i1, j1, k1, index] += 1
    return local_hist

# Run in parallel
if __name__ == "__main__":
    with mp.Pool() as pool:
        results = pool.map(process_frame, range(0, ntime, stride))

    # Sum all per-frame histograms
    for hist in results:
        angleijk += hist



    # Save to file
    #np.savetxt('angle.dat',np.c_[theta,(angleijk[i1,j1,k1,:] for i1 in range(ntype) for j1 in range(i1,ntype) for k1 in range(j1,ntype))])
    cols_to_save = [angleijk[i1, j1,k1, :] for i1 in range(ntype) for j1 in range(i1, ntype) for k1 in range(j1,ntype)]

    np.savetxt('angle.dat',np.c_[theta,*cols_to_save])
#np.savetxt('angle.dat',np.c_[theta,(((angleijk[i1,j1,k1,:] for i1 in range(ntype)) for j1 in range(i1,ntype)) for k1 in range(j1,ntype))])
'''



'''

import numpy as np
stride=10
# Compute Bond-Angle
def compute_bond_angle(atomA, atomB, atomC, cutoff=3.0):
    """
    Computes the bond angle (in degrees) between three atoms A-B-C
    if all distances are within the given cutoff.
    """
    BA = atomA - atomB
    BC = atomC - atomB

    if np.linalg.norm(BA) < cutoff and np.linalg.norm(BC) < cutoff:
        BA_unit = BA / np.linalg.norm(BA)
        BC_unit = BC / np.linalg.norm(BC)
        cos_angle = np.dot(BA_unit, BC_unit)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)
    else:
        return None

# Setup parameters
angle_bins = 360
angleijk = np.zeros((ntype, ntype, ntype, angle_bins), dtype=np.float64)
theta = np.arange(angle_bins)  # 0 to 359 degrees

# Loop over atom triplets
for i in range(natom):
    i1 = d[i]
    for j in range(i + 1, natom):
        j1 = d[j]
        for k in range(j + 1, natom):  # Avoid repeat triplets
            k1 = d[k]
            if len({i, j, k}) == 3:
                for it in range(0, ntime, stride):
                    angle = compute_bond_angle(pos[it, i, :], pos[it, j, :], pos[it, k, :])
                    if angle is not None:
                        index = int(round(angle)) % 360  # bin index from 0 to 359
                        angleijk[i1, j1, k1, index] += 1

# Save the angle histogram
with open('angle.dat', 'w') as f:
    for i1 in range(ntype):
        for j1 in range(i1, ntype):  # avoid symmetric duplicates
            for k1 in range(j1, ntype):
                hist = angleijk[i1, j1, k1, :]
                for t, val in zip(theta, hist):
                    f.write(f"{i1} {j1} {k1} {t} {val:.4f}\n")
                f.write("\n")  # blank line between triplets
'''




'''
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
#plt.plot(4.136*xf, np.sum(dostype[:, :],axis=1))
#plt.xlim(-0, 80)
#plt.ylim(0, )
#print(dos[:, 1])
#plt.show()

# Compute PDF


#Create Dicstionary

d={}
for i in range(ntype):
    n1 = int(np.sum(nntype[:i]))
    n2 = int(np.sum(nntype[:i + 1]))
    for j in range(n1,n2):
     d.update({j:i})


#print (d)

delr=0.02
rmax=10
rmin=1.0
dist=np.arange(0,delr*ngrid,delr)
#print(dist)
for i in range(natom):
    for j in range(natom):
        i1=d[i]
        j1=d[j]  
        if i != j:
         for it in range(ntime):
            rij=pos[it, i, :] - pos[it, j, :]
            rij=np.matmul(aslat,rij)
            rij=rij-np.round(rij)
            rij=np.matmul(rij,slat)
            rabs=np.linalg.norm(rij)

            rbin=int(rabs/delr)
            #print(rbin,i1,j1)
            pdf[i1,j1,rbin] += 1.0

            #if i1==0 and j1==0:
            #    print (pdf[i1,j1,rbin])
            #rabs=numpy.linalg.norm(rij)pdf[i,j,]
            #rabs=numpy.linalg.norm(rij)#if (i==1) & (j==2):
            
            #np.where(rij < 0.5, rij-1.0,rij )
            #np.where(rij < -0.5, rij+1, rij)
            #print(rij)
#print(pdf[0,0,:])
fact=4.0*np.pi*dist*dist
pdf[:,:,:]=pdf[:,:,:]/(fact)
#f=open('pdf.dat','w')
np.savetxt('pdf.data',np.c_[dist,pdf[0,0,:],pdf[0,1,:],pdf[0,2,:],pdf[1,1,:],pdf[1,2,:],pdf[2,2,:]])

end_time = time.perf_counter()
#f.write(dist,pdf[0,0,:]
plt.plot(dist,pdf[0,0,:])
plt.plot(dist,pdf[0,1,:])
plt.plot(dist,pdf[0,2,:])
plt.plot(dist,pdf[1,1,:])
plt.plot(dist,pdf[1,2,:])
plt.plot(dist,pdf[2,2,:])
plt.xlim(-0, 10)
plt.ylim(0, )
plt.show()
'''
