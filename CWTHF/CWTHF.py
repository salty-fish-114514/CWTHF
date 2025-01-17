
# mpiexec -np 20 python /home/limx/halo_identify/CWTHF/CWTHF.py --n_ref 400 --kw_low 0.1 --kw_high 2.5 --norm_pos 50000 --Delta_dens 4 --w_resolution 20 --snap_path /home/limx/testdata/snap_m50n512_151.hdf5 --pos_path PartType2/Coordinates --vel_path PartType2/Velocities --output_path /home/limx/halo_identify    
    

from mpi4py import MPI
import numpy as np
from functions import GDW3d,periodic_shift,get_paramters,scatter_CIC_density,CWT_calculate_array,CWT_calculate_by_grid,gather_CWT_from_processes,MPI_grid_maxima,amend_peaks,arrange_particle,MPI_halo_segment
import time 
import h5py

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

batchsize = 512**3/8

# MpcM_sun−1(km/s)2
G = 4.299*10**-9 
# M_sun, the mass resolution of the simulation
M = 95782383.78

L_box = 1000
n_min = 150

if rank == 0:    
    n_ref, kw_low, kw_high, norm_pos, Delta_dens, w_resolution, snap_path, pos_path, vel_path, output_path = get_paramters(comm, rank)
else:
    n_ref, kw_low, kw_high, norm_pos, Delta_dens, w_resolution = get_paramters(comm, rank)

# the larger l corresponds to a more linear w_array
l = 1.2
# set the range of scales
w_array = (np.geomspace(kw_low+l,kw_high+l,w_resolution)-l)

if rank == 0:
    t1 = time.time()
    f1 = h5py.File(snap_path,mode='r')
    particle_pos = f1[pos_path][...]
    particle_vel = f1[vel_path][...]

    particle_index = np.zeros(particle_pos.shape[0],dtype='int32')
    print('data load success, load ' + str(particle_pos.shape[0]) + ' particles',flush=True)
    calculating_peaks = np.empty
    calculating_peakvalue = np.empty
    average_density = particle_pos.shape[0]/50**3

    halo_dens = np.array([])
    index_wi = np.array([])

    halo_index = 1
    f1.close()

for i in range(w_array.shape[0]):
    n_grid = int(n_ref*w_array[i]+n_min)  
# to avoid too large data that causes truncate error, we split data in y-axis and scatter the data 
# in batches
    if rank == 0:
        Bcasted_data, recv_dens_grid = scatter_CIC_density(comm, size, rank, batchsize, n_grid, norm_pos, particle_pos)
    else:
        Bcasted_data, recv_dens_grid = scatter_CIC_density(comm, size, rank, batchsize, n_grid)

    gdw_resolution =  int(1.5/w_array[i]+5)

# calculate the gdw in one grid point
    x_gdw = y_gdw = z_gdw = np.linspace(-gdw_resolution,gdw_resolution,2*gdw_resolution+1) 
    X,Y,Z = np.meshgrid(x_gdw,y_gdw,z_gdw)
    GDW = GDW3d(w_array[i],(X)* L_box/n_grid,(Y)* L_box/n_grid,(Z)* L_box/n_grid)

# the cache for grid CWT with ghost cells extend gdw_resolution grid points in each direction
    grid = np.zeros([Bcasted_data[rank+1]-Bcasted_data[rank]+2*gdw_resolution,n_grid+2*gdw_resolution,n_grid+2*gdw_resolution])

    if w_array[i] < 0.6:
        CWT_calculate_array(4, n_grid, recv_dens_grid, gdw_resolution, X, Y, Z, GDW, grid)

    else:
        CWT_calculate_by_grid(recv_dens_grid, gdw_resolution, X, Y, Z, GDW, grid)

# gather the calculated CWT grid from every processes
    Gathered_data = gather_CWT_from_processes(comm, size, rank, batchsize, n_grid, Bcasted_data, gdw_resolution, grid)

# shift the boundary of the gathered CWT grid
    if rank == 0:
        grid = periodic_shift(Gathered_data,n_grid,gdw_resolution)
        grid = grid.astype('float32')

    if rank == 0:
        recv_counts, disp, peaks, relative_peak = MPI_grid_maxima(comm, size, rank, batchsize, grid, n_grid)
    else:
        grid = None
        relative_peak = MPI_grid_maxima(comm, size, rank, batchsize, grid, n_grid)

    if rank == 0:
        amend_peak_value = np.zeros(peaks.shape[0])  
# amend the value of peaks in grid caused finite resolution
        amend_peaks(grid, peaks, amend_peak_value, n_ref, n_min, i, w_array) 

# in circulation more than 0, we arrange the particles utilizing s(egmented grid i-1) and (grid i)
        if i != 0:
            halo_index,confirmed_dens = arrange_particle(n_ref, n_min, n_grid, L_box, norm_pos, w_array, G, M, particle_pos, particle_vel, particle_index, grid, average_density, Delta_dens, halo_index, i, halo_size, calculating_peaks, calculating_peakvalue, Gathered_result)

            halo_dens = np.concatenate((halo_dens,confirmed_dens))
            index_wi = np.append(index_wi,halo_index)

            t2 = time.time()
            print('current halo number is: '+str(halo_index),flush=True)
            print(f"time used by {size} process is: {t2-t1} seconds",flush=True)

# store the position and value of these peaks        
        calculating_peaks = peaks
        calculating_peakvalue = amend_peak_value   

# segment the CWT grid and gather the peaks, sizes, and segment result
    if rank == 0:
        halo_size, Gathered_peaks, Gathered_result = MPI_halo_segment(comm, size, rank, batchsize, grid, n_grid, gdw_resolution, relative_peak)
   
    else:
        grid=None
        MPI_halo_segment(comm, size, rank, batchsize, grid, n_grid, gdw_resolution, relative_peak)


if rank == 0:
    empty_grid = np.zeros([n_grid,n_grid,n_grid])
    ti = time.time()
    halo_index,confirmed_dens = arrange_particle(n_ref, n_min, n_grid, L_box, norm_pos, w_array, G, M, particle_pos, particle_vel, particle_index, empty_grid, average_density, Delta_dens, halo_index, i, halo_size, calculating_peaks, calculating_peakvalue, Gathered_result)

    halo_dens = np.concatenate((halo_dens,confirmed_dens))
    index_wi = np.append(index_wi,halo_index)
    
    t2 = time.time()
    print('current halo number is: '+str(halo_index),flush=True)
    print(f"time used by {size} process is: {t2-t1} seconds",flush=True)


if rank == 0:            
    print(f"total halo particles are: {particle_index[particle_index!=0].shape[0]}")
    np.save(output_path+r'/CWTHF'+str(kw_low)+r'-'+str(kw_high)+r'_delta'+str(Delta_dens)+r'_resolution'+str(n_ref)+'_w'+str(w_resolution)+r'.npy',particle_index)
    np.save(output_path+r'/CWTHF_index'+str(kw_low)+r'-'+str(kw_high)+r'_delta'+str(Delta_dens)+r'_resolution'+str(n_ref)+'_w'+str(w_resolution)+r'.npy',index_wi)