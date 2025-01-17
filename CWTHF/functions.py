import numpy as np
import h5py
import numba
from mpi4py import MPI
from halo_segment import expand_structure3d1
from CWT_calculate_by_grid_cy import CWT_calculate_by_grid_cy
import argparse
import scipy.ndimage

def GDW3d(w,X,Y,Z):
    R = np.sqrt((X)**2 + (Y)**2 + (Z)**2)
    return w**(3/2) * (6 - (R*w)**2) * np.exp(-(R*w)**2/4)


def grid_local_maxima(grid,threshold=0):
# this function calculate all the local maxima of the input 3d array 'grid'.
# this version works well with the dense maxima distribution
    n1 = grid.shape[0]
    n2 = grid.shape[1]
    n3 = grid.shape[2]

    result = np.ones([n1-2,n2-2,n3-2],dtype='bool')
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                if (i==0 and j==0 and k==0):
                    temp = (grid[1:n1-1,1:n2-1,1:n3-1]>threshold)
                    result *= temp

                temp = grid[1:n1-1,1:n2-1,1:n3-1]-grid[1-i:n1-1-i,1-j:n2-1-j,1-k:n3-1-k]
                temp = (temp>=0)
                result *= temp
    del temp
    del grid
    x,y,z = result.nonzero()
    return x+1,y+1,z+1

def periodic_shift(grid,n_grid,gdw_resolution):
# set gdw_resolution grids in each direction for cache of the peroidic bondary condition
# 'point'
    grid[gdw_resolution:2 * gdw_resolution,gdw_resolution:2 * gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[n_grid+gdw_resolution:,n_grid+gdw_resolution:,n_grid+gdw_resolution:]
    grid[gdw_resolution:2 * gdw_resolution,gdw_resolution:2 * gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[n_grid+gdw_resolution:,n_grid+gdw_resolution:,0:gdw_resolution]
    grid[gdw_resolution:2 * gdw_resolution,n_grid:n_grid + gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[n_grid+gdw_resolution:,0:gdw_resolution,n_grid+gdw_resolution:]
    grid[gdw_resolution:2 * gdw_resolution,n_grid:n_grid + gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[n_grid+gdw_resolution:,0:gdw_resolution,0:gdw_resolution]
    grid[n_grid:n_grid + gdw_resolution,gdw_resolution:2 * gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[0:gdw_resolution,n_grid+gdw_resolution:,n_grid+gdw_resolution:]
    grid[n_grid:n_grid + gdw_resolution,gdw_resolution:2 * gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[0:gdw_resolution,n_grid+gdw_resolution:,0:gdw_resolution]
    grid[n_grid:n_grid + gdw_resolution,n_grid:n_grid + gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[0:gdw_resolution,0:gdw_resolution,n_grid+gdw_resolution:]
    grid[n_grid:n_grid + gdw_resolution,n_grid:n_grid + gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[0:gdw_resolution,0:gdw_resolution,0:gdw_resolution]
    # 'line'
    grid[gdw_resolution:n_grid+gdw_resolution,n_grid:n_grid + gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,0:gdw_resolution,0:gdw_resolution]
    grid[gdw_resolution:n_grid+gdw_resolution,n_grid:n_grid + gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,0:gdw_resolution,n_grid+gdw_resolution:]
    grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:2 * gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,n_grid+gdw_resolution:,0:gdw_resolution]
    grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:2 * gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,n_grid+gdw_resolution:,n_grid+gdw_resolution:]

    grid[n_grid:n_grid + gdw_resolution,gdw_resolution:n_grid+gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[0:gdw_resolution,gdw_resolution:n_grid+gdw_resolution,0:gdw_resolution]
    grid[n_grid:n_grid + gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[0:gdw_resolution,gdw_resolution:n_grid+gdw_resolution,n_grid+gdw_resolution:]
    grid[gdw_resolution:2 * gdw_resolution,gdw_resolution:n_grid+gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[n_grid+gdw_resolution:,gdw_resolution:n_grid+gdw_resolution,0:gdw_resolution]
    grid[gdw_resolution:2 * gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[n_grid+gdw_resolution:,gdw_resolution:n_grid+gdw_resolution,n_grid+gdw_resolution:]

    grid[n_grid:n_grid + gdw_resolution,n_grid:n_grid + gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[0:gdw_resolution,0:gdw_resolution,gdw_resolution:n_grid+gdw_resolution]
    grid[n_grid:n_grid + gdw_resolution,gdw_resolution:2 * gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[0:gdw_resolution,n_grid+gdw_resolution:,gdw_resolution:n_grid+gdw_resolution]
    grid[gdw_resolution:2 * gdw_resolution,n_grid:n_grid + gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[n_grid+gdw_resolution:,0:gdw_resolution,gdw_resolution:n_grid+gdw_resolution]
    grid[gdw_resolution:2 * gdw_resolution,gdw_resolution:2 * gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[n_grid+gdw_resolution:,n_grid+gdw_resolution:,gdw_resolution:n_grid+gdw_resolution]
    # 'plane'
    grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution,n_grid:n_grid + gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution,0:gdw_resolution]
    grid[gdw_resolution:n_grid+gdw_resolution,n_grid:n_grid + gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,0:gdw_resolution,gdw_resolution:n_grid+gdw_resolution]
    grid[n_grid:n_grid + gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[0:gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution]
    grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:2 * gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution,n_grid+gdw_resolution:]
    grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:2 * gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[gdw_resolution:n_grid+gdw_resolution,n_grid+gdw_resolution:,gdw_resolution:n_grid+gdw_resolution]
    grid[gdw_resolution:2 * gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution] += grid[n_grid+gdw_resolution:,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution]

    return grid[gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution,gdw_resolution:n_grid+gdw_resolution]

def CWT_calculate_by_grid(recv_dens_grid, gdw_resolution, X, Y, Z, GDW, grid):
    grid_x, grid_y, grid_z = np.where(recv_dens_grid > 0) 

    R = np.sqrt(X**2+Y**2+Z**2)
    r = np.sort(np.unique(R))
    r = r[r<=gdw_resolution]
    for j in range(r.shape[0]):
        x,y,z = np.where(R==r[j])
        x -= gdw_resolution
        y -= gdw_resolution
        z -= gdw_resolution

        x = x.astype('int16')
        y = y.astype('int16')
        z = z.astype('int16')

        grid_x = grid_x.astype('int16')
        grid_y = grid_y.astype('int16')
        grid_z = grid_z.astype('int16')

        CWT_calculate_by_grid_cy(gdw_resolution, recv_dens_grid, grid_x, grid_y, grid_z, GDW, x, y, z, grid)

def halo_segmentation3d(data, peaks,labels=None):
# This array store the structures we find in every loop and after the loop is finished, this array 
# stores the result structures
# input 'data' should be a 3d array, peaks should be a array of shape (x,3) that stores the x,y,z coordinate 
# of the local maxima
    result = np.zeros(data.shape,dtype='int32')

# This array store the grid we consider to extend in every loop. In this method, we continue extend 
# the structures outwards for 1 grid in each loop until the structure no longer include new grid. To 
# simplify the calculation, this array only store the possible new grid, not include the already stored 
# grids.
    extend_index = np.zeros(data.shape,dtype='bool')



# Seeding the initial structures, every structure only consists of 1 grid at this time.
    if np.any(labels == None):
        structure_index = 1
        for i in range(peaks.shape[0]):
            result[peaks[i,0],peaks[i,1],peaks[i,2]] += structure_index
            structure_index += 1

    else:
        for i in range(peaks.shape[0]):
            result[peaks[i,0],peaks[i,1],peaks[i,2]] += labels[i]

    x,y,z = np.where(result.astype('bool'))
    x = x.astype('int16')
    y = y.astype('int16')
    z = z.astype('int16')
    j=0
    while x.shape[0]!=0 and j<1000 :
# this array stores the information of already stored grids, which is just the bool version of array 'result'
        x,y,z = expand_structure3d1(data, result, x, y, z)

        boundary_index = (x==(result.shape[0]-1))
        boundary_index = np.logical_or(boundary_index,x==0)
        boundary_index = np.logical_or(boundary_index,y==result.shape[1]-1)
        boundary_index = np.logical_or(boundary_index,y==0)
        boundary_index = np.logical_or(boundary_index,z==result.shape[2]-1)
        boundary_index = np.logical_or(boundary_index,z==0)

        index = np.logical_not(boundary_index)
        x = x[index].astype('int16')
        y = y[index].astype('int16')
        z = z[index].astype('int16')
        j+=1
    return result,j

def CWT_calculate_array(CWT_batch, n_grid, recv_dens_grid, gdw_resolution, X, Y, Z, GDW, grid):
    R = np.sqrt(X**2+Y**2+Z**2)
    r = np.sort(np.unique(R))
    r = r[r<=gdw_resolution]
    for j in range(r.shape[0]):
        x,y,z = np.where(R==r[j])
        x -= gdw_resolution
        y -= gdw_resolution
        z -= gdw_resolution

        grid_start_y = np.zeros(CWT_batch+1,dtype='int')

        for k in range(CWT_batch):
            grid_start_y[k] = int((n_grid)/CWT_batch)*k

        grid_start_y[-1] = n_grid

        for l in range(CWT_batch):
            weighted_dens = GDW[x[0]+gdw_resolution,y[0]+gdw_resolution,z[0]+gdw_resolution] * recv_dens_grid[:,grid_start_y[l]:grid_start_y[l+1],:]

            for k in range(x.shape[0]):
                grid[gdw_resolution+x[k]:gdw_resolution+x[k]+recv_dens_grid.shape[0],gdw_resolution+y[k]+grid_start_y[l]:gdw_resolution+y[k]+grid_start_y[l+1],gdw_resolution+z[k]:gdw_resolution+z[k]+n_grid] += weighted_dens

def scatter_CIC_density(comm, size, rank, batchsize, n_grid, norm_pos=None, particle_pos=None):

    length_y = int(batchsize/n_grid**2)
    if int(n_grid/length_y)<n_grid/length_y:
        number_of_batch = int(n_grid/length_y)+1
        residual_length = n_grid - int(n_grid/length_y)*length_y
    else:
        number_of_batch = int(n_grid/length_y)
        residual_length = length_y       

    if rank == 0:
        dens_grid = CIC3d_batch(n_grid,particle_pos,norm_pos)
        grid_x, grid_y, grid_z = np.where(dens_grid > 0)
# the size of grid points per process        
        size_per_process = int(grid_x.shape[0]/size)
# split grid data based on their x-axis coordinates so that each process contains simliar data size
# the index of x-axis grid to split data
        index_gridx = np.zeros(size+1,dtype ='i')
        for j in range(size-1):
            index_gridx[j+1] = grid_x[size_per_process*(j+1)]
        index_gridx[-1] = n_grid

# bcast index in x-axis of data, thus we can set proper cache
    if rank == 0: 
        Bcasted_data = index_gridx
    else:                                                                      
        Bcasted_data = np.empty(size+1,dtype='i')                                                            
    comm.Bcast(Bcasted_data, root=0)                                            


# cache for receiving scattered data
    recv_dens_grid = np.zeros([Bcasted_data[rank+1]-Bcasted_data[rank],n_grid,n_grid])

    for j in range(number_of_batch):
        recv_shapes = []
        if j != number_of_batch-1:
            for k in range(size):
                recv_shapes.append((Bcasted_data[k+1]-Bcasted_data[k],length_y,n_grid))
        else:
            for k in range(size):
                recv_shapes.append((Bcasted_data[k+1]-Bcasted_data[k],residual_length,n_grid))

        recv_counts = [np.prod(shape) for shape in recv_shapes]
    
        disp = [0]

# 计算每个进程的位移
        for k in range(1, len(recv_counts)):
            disp.append(disp[k-1] + recv_counts[k-1])

        recvbuf = np.empty(recv_shapes[rank], dtype='float64')

        if rank == 0:
            data_scatter = np.ascontiguousarray(dens_grid[:,length_y*j:length_y*j+recv_shapes[0][1],:])

        else:
            data_scatter = None

        comm.Scatterv([data_scatter, recv_counts, disp, MPI.DOUBLE],
                    recvbuf,
                    root=0)

        recv_dens_grid[:,length_y*j:length_y*j+recv_shapes[0][1],:] = recvbuf
    return Bcasted_data,recv_dens_grid


def gather_CWT_from_processes(comm, size, rank, batchsize, n_grid, Bcasted_data, gdw_resolution, grid):
    length_y = int(batchsize/(n_grid+2*gdw_resolution)**2)
    if int((n_grid+2*gdw_resolution)/length_y)<(n_grid+2*gdw_resolution)/length_y:
        number_of_batch = int((n_grid+2*gdw_resolution)/length_y)+1
        residual_length = (n_grid+2*gdw_resolution) - int((n_grid+2*gdw_resolution)/length_y)*length_y
    else:
        number_of_batch = int((n_grid+2*gdw_resolution)/length_y)
        residual_length = length_y  

    if rank == 0:
        Gathered_data = np.zeros([n_grid+2*gdw_resolution,n_grid+2*gdw_resolution,n_grid+2*gdw_resolution])

    for j in range(number_of_batch):
        recv_shapes = []
        if j != number_of_batch-1:
            for k in range(size):
                recv_shapes.append((Bcasted_data[k+1]-Bcasted_data[k]+2*gdw_resolution,length_y,n_grid+2*gdw_resolution))

        else:
            for k in range(size):
                recv_shapes.append((Bcasted_data[k+1]-Bcasted_data[k]+2*gdw_resolution,residual_length,n_grid+2*gdw_resolution))

        recv_counts = [np.prod(shape) for shape in recv_shapes]
    
        disp = [0]

        # 计算每个进程的位移
        for k in range(1, len(recv_counts)):
            disp.append(disp[k-1] + recv_counts[k-1])

        if rank == 0:
            recvbuf = np.empty(sum(recv_counts), dtype=grid.dtype)
        else:
            recvbuf = None

        data_gather = np.ascontiguousarray(grid[:,length_y*j:length_y*j+recv_shapes[0][1],:])

        comm.Gatherv(data_gather.ravel(), [recvbuf, recv_counts, disp, MPI.DOUBLE], root=0)

        if rank == 0:
            start_index = 0
            for k in range(size):
                Gathered_data[Bcasted_data[k]:Bcasted_data[k+1]+2*gdw_resolution,length_y*j:length_y*j+recv_shapes[0][1],:] += recvbuf[start_index:start_index + recv_counts[k]].reshape(recv_shapes[k])
                start_index += recv_counts[k]

    if rank == 0:
        return Gathered_data
    else:
        return None

def gather_1d_data(comm, size, rank, halo_size):
    number_of_halo = None
    if rank == 0:
        number_of_halo = np.empty((size), dtype='int32')

    number_of_halo_process = np.array([halo_size.shape[0]], dtype='int32')  
    comm.Gather(number_of_halo_process, number_of_halo, root=0)
    if rank == 0:
        recv_counts = number_of_halo.tolist()

        disp = [0]

            # 计算每个进程的位移
        for k in range(1, len(recv_counts)):
            disp.append(disp[k-1] + recv_counts[k-1])
    else:
        recv_counts = None    

    if rank == 0:
        recvbuf = np.empty(sum(recv_counts), dtype='int32')
    else:
        recvbuf = None
        disp = None

# gather the size of halos in number of grid points
    comm.Gatherv(halo_size, [recvbuf, recv_counts, disp, MPI.INT], root=0)

    if rank == 0:
        return recv_counts,disp,recvbuf,number_of_halo
    else:
        return recv_counts,disp,None,number_of_halo    

def gather_halogrid(comm, size, rank, n_grid, CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, result, number_of_halo, Gathered_result=None):
    for j in range(number_of_batch):
        recv_shapes = []
        if j != number_of_batch-1:
            for k in range(size):
                recv_shapes.append((CWTgrid_size[k]+extend_left[k]+extend_right[k],length_y,n_grid))
        else:
            for k in range(size):
                recv_shapes.append((CWTgrid_size[k]+extend_left[k]+extend_right[k],residual_length,n_grid))

        recv_counts = [np.prod(shape) for shape in recv_shapes]
        
        disp = [0]

# 计算每个进程的位移
        for k in range(1, len(recv_counts)):
            disp.append(disp[k-1] + recv_counts[k-1])

        if rank == 0:
            recvbuf = np.empty(sum(recv_counts), dtype='int32')
        else:
            recvbuf = None

        data_gather = np.ascontiguousarray(result[:,length_y*j:length_y*j+recv_shapes[0][1],:]).astype('int32')

        comm.Gatherv(data_gather.ravel(), [recvbuf, recv_counts, disp, MPI.INT], root=0)

        if rank == 0:
            start_index = 0
            halo_index_process = 0
            for k in range(size):
                recv_result = recvbuf[start_index:start_index + recv_counts[k]].reshape(recv_shapes[k])
                recv_result[recv_result>0] += halo_index_process
                Gathered_result[grid_x_index_start[k]-extend_left[k]:grid_x_index_start[k]+CWTgrid_size[k]+extend_right[k],length_y*j:length_y*j+recv_shapes[0][1],:] += recv_result
                start_index += recv_counts[k]
                halo_index_process += number_of_halo[k]

@numba.jit(nopython=True)
def particle_assign(particle_pos, particle_index, result, index, halo_ID):
    for k in range(particle_pos[particle_index==0].shape[0]):
        halo_ID[k] += result[index[k,0],index[k,1],index[k,2]]

@numba.jit(nopython=True)
def particle_in_halo_calculate(halo_ID, remaining_particle, halo_number):
    for k in range(remaining_particle.shape[0]):
        halo_number[halo_ID[k]] += 1
        
@numba.jit(nopython=True)
def cross_scale_maximum_check(n_ref, n_min, L_box, w_array, grid, peak, peakvalue, j, n_grid, result, dense_halo, halo_index_temp):
    for i in range(peak.shape[0]):
        if dense_halo[result[peak[i,0],peak[i,1],peak[i,2]]-1]:
            distance = peak[i,:] * L_box/int(n_ref*w_array[j-1]+n_min)
            next_n_grid = n_grid
            next_index = (distance * next_n_grid/L_box).astype(np.int32)

            if (np.all(peakvalue[i]>grid[next_index[0]-int(0.2/w_array[j-1]+3):next_index[0]+int(0.2/w_array[j-1]+3),next_index[1]-int(0.2/w_array[j-1]+3):next_index[1]+int(0.2/w_array[j-1]+3),next_index[2]-int(0.2/w_array[j-1]+3):next_index[2]+int(0.2/w_array[j-1]+3)])):          
                halo_index_temp.append(result[peak[i,0],peak[i,1],peak[i,2]])

def scatter_CWT_with_ghostcell(comm, size, rank, batchsize, n_grid, ghost_cell_size, calculating_grid = None):

# set the size of CWT grid in each process
    n_grid_per_process = int(n_grid/size)
    residual_n_grid = n_grid - n_grid_per_process*size

    CWTgrid_size = np.ones(size,dtype='int')*n_grid_per_process
    for j in range(residual_n_grid):
        CWTgrid_size[j] += 1

# the start index where we cut the grid CWT
    grid_x_index_start = np.zeros(size,dtype='int')
    for j in range(size-1):
        grid_x_index_start[j+1] += grid_x_index_start[j] + CWTgrid_size[j]

# the extended grid points in left and right direction
    extend_left = np.ones(size,dtype='int') * ghost_cell_size
    extend_right = np.ones(size,dtype='int') * ghost_cell_size

    for j in range(size):
        if extend_left[j]>grid_x_index_start[j]:
            extend_left[j]=grid_x_index_start[j]
        else:
            break

    for j in range(1,size):
        if extend_right[-j]>n_grid-(grid_x_index_start[-j]+CWTgrid_size[-j]):
            extend_right[-j]=n_grid-(grid_x_index_start[-j]+CWTgrid_size[-j])
        else:
            break

# to avoid too large data that causes truncate error, we split data in y-axis and scatter the data 
# in batches
    length_y = int(batchsize/(n_grid*(n_grid+ghost_cell_size*2*size)))
    if int(n_grid/length_y)<n_grid/length_y:
        number_of_batch = int(n_grid/length_y)+1
        residual_length = n_grid - int(n_grid/length_y)*length_y
    else:
        number_of_batch = int(n_grid/length_y)
        residual_length = length_y   

    recv_CWT_grid = np.zeros([CWTgrid_size[rank]+extend_left[rank]+extend_right[rank],n_grid,n_grid])

    for j in range(number_of_batch):
        recv_shapes = []
        if j != number_of_batch-1:
            for k in range(size):
                recv_shapes.append((CWTgrid_size[k]+extend_left[k]+extend_right[k],length_y,n_grid))
        else:
            for k in range(size):
                recv_shapes.append((CWTgrid_size[k]+extend_left[k]+extend_right[k],residual_length,n_grid))

        recv_counts = [np.prod(shape) for shape in recv_shapes]
        
        disp = [0]

# 计算每个进程的位移
        for k in range(1, len(recv_counts)):
            disp.append(disp[k-1] + recv_counts[k-1])

        recvbuf = np.empty(recv_shapes[rank], dtype='float32')

        if rank == 0:
            data_scatter = np.zeros((n_grid+extend_left.sum()+extend_right.sum(),recv_shapes[0][1],n_grid), dtype='float32')
            index_x = 0
            for k in range(size):
                data_scatter[index_x:index_x+CWTgrid_size[k]+extend_left[k]+extend_right[k]] = calculating_grid[grid_x_index_start[k]-extend_left[k]:grid_x_index_start[k]+CWTgrid_size[k]+extend_right[k],length_y*j:length_y*j+recv_shapes[0][1],:]
                index_x += CWTgrid_size[k]+extend_left[k]+extend_right[k]

        else:
            data_scatter = None

        comm.Scatterv([data_scatter, recv_counts, disp, MPI.FLOAT],
                        recvbuf,
                        root=0)

        recv_CWT_grid[:,length_y*j:length_y*j+recv_shapes[0][1],:] = recvbuf
    return CWTgrid_size,grid_x_index_start,extend_left,extend_right,length_y,number_of_batch,residual_length,recv_CWT_grid


def locate_maxima_in_ghostcell(size, rank, CWTgrid_size, extend_left, recv_CWT_grid, relative_peak):
    peaks = relative_peak  
    if rank != 0:
        peaks[:,0] += (extend_left[rank]-1)    
# the max index of true halo
    halo_index_max = peaks.shape[0]   

# maxima in left ghost cell
    if rank != 0:
        x,y,z = grid_local_maxima(recv_CWT_grid[:extend_left[rank]+1,:,:])
        peaks_left=np.array([x,y,z]).T
        peaks = np.concatenate((peaks,peaks_left))

# maxima in right ghost cell
    if rank != size-1:
        x,y,z = grid_local_maxima(recv_CWT_grid[extend_left[rank]+CWTgrid_size[rank]-1:,:,:])
        peaks_right=np.array([x+extend_left[rank]+CWTgrid_size[rank]-1,y,z]).T
        peaks = np.concatenate((peaks,peaks_right))
    return peaks,halo_index_max

def MPI_halo_segment(comm, size, rank, batchsize, calculating_grid, n_grid, gdw_resolution, relative_peak):
    # scatter grid CWT with ghost cell to segment them
    if rank == 0:
        CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, recv_CWT_grid = scatter_CWT_with_ghostcell(comm, size, rank, batchsize, n_grid, int(2.5*gdw_resolution+6), calculating_grid)
    else:
        CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, recv_CWT_grid = scatter_CWT_with_ghostcell(comm, size, rank, batchsize, n_grid, int(2.5*gdw_resolution+6))

# locate the maxima in cut grid CWT, does not include ghost cells
    peaks, halo_index_max = locate_maxima_in_ghostcell(size, rank, CWTgrid_size, extend_left, recv_CWT_grid, relative_peak)
     
    result, iteration_time = halo_segmentation3d(recv_CWT_grid, peaks)

# eliminate the structures that correspond to maxima in ghost cells
    result[result>halo_index_max] = 0

# The array 'halo_size' starts with index 1, so the index is equal to the index of the structures, 
# which also starts with 1.
    halo_size = scipy.ndimage.sum(input=np.ones(result.shape),labels=result,index=np.linspace(1,result.max(),result.max()).astype('int')).astype('int32')

# gather the number of halos in each process thus we can set proper cache
    recv_counts, disp, recvbuf, number_of_halo = gather_1d_data(comm, size, rank, halo_size)

    if rank == 0:
# store the gathered halo size
        halo_size = recvbuf
# Gathered_peaks stores the all the peaks in CWT grid            
        Gathered_peaks = np.zeros([halo_size.shape[0],3], dtype='int32')
# Gathered_result stores the result of halo segmentation
        Gathered_result = np.zeros([n_grid,n_grid,n_grid], dtype=result.dtype)

# adjust the position of peaks in x-axis to the full grid instead of the cut grid
    if rank != 0:
        peaks[0:halo_index_max,0] -= (extend_left[rank] - grid_x_index_start[rank])

    for j in range(3):
        if rank == 0:
            recvbuf = np.empty(sum(recv_counts), dtype='int32')
        else:
            recvbuf = None

        data_gather = np.ascontiguousarray(peaks[0:halo_index_max,j]).astype('int32')
    # gather the local maxima that corresponds to structures
        comm.Gatherv(data_gather, [recvbuf, recv_counts, disp, MPI.INT], root=0)

        if rank == 0:
            Gathered_peaks[:,j] = recvbuf

# gather the segemented halo grid
    if rank == 0:
        gather_halogrid(comm, size, rank, n_grid, CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, result, number_of_halo, Gathered_result)
        return halo_size,Gathered_peaks,Gathered_result        
    else:
        gather_halogrid(comm, size, rank, n_grid, CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, result, number_of_halo)

def MPI_grid_maxima(comm, size, rank, batchsize, grid, n_grid):
    # scatter grid CWT with ghost cell to segment them
    if rank == 0:
        CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, recv_CWT_grid = scatter_CWT_with_ghostcell(comm, size, rank, batchsize, n_grid, 1, grid)
    else:
        CWTgrid_size, grid_x_index_start, extend_left, extend_right, length_y, number_of_batch, residual_length, recv_CWT_grid = scatter_CWT_with_ghostcell(comm, size, rank, batchsize, n_grid, 1)

# locate the maxima in cut grid CWT, does not include ghost cells
    if rank == 0:
        x,y,z = grid_local_maxima(recv_CWT_grid[extend_left[rank]:extend_left[rank]+CWTgrid_size[rank]+1,:,:])
        peaks=np.array([x+extend_left[rank],y,z]).T

    elif rank == size-1:
        x,y,z = grid_local_maxima(recv_CWT_grid[extend_left[rank]-1:extend_left[rank]+CWTgrid_size[rank],:,:])
        peaks=np.array([x+extend_left[rank]-1,y,z]).T

    else:
        x,y,z = grid_local_maxima(recv_CWT_grid[extend_left[rank]-1:extend_left[rank]+CWTgrid_size[rank]+1,:,:])
        peaks=np.array([x+extend_left[rank]-1,y,z]).T

    relative_peaks = peaks.copy()
# adjust the position of peaks in x-axis to the full grid instead of the cut grid
    if rank != 0:
        peaks[:,0] -= (extend_left[rank] - grid_x_index_start[rank])

# gather peaks from processes
    recv_counts, disp, recvbuf, number_of_peaks = gather_1d_data(comm, size, rank, peaks[:,0].astype('int32'))
    if rank == 0:
        Gathered_peaks = np.zeros([recvbuf.shape[0],3], dtype='int32')
        Gathered_peaks[:,0] += recvbuf
    recv_counts, disp, recvbuf, number_of_peaks = gather_1d_data(comm, size, rank, peaks[:,1].astype('int32'))
    if rank == 0:
        Gathered_peaks[:,1] += recvbuf
    recv_counts, disp, recvbuf, number_of_peaks = gather_1d_data(comm, size, rank, peaks[:,2].astype('int32'))
    if rank == 0:    
        Gathered_peaks[:,2] += recvbuf

    if rank == 0:
        return recv_counts, disp, Gathered_peaks, relative_peaks
    else:
        return relative_peaks
    
def selfboundness_check(G, M, particle_pos, particle_vel, halo_ID, remaining_particle, halo_number, particle_index_start, arranged_ID, halo_index_temp, confirmed_index):
    for j in range(halo_index_temp.shape[0]):
# 'indices' stores the index of particles that belong to the halo halo_index_temp[j]. halo_index_temp[j] means the j-th halo after 
# cross-scale check and density check.
        if halo_number[halo_index_temp[j]]<10:
            continue
        indices = arranged_ID[particle_index_start[halo_index_temp[j]]:particle_index_start[halo_index_temp[j]]+halo_number[halo_index_temp[j]]]
        halo_j_pos = particle_pos[remaining_particle[indices]]
        halo_j_vel = particle_vel[remaining_particle[indices]]
        selfbound_id = np.ones(halo_number[halo_index_temp[j]],dtype='bool')
        a = 0
        while True:
            halo_pos = halo_j_pos[selfbound_id==True].sum(axis=0)/halo_j_pos[selfbound_id==True].shape[0]
            halo_vel = halo_j_vel[selfbound_id==True].sum(axis=0)/halo_j_pos[selfbound_id==True].shape[0]

            distance_square_to_center = (halo_j_pos[selfbound_id==True][:,0]-halo_pos[0])**2 + (halo_j_pos[selfbound_id==True][:,1]-halo_pos[1])**2 + (halo_j_pos[selfbound_id==True][:,2]-halo_pos[2])**2
            if distance_square_to_center.shape[0]/selfbound_id.shape[0] < 0.6:
                selfbound_id = np.zeros(halo_number[halo_index_temp[j]],dtype='bool')
                a = 1
                break
                # the rank of particle after sort can be attained using np.argsort twice
            rank_of_particle = np.argsort(np.argsort(distance_square_to_center))
            distance_square_to_center = np.sort(distance_square_to_center)
            distance_to_center = np.sqrt(distance_square_to_center)
            delta_r = distance_to_center[1:] - distance_to_center[:-1]

            mass = np.linspace(1,distance_to_center.shape[0]-1,distance_to_center.shape[0]-1).astype('int')
            delta_phi = mass * delta_r/distance_square_to_center[1:]
            phi = np.zeros(distance_to_center.shape[0])
            for m in range(distance_to_center.shape[0]-1):
                phi[m+1] = phi[m] + delta_phi[m]

            phi_0 = phi[-1] + mass[-1]/distance_to_center[-1]

            phi -= phi_0
                # adjust the unit of length in phi to Mpc
            phi = G * phi * 1000 * 0.68 * M

                # the unit of velosity is km/s. escape velosity is sorted, but the relative velosity is not.
            vel_escape_square = 2 * abs(phi)

                # resort the escape velosity to their original order
            vel_escape_square = vel_escape_square[rank_of_particle]

            relative_vel_square = (halo_j_vel[selfbound_id==True][:,0]-halo_vel[0])**2 + (halo_j_vel[selfbound_id==True][:,1]-halo_vel[1])**2 + (halo_j_vel[selfbound_id==True][:,2]-halo_vel[2])**2

            if np.all(relative_vel_square<vel_escape_square):
                break

            remaining_halo_particle, = np.where(selfbound_id==True)
            selfbound_id[remaining_halo_particle[relative_vel_square>vel_escape_square]] = False

        if a==0:
            discard_particles, = np.where(selfbound_id==False)
            halo_ID[indices[discard_particles]] = 0
            confirmed_index[j] = j

@numba.jit(nopython=True)
def halo_particle_update(particle_index, halo_ID, remaining_particle, structure_index):
    for k in range(remaining_particle.shape[0]):
        particle_index[remaining_particle[k]] = structure_index[halo_ID[k]]   

def CIC3d_batch(n_grid, partical_pos,box_length):
    data_size = 512**3
    n_batch = int(partical_pos.shape[0]/data_size)
    data_grid = np.zeros([n_grid,n_grid,n_grid],dtype='float64')
    for i in range(n_batch-1):
        data = (partical_pos[i*data_size:(i+1)*data_size,:]/box_length) * (n_grid-1)
        data_int = data.astype('int32')
        data_difference = data - data_int
        calculate_grid3d_batch(data_grid, data_int, data_difference)
        del data,data_int,data_difference

    data = (partical_pos[(n_batch-1)*data_size:,:]/box_length) * (n_grid-1)
    data_int = data.astype('int32')
    data_difference = data - data_int
    calculate_grid3d_batch(data_grid, data_int, data_difference)
    del data,data_int,data_difference
# return grid data
    print('data has been assigned to the grid')
    return(data_grid)

@numba.jit(nopython=True)
def calculate_grid3d_batch(data_grid, data_int, data_difference):
    for i in range(data_int.shape[0]):
        data_grid[data_int[i,0],data_int[i,1],data_int[i,2]] +=((1-data_difference[i,0])
        * (1-data_difference[i,1]) * (1-data_difference[i,2]))
        data_grid[data_int[i,0],data_int[i,1],data_int[i,2]+1] +=((1-data_difference[i,0])
        * (1-data_difference[i,1]) * (data_difference[i,2]))
        data_grid[data_int[i,0],data_int[i,1]+1,data_int[i,2]] +=((data_difference[i,0])
        * (1-data_difference[i,1]) * (1-data_difference[i,2]))
        data_grid[data_int[i,0],data_int[i,1]+1,data_int[i,2]+1] +=((data_difference[i,0])
        * (1-data_difference[i,1]) * (data_difference[i,2]))
        data_grid[data_int[i,0]+1,data_int[i,1],data_int[i,2]] +=((1-data_difference[i,0])
        * (data_difference[i,1]) * (1-data_difference[i,2]))
        data_grid[data_int[i,0]+1,data_int[i,1],data_int[i,2]+1] +=((1-data_difference[i,0])
        * (data_difference[i,1]) * (data_difference[i,2]))
        data_grid[data_int[i,0]+1,data_int[i,1]+1,data_int[i,2]] +=((data_difference[i,0])
        * (data_difference[i,1]) * (1-data_difference[i,2]))
        data_grid[data_int[i,0]+1,data_int[i,1]+1,data_int[i,2]+1] +=((data_difference[i,0])
        * (data_difference[i,1]) * (data_difference[i,2]))

@numba.jit(nopython=True)
def rearrange_ID(halo_ID, particle_index_start, current_ID, arranged_ID):
    for j in range(halo_ID.shape[0]):
        arranged_ID[particle_index_start[halo_ID[j]]+current_ID[halo_ID[j]]] = j
        current_ID[halo_ID[j]] += 1

@numba.jit(nopython=True)
def amend_peaks(grid, Gathered_peaks, amend_peak_value, n_ref, n_min, i, w_array):
    n_grid = int(n_ref*w_array[i]+n_min)
    if i >= w_array.shape[0]-1:
        ref_grid = n_grid
    else:
        ref_grid = int(n_ref*w_array[i+1]+n_min)
        
    peak_index = 0
    for peak in Gathered_peaks:
        l1_x = abs(grid[peak[0]-1,peak[1],peak[2]]-grid[peak[0]+1,peak[1],peak[2]])
        l2_x = grid[peak[0],peak[1],peak[2]] - min(grid[peak[0]-1,peak[1],peak[2]],grid[peak[0]+1,peak[1],peak[2]])
        l1_y = abs(grid[peak[0],peak[1]-1,peak[2]]-grid[peak[0],peak[1]+1,peak[2]])
        l2_y = grid[peak[0],peak[1],peak[2]] - min(grid[peak[0],peak[1]-1,peak[2]],grid[peak[0],peak[1]+1,peak[2]])                        
        l1_z = abs(grid[peak[0],peak[1],peak[2]-1]-grid[peak[0],peak[1],peak[2]+1])
        l2_z = grid[peak[0],peak[1],peak[2]] - min(grid[peak[0],peak[1],peak[2]-1],grid[peak[0],peak[1],peak[2]+1])
        bias = l1_x**2 / (16 * l2_x -8 * l1_x) + l1_y**2 / (16 * l2_y -8 * l1_y) + l1_z**2 / (16 * l2_z -8 * l1_z)
        # bias_grid represent the bias caused by different grid length when compare maxima cross scales (CWT value in a certain grid point represent the value averaged in this grid, thus smaller grid corresponds to larger peak value).
#        bias_grid = (l2_x-l1_x/2+l2_y-l1_y/2+l2_z-l1_z/2) * (1-n_grid/ref_grid)
        bias_grid = (l2_x-l1_x/2+l2_y-l1_y/2+l2_z-l1_z/2)/3 * (1-(n_grid/ref_grid)**2) * 1.5/4
        grid[peak[0],peak[1],peak[2]] += bias
        amend_peak_value[peak_index] += (grid[peak[0],peak[1],peak[2]] + bias_grid)
        peak_index += 1

def arrange_particle(n_ref, n_min, n_grid, L_box, norm_pos, w_array, G, M, particle_pos, particle_vel, particle_index, grid, average_density, Delta_dens, halo_index, i, halo_size, calculating_peaks, calculating_peakvalue, Gathered_result):
    Gathered_result[Gathered_result>halo_size.shape[0]] = 0
#            np.save(r'/home/limx/halo_identify/TNG100_0.1-2_delta7_MPI_halogrid'+str(w_array[i-1])+'.npy',Gathered_result)
#            np.save(r'/home/limx/halo_identify/TNG100_0.1-2_delta7_MPI_halosize'+str(w_array[i-1])+'.npy',halo_size)

    index = (particle_pos[particle_index==0]/norm_pos * Gathered_result.shape[0]).astype('int32')
    halo_ID = np.zeros(particle_pos[particle_index==0].shape[0],dtype='int32')
    particle_assign(particle_pos, particle_index, Gathered_result, index, halo_ID)
    remaining_particle = np.where(particle_index==0)[0]

    halo_number = np.zeros((Gathered_result.max()+1),dtype='int32')
    particle_in_halo_calculate(halo_ID, remaining_particle, halo_number) 
    particle_index_start = np.zeros((Gathered_result.max()+1),dtype='int32')

    for j in range(Gathered_result.max()):
        particle_index_start[j+1] = particle_index_start[j] + halo_number[j]
    current_ID = np.zeros((Gathered_result.max()+1),dtype='int32')
 
# this array stores the index of particle ID that sorted by halo index
    arranged_ID = np.zeros(halo_ID.shape[0],dtype='int32')
    rearrange_ID(halo_ID, particle_index_start, current_ID, arranged_ID)
 
    halo_density = halo_number[1:]/(halo_size * (50/Gathered_result.shape[0])**3)
    dense_halo = halo_density>Delta_dens*average_density
    halo_index_temp = [0]    
    cross_scale_maximum_check(n_ref, n_min, L_box, w_array, grid, calculating_peaks, calculating_peakvalue, i, n_grid, Gathered_result, dense_halo, halo_index_temp)

    halo_index_temp = np.array(halo_index_temp)
    halo_index_temp = halo_index_temp[1:]
    print(halo_index_temp.shape,flush=True)
    confirmed_index = np.zeros(halo_index_temp.shape[0],dtype='int32')
    selfboundness_check(G, M, particle_pos, particle_vel, halo_ID, remaining_particle, halo_number, particle_index_start, arranged_ID, halo_index_temp, confirmed_index)

    halo_index_temp = halo_index_temp[confirmed_index[confirmed_index.nonzero()]]
    structure_index = np.zeros((Gathered_result.max()+1),dtype='int32')
    for k in range(halo_index_temp.shape[0]):
        structure_index[halo_index_temp[k]] += halo_index
        halo_index += 1

    confirmed_dens = halo_density[halo_index_temp-1]

    halo_particle_update(particle_index, halo_ID, remaining_particle, structure_index)

    return halo_index,confirmed_dens

def get_paramters(comm, rank):
    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_ref", type=int, help="grid resolution")
        parser.add_argument("--kw_low", type=float, help="kw_low")
        parser.add_argument("--kw_high", type=float, help="kw_high")
        parser.add_argument("--norm_pos", type=int, help="norm_pos")
        parser.add_argument("--Delta_dens", type=int, help="Delta_dens")
        parser.add_argument("--w_resolution", type=int, help="w_resolution")
        parser.add_argument("--snap_path", type=str, help="path of input data")
        parser.add_argument("--pos_path", type=str, help="path of particle position in h5 file")
        parser.add_argument("--vel_path", type=str, help="path of particle velocity in h5 file")
        parser.add_argument("--output_path", type=str, help="path of identify results")
        args = parser.parse_args()
        n_ref = args.n_ref
        kw_low = args.kw_low
        kw_high = args.kw_high
        norm_pos = args.norm_pos
        Delta_dens = args.Delta_dens
        w_resolution = args.w_resolution
        snap_path = args.snap_path
        pos_path = args.pos_path
        vel_path = args.vel_path
        output_path = args.output_path

# bcast int paraments from process 0
    if rank == 0:                                                            
        Bcasted_data = np.array([n_ref,norm_pos,Delta_dens,w_resolution],dtype ='i')
    else:                                                                      
        Bcasted_data = np.empty(4,dtype='i')                                                            
    comm.Bcast(Bcasted_data, root=0)   

    n_ref        = Bcasted_data[0]
    norm_pos     = Bcasted_data[1]
    Delta_dens   = Bcasted_data[2]
    w_resolution = Bcasted_data[3]

# bcast float paraments from process 0
    if rank == 0:                                                            
        Bcasted_data = np.array([kw_low,kw_high],dtype ='f')
    else:                                                                      
        Bcasted_data = np.empty(2,dtype='f')                                                            
    comm.Bcast(Bcasted_data, root=0)   

    kw_low        = Bcasted_data[0]
    kw_high       = Bcasted_data[1]

    if rank == 0:    
        return n_ref,kw_low,kw_high,norm_pos,Delta_dens,w_resolution,snap_path,pos_path,vel_path,output_path
    else:
        return n_ref,kw_low,kw_high,norm_pos,Delta_dens,w_resolution