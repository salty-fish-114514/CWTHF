#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def expand_structure3d(np.ndarray[np.float64_t, ndim = 3] data, np.ndarray[np.int32_t, ndim = 3] result, np.ndarray[np.npy_bool, ndim = 3] extend_index,  np.ndarray[np.int32_t, ndim = 1] x, np.ndarray[np.int32_t, ndim = 1] y, np.ndarray[np.int32_t, ndim = 1] z):

    cdef np.int64_t i, x_size
    cdef np.int32_t current_result
    cdef np.float64_t current_data
    cdef np.int16_t xi, yi, zi

    x_size = x.shape[0]

    for i in range(x_size):

        xi = x[i]
        yi = y[i]
        zi = z[i]

        current_data = data[xi,yi,zi]
        current_result = result[xi,yi,zi]

        if (result[xi+1,yi,zi] == 0) and (data[xi+1,yi,zi] < current_data):
            extend_index[xi+1,yi,zi] = True
            result[xi+1,yi,zi] = current_result
            
        if (result[xi,yi+1,zi] == 0) and (data[xi,yi+1,zi] < current_data):
            extend_index[xi,yi+1,zi] = True
            result[xi,yi+1,zi] = current_result

        if (result[xi,yi,zi+1] == 0) and (data[xi,yi,zi+1] < current_data):
            extend_index[xi,yi,zi+1] = True
            result[xi,yi,zi+1] = current_result

        if (result[xi-1,yi,zi] == 0) and (data[xi-1,yi,zi] < current_data):
            extend_index[xi-1,yi,zi] = True
            result[xi-1,yi,zi] = current_result

        if (result[xi,yi-1,zi] == 0) and (data[xi,yi-1,zi] < current_data):
            extend_index[xi,yi-1,zi] = True
            result[xi,yi-1,zi] = current_result

        if (result[xi,yi,zi-1] == 0) and (data[xi,yi,zi-1] < current_data):
            extend_index[xi,yi,zi-1] = True
            result[xi,yi,zi-1] = current_result

@cython.boundscheck(False)
@cython.wraparound(False)
def expand_structure3d1(np.ndarray[np.float64_t, ndim = 3] data, np.ndarray[np.int32_t, ndim = 3] result,  np.ndarray[np.int16_t, ndim = 1] x, np.ndarray[np.int16_t, ndim = 1] y, np.ndarray[np.int16_t, ndim = 1] z):

    cdef np.int64_t i, x_size
    cdef np.int32_t current_result
    cdef np.float64_t current_data
    cdef np.int16_t xi, yi, zi
    cdef list[int16_t] next_x = []
    cdef list[int16_t] next_y = []
    cdef list[int16_t] next_z = []

    x_size = x.shape[0]

    for i in range(x_size):

        xi = x[i]
        yi = y[i]
        zi = z[i]

        current_data = data[xi,yi,zi]
        current_result = result[xi,yi,zi]

        if (result[xi+1,yi,zi] == 0) and (data[xi+1,yi,zi] < current_data):
            result[xi+1,yi,zi] = current_result
            next_x.append(xi+1)
            next_y.append(yi)
            next_z.append(zi)  

        if (result[xi,yi+1,zi] == 0) and (data[xi,yi+1,zi] < current_data):
            result[xi,yi+1,zi] = current_result
            next_x.append(xi)
            next_y.append(yi+1)
            next_z.append(zi) 

        if (result[xi,yi,zi+1] == 0) and (data[xi,yi,zi+1] < current_data):
            result[xi,yi,zi+1] = current_result
            next_x.append(xi)
            next_y.append(yi)
            next_z.append(zi+1) 

        if (result[xi-1,yi,zi] == 0) and (data[xi-1,yi,zi] < current_data):
            result[xi-1,yi,zi] = current_result
            next_x.append(xi-1)
            next_y.append(yi)
            next_z.append(zi) 

        if (result[xi,yi-1,zi] == 0) and (data[xi,yi-1,zi] < current_data):
            result[xi,yi-1,zi] = current_result
            next_x.append(xi)
            next_y.append(yi-1)
            next_z.append(zi) 

        if (result[xi,yi,zi-1] == 0) and (data[xi,yi,zi-1] < current_data):
            result[xi,yi,zi-1] = current_result
            next_x.append(xi)
            next_y.append(yi)
            next_z.append(zi-1) 

    return np.array(next_x),np.array(next_y),np.array(next_z)

@cython.boundscheck(False)
@cython.wraparound(False)
def expand_structure3d2(np.ndarray[np.float64_t, ndim = 3] data, np.ndarray[np.int32_t, ndim = 3] result, np.ndarray[np.npy_bool, ndim = 3] extend_index, np.ndarray[np.npy_bool, ndim = 3] inside_index, np.ndarray[np.int32_t, ndim = 1] x, np.ndarray[np.int32_t, ndim = 1] y, np.ndarray[np.int32_t, ndim = 1] z):

    cdef np.int64_t i
    cdef np.int32_t current_result
    cdef np.float64_t current_data

    for i in range(x.shape[0]):

        current_data = data[x[i],y[i],z[i]]
        current_result = result[x[i],y[i],z[i]]

        if (data[x[i]+1,y[i],z[i]] < current_data):
            extend_index[x[i]+1,y[i],z[i]] = True
            result[x[i]+1,y[i],z[i]] = current_result
            
        if (data[x[i],y[i]+1,z[i]] < current_data):
            extend_index[x[i],y[i]+1,z[i]] = True
            result[x[i],y[i]+1,z[i]] = current_result

        if (data[x[i],y[i],z[i]+1] < current_data):
            extend_index[x[i],y[i],z[i]+1] = True
            result[x[i],y[i],z[i]+1] = current_result

        if (data[x[i]-1,y[i],z[i]] < current_data):
            extend_index[x[i]-1,y[i],z[i]] = True
            result[x[i]-1,y[i],z[i]] = current_result

        if (data[x[i],y[i]-1,z[i]] < current_data):
            extend_index[x[i],y[i]-1,z[i]] = True
            result[x[i],y[i]-1,z[i]] = current_result

        if (data[x[i],y[i],z[i]-1] < current_data):
            extend_index[x[i],y[i],z[i]-1] = True
            result[x[i],y[i],z[i]-1] = current_result