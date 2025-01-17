#cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def CWT_calculate_by_grid_cy(np.int16_t gdw_resolution, np.ndarray[np.float64_t, ndim = 3] dens_grid, 
np.ndarray[np.int16_t, ndim = 1] grid_x, np.ndarray[np.int16_t, ndim = 1] grid_y, np.ndarray[np.int16_t, ndim = 1] grid_z, 
np.ndarray[np.float64_t, ndim = 3] GDW, 
np.ndarray[np.int16_t, ndim = 1] x, np.ndarray[np.int16_t, ndim = 1] y, np.ndarray[np.int16_t, ndim = 1] z, 
np.ndarray[np.float64_t, ndim = 3] grid):

    cdef np.int64_t i, size
    cdef np.int16_t k, gridxi, gridyi, gridzi, xshape
    cdef np.float64_t weighted_gp

    size = grid_x.shape[0]
    xshape = x.shape[0]

    for i in range(size):

        gridxi = grid_x[i]
        gridyi = grid_y[i]
        gridzi = grid_z[i]

        weighted_gp = GDW[x[0]+gdw_resolution,y[0]+gdw_resolution,z[0]+gdw_resolution] * dens_grid[gridxi,gridyi,gridzi]

        for k in range(xshape):
            grid[gdw_resolution+gridxi+x[k],gdw_resolution+gridyi+y[k],gdw_resolution+gridzi+z[k]] += weighted_gp




