# CWTHF
The code of the CWTHF (Continuous Wavelet Transform Halo Finder)

Environment: Numpy, Scipy, Mpi4py, Numba, H5py.

How to use CWTHF: 

1.Compile two cython pyx files using setup.py, move the result pyd file (windows) or the so file (linux) to the CWTHF folder.

2.Run with the command: mpiexec -np 20 python /home/limx/halo_identify/CWTHF/CWTHF.py --n_ref 400 --kw_low 0.1 --kw_high 2.5 --norm_pos 50000 --Delta_dens 4 --w_resolution 20 --snap_path /home/limx/testdata/snap_m50n512_151.hdf5 --pos_path PartType2/Coordinates --vel_path PartType2/Velocities --output_path /home/limx/halo_identify 

20 is the number of processes 
/home/limx/halo_identify/CWTHF/CWTHF.py is the path of the CWTHF.py 
--n_ref 400 --kw_low 0.1 --kw_high 2.5 --norm_pos 50000 --Delta_dens 4 --w_resolution 20 are parameters mentioned in CWTHF 
/home/limx/testdata/snap_m50n512_151.hdf5 is the path of the simulation data 
PartType2/Coordinates and PartType2/Velocities are the path of the position and velosity of particles in the hdf5 file 
/home/limx/halo_identify is the path of the output identify results.

How results are organized: 

/CWTHF file stores the halo index of each particle. The size of this array is the same as the particle number, and the value in the i-th element indicates the attribution of the i-th particle. 0 means not belong to any halo. 
/CWTHF_index file stores the index of halos tunacated by their identified k_w.
