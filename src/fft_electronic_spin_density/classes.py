"""Contains the Density class for reading, visualizing and fourier transforming (spin) density from gaussian .cube files.
"""

# include cubetools.py in folder utils/

from fft_electronic_spin_density.utils.cubetools import read_cube, write_cube
from fft_electronic_spin_density.utils.construct_d_wave_function_from_occupations import make_wavefunction_from_coeffs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.fft as fft
# import bohr radius
from scipy.constants import physical_constants
import matplotlib as mpl
from matplotlib import pylab
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import os
from matplotlib.ticker import FuncFormatter
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
from matplotlib.ticker import MultipleLocator
from scipy.spatial.transform import Rotation
import pickle

from ase.io.xsf import read_xsf
from ase.io.cube import write_cube as write_cube_ase

from copy import deepcopy

mpl.rcParams['agg.path.chunksize'] = 10000

a0 = physical_constants['Bohr radius'][0] * 1e10 # Bohr radius in units of Angstrom

r_mt_Cu = 1.1 #Angstrom
r_mt_O = 0.9 #Angstrom

base_path = 'C:/Users/vojac/OneDrive/Dokumenty/PSI-vojac/Cu_acetate/density_projections_in_energy_windows/7.3eV/kpoints2x2x2/' #./outputs/Cu2AC4/512/'
scale_factor = 1.0 # 4.0

R1 = np.array((4.85991, 5.28091, 3.56158)) # np.array(list(self.metadata['atoms'][R_idx1][1])[1:])*physical_constants['Bohr radius'][0]*1e10
R2 = np.array((3.01571, 6.45289, 4.99992)) #np.array(list(self.metadata['atoms'][R_idx2][1])[1:])*physical_constants['Bohr radius'][0]*1e10


R_O18 = np.array((5.54428, 4.75066, 5.31609))
R_O34 = np.array((3.97804, 5.77690, 6.54433))

R_O9 = np.array((3.64655, 3.70753, 3.52220))
R_O1 = np.array((2.10020, 4.69340, 4.79607))

def windows_long_path(path):
    if os.name == 'nt':
        path = os.path.abspath(path)
        if not path.startswith('\\\\?\\'):
            path = '\\\\?\\' + path
    return path

class Density:
    """Read, visualize and fourier transform (spin) density from gaussian .cube files.
        Replace by a model function if required.
    """

    def __init__(self, fname_cube_file='./seedname.cube', permutation=None, verbose=True, scale_factor=1.0, R_atoms_idx=[0,1], output_folder='./',):
        """_summary_

        Args:
            kz (int, optional): The selected cut at k_z in 1/Angstrom. Defaults to 5.
            verbose (bool, optional): _description_. Defaults to True.
            plot_real_space_spin_density (bool, optional): _description_. Defaults to False.
        """

        self.output_folder = output_folder

        # ================== READ CUBE FILE ==================

        # cube[0] is the scalar field numpy array
        cube_data = read_cube(fname_cube_file)
        if permutation:
            cube_data = self.coordinate_permutation(cube_data, permutation=permutation)

        # cube[1] contains dictionary with metadata - keys are 'org', 'xvec', 'yvec', 'zvec', 'atoms'
        # get unit cell size: 'xvec' gives 
        #    the >>spacing<< between grid points 
        #     along the first index of the array (--> a lattice vector of the unit celll) 
        #      in units of Bohr radii
        # ... being used below

        # metadata
        self.metadata = cube_data[1]
        
        # numpy array dimensions
        self.array = cube_data[0]
        self.na, self.nb, self.nc = self.array.shape

        # 'R' vector
        #    - define the real-space separation vector between the two 'R_atoms'
                # positions of copper atoms (R = r_Cu2 - r_Cu1)
                    # Cu2_xyz = (3.01571, 6.45289, 4.99992)
                    # Cu1_xyz = (4.85991, 5.28091, 3.56158)
                    # self.R = np.array(Cu2_xyz) - np.array(Cu1_xyz)
        R_idx1, R_idx2 = R_atoms_idx
        self.R1 = R1 # np.array(list(self.metadata['atoms'][R_idx1][1])[1:])*physical_constants['Bohr radius'][0]*1e10
        self.R2 = R2 #np.array(list(self.metadata['atoms'][R_idx2][1])[1:])*physical_constants['Bohr radius'][0]*1e10
        
        # vector
        self.R_vec = self.R2 - self.R1  # = (-1.8442, 1.17198, 1.43834) A^-1 with a length of 2.616 A^-1 
        # its length
        self.R_xy = np.sqrt(self.R_vec[0]**2 + self.R_vec[1]**2)

        # point in k-space used for normalization: position of the maxima close to the center
        self.kx_for_norm = np.pi * np.abs(self.R_vec[0])/self.R_xy**2
        self.ky_for_norm = -np.pi * np.abs(self.R_vec[1])/self.R_xy**2

        self.metadata_orig = deepcopy(self.metadata)
        self.array_orig = deepcopy(self.array)


        # ================== UNITS ==================

        # unit conversion: see http://publish.illinois.edu/yubo-paul-yang/tutorials/quantum-espresso/understand-fast-fourier-transform/
        #    and   https://en.wikipedia.org/wiki/Reciprocal_lattice (formulas for reciprocal lattice vectors in 3D)

        #--- REAL SPACE ---
        # real-space grid spacing in Angstrom
        self.da = np.array(list(cube_data[1]['xvec'])) * physical_constants['Bohr radius'][0] * 1e10  # Angstrom
        self.db = np.array(list(cube_data[1]['yvec'])) * physical_constants['Bohr radius'][0] * 1e10  # Angstrom
        self.dc = np.array(list(cube_data[1]['zvec'])) * physical_constants['Bohr radius'][0] * 1e10  # Angstrom

        if verbose: print('da, db, dc', self.da, self.db, self.dc, 'Angstrom')

        # real-space lattice vectors (nx, ny, nz are the number of grid points in each direction - the dimensions of the cube numpy array)
        self.a =  self.da * self.na # Angstrom
        self.b =  self.db * self.nb # Angstrom
        self.c =  self.dc * self.nc # Angstrom

        # volume of the unit cell
        self.V = np.dot(self.a, np.cross(self.b, self.c))  # Angstrom^3

        # lattice parameter matrix (Angstrom) - should match the lattice parameters in scf.in file - first row is a, second row is b, third row is c
        self.A = np.vstack((self.a, self.b, self.c))

        # make a grid in x,y,z cartesian coordinates of the real-space grid points
        a_idx = np.arange(self.na) / self.na
        b_idx = np.arange(self.nb) / self.nb
        c_idx = np.arange(self.nc) / self.nc

        # mesh arrays are 3D arrays
        a_idx_mesh, b_idx_mesh, c_idx_mesh = np.meshgrid(a_idx, b_idx, c_idx, indexing='ij')
        
        # flatten them into (nx*ny*nz, 3) array of reciprocal coordinates
        a_idx_mesh_flat = a_idx_mesh.flatten()
        b_idx_mesh_flat = b_idx_mesh.flatten()
        c_idx_mesh_flat = c_idx_mesh.flatten()

        # check that we can reshape back to the original shape
        # a_idx_mesh_reshaped = a_idx_mesh_flat.reshape((self.na, self.nb, self.nc))
        # assert np.allclose(a_idx_mesh, a_idx_mesh_reshaped), 'Meshed arrays are not the same after flattening and reshaping back again!'

        # concatenate the flattened arrays
        r_rec_mesh_flat = np.vstack((a_idx_mesh_flat, b_idx_mesh_flat, c_idx_mesh_flat)).T

        # convert to cartesian coordinates
        r_cart_mesh_flat = r_rec_mesh_flat @ self.A

        # cartesian coordinates 3D mesh arrays - ready for use in plotting (in Angstrom)
        self.x_cart_mesh = r_cart_mesh_flat[:, 0].reshape((self.na, self.nb, self.nc))
        self.y_cart_mesh = r_cart_mesh_flat[:, 1].reshape((self.na, self.nb, self.nc))
        self.z_cart_mesh = r_cart_mesh_flat[:, 2].reshape((self.na, self.nb, self.nc))

        #--- RECIPROCAL SPACE ---

        if np.abs(scale_factor - 1) > 1e-6:
            self.pad_x = int(self.na * (scale_factor - 1))//2
            self.pad_y = int(self.nb * (scale_factor - 1))//2
            self.pad_z = int(self.nc * (scale_factor - 1))//2
        else:
            self.pad_x = 0
            self.pad_y = 0
            self.pad_z = 0

        self.nka = self.na + 2*self.pad_x
        self.nkb = self.nb + 2*self.pad_y
        self.nkc = self.nc + 2*self.pad_z

        # get reciprocal lattice spacings
        self.dka = 2 * np.pi * np.cross(self.b,self.c) / self.V * self.na/self.nka  # 1/Angstrom
        self.dkb = 2 * np.pi * np.cross(self.c,self.a) / self.V * self.nb/self.nkb # 1/Angstrom
        self.dkc = 2 * np.pi * np.cross(self.a,self.b) / self.V * self.nc/self.nkc # 1/Angstrom

        # reciprocal vectors
        self.ka = self.dka * self.nka  # 1/Angstrom
        self.kb = self.dkb * self.nkb  # 1/Angstrom
        self.kc = self.dkc * self.nkc  # 1/Angstrom

        if verbose: print('dka, dkb, dkc', self.dka, self.dkb, self.dkc, '1/Angstrom')

        # reciprocal lattice parameter matrix (1/Angstrom) - first row is ka, second row is kb, third row is kc
        self.B = np.vstack((self.ka, self.kb, self.kc))

        if verbose:
            print('A (Angstrom)\n', self.A)
            print('\nB (1/Angstrom)\n', self.B)

        # make a grid in kx,ky,kz cartesian coordinates of the reciprocal-space grid points
        ka_idx = np.arange(self.nka) / self.nka
        kb_idx = np.arange(self.nkb) / self.nkb
        kc_idx = np.arange(self.nkc) / self.nkc

        # mesh arrays are 3D arrays
        ka_idx_mesh, kb_idx_mesh, kc_idx_mesh = np.meshgrid(ka_idx, kb_idx, kc_idx, indexing='ij')

        # flatten them into (nka*nkb*nkc, 3) array of reciprocal coordinates
        ka_idx_mesh_flat = ka_idx_mesh.flatten()
        kb_idx_mesh_flat = kb_idx_mesh.flatten()
        kc_idx_mesh_flat = kc_idx_mesh.flatten()

        # concatenate the flattened arrays
        k_rec_mesh_flat = np.vstack((ka_idx_mesh_flat, kb_idx_mesh_flat, kc_idx_mesh_flat)).T

        # make a grid in kx,ky,kz cartesian coordinates of the reciprocal-space grid points
        k_cart_mesh_flat = k_rec_mesh_flat @ self.B

        # reciprocal-space cartesian coordinates 3D mesh arrays - ready for use in plotting (in 1/Angstrom)
        self.kx_cart_mesh = k_cart_mesh_flat[:, 0].reshape((self.nka, self.nkb, self.nkc))
        self.ky_cart_mesh = k_cart_mesh_flat[:, 1].reshape((self.nka, self.nkb, self.nkc))
        self.kz_cart_mesh = k_cart_mesh_flat[:, 2].reshape((self.nka, self.nkb, self.nkc))

        self.kx_cart_mesh_centered = self.kx_cart_mesh - (self.ka[0] + self.kb[0] + self.kc[0]) / 2
        self.ky_cart_mesh_centered = self.ky_cart_mesh - (self.ka[1] + self.kb[1] + self.kc[1]) / 2
        self.kz_cart_mesh_centered = self.kz_cart_mesh - (self.ka[2] + self.kb[2] + self.kc[2]) / 2

        # need to convert also units of the scalar field >>contained<< in the numpy array
        # spin density in units of electron per a_Bohr^3

    def multiply_with(self, other):
        """Multiply the density by another density object in-place.

        Args:
            other (Density): The other density object to multiply with.

        Returns:
            np.array: The product of the two densities.
        """
        self.array = np.multiply(self.array, other.array)
    
    def conjugate(self):
        """Conjugate the density in-place.

        Returns:
            np.array: The conjugated density.
        """
        self.array = np.conjugate(self.array)

    def square_data(self):
        """Square the density in-place.

        Returns:
            np.array: The squared density.
        """
        self.array = np.square(self.array)

    def get_sites_of_atoms(self, site_idx):
        """Return the site centers of the atoms at the given indices.

        Args:
            site_idx (list of integers): integers of the required sites

        Returns:
            list of tuples: tuples are cartesian coordinates (xyz) in Angstrom of the site centers
        """
        site_centers = []
        for idx in site_idx:
            # self.metadata['atoms'][idx] is a tuple with the atomic mass and the coordinates of the atom in the unit cell
            #   coordinates is a map object -> convert to list: first element is again the atomic mass -> take the rest: units are in Bohr radii -> convert to Angstrom -> convert to tuple -> add to list
            #     need to make a deepcopy of the list, otherwise would be destroyed and unable to read a second time
            coords_map = deepcopy(self.metadata['atoms'][idx])
            site_centers.append(tuple(np.array(list(coords_map[1])[1:])*physical_constants['Bohr radius'][0]*1e10))
        return site_centers

    def mask_except_sites(self, leave_sites, density_to_mask=None):
        """Mask the density except for the sites given in leave_sites.
        If density_to_mask is given, mask and return this density instead of the loaded density.

        Args:
            leave_sites (dict): Dictionary with keys 'site_centers' and 'site_radii' for the sites to leave in the model.
            density_to_mask (np.array, optional): Density to mask. Defaults to None.

        Returns:
            np.array: Masked density.
        """
        # initialize true array
        mask = np.zeros_like(self.x_cart_mesh, dtype=bool)
        for center, radius in zip(leave_sites['site_centers'], leave_sites['site_radii']):
            mask_i = np.sqrt((self.x_cart_mesh - center[0])**2 + (self.y_cart_mesh - center[1])**2 + (self.z_cart_mesh - center[2])**2) < radius
            mask = np.logical_or(mask, mask_i)
        
        self.mask = mask

        if density_to_mask is not None:
            density_to_mask[~mask] = 0
            return density_to_mask
        else:
            # just update the DFT uploaded density
            self.array[~mask] = 0

    def get_kz_at_index(self, kz_index=30):
        """Return the k_z value (in Angstrom^-1) at a given index: first or last indices are -+ k_max/2, the middle index is k_z=0 (data is zero-centered; fftshifted after fft was performed).
        Check that index is in range.

        Args:
            kz_index (int, optional): Index of the k_z value. Defaults to 30.

        Returns:
            float: k_z value in Angstrom^-1.
        """
        assert kz_index < self.nkc, f'kz_index must be between 0 and {self.nkc-1} (inclusive)'
        kz = (kz_index - self.nkc//2) * self.dkc[2]
        print(f'kz at index {kz_index} is {kz:.6f} 1/Angstrom')
        return kz
    

    def get_index_at_kz(self, kz_target=15):
        """Return the index at a given k_z value (in Angstrom^-1). Check that the k_z value is in range.

        Args:
            kz_target (int, optional): k_z value in Angstrom^-1. Defaults to 15.

        Returns:
            int: Index of the k_z value.
        """
        if kz_target > np.abs(self.kc[2]):
            raise ValueError(f'kz_target must be between 0.00 and {np.abs(self.kc[2]):.6f} 1/Angstrom')
        i_kz = np.argmin(np.abs((np.arange(self.nkc) - self.nkc//2) * self.dkc[2] - kz_target))
        print(f'index for kz_target {(i_kz - self.nkc//2) * self.dkc[2]:.6f} 1/Angstrom is {i_kz}')
        return i_kz


    def replace_by_model(self, fit=False, parameters={'type':['gaussian'], 'sigmas':[0.5], 'centers':[(0.5, 0.5, 0.5)], 'spin_down_orbital_all':[False], 'fit_params_init_all':{'amplitude':[1]},}, 
                         leave_sites=None, leave_as_wavefunction=False):
        """Replace the scalar field in the numpy array by a model function.

        Args:
            type (str, optional): Type of the model function. Defaults to 'gaussian'.
            fit (bool, optional): Fit the model to the data. Defaults to False.
            parameters (dict, optional): Parameters of the model function. Defaults to {'sigmas':[0.5], 'centers':[(0.5, 0.5, 0.5)], 'fit_params_init_all':{'amplitude':[1]}}
            leave_sites (dict, optional): Dictionary with keys 'site_centers' and 'site_radii' for the sites to leave in the model. If provided, the fitted density will be masked after its construction.
            leave_as_wavefunction (bool, optional): Leave the model as a wavefunction instead of the density (wavefunction squared). Defaults to False.
            Defaults to None.
        """

        def center_and_rotate(x, y, z, center=(3,3,3), theta0=0, phi0=0, seq='yzy'):
            x, y, z = x-center[0], y-center[1], z-center[2]
            # Rot is a matrix that rotates the orbital in that way (the inv ensures that in fact)
            #   - extrinsic rotations 'yzy' along the laboratory coordinate system axes
            Rot = R.from_euler(seq, [theta0, phi0, 0], degrees=False).inv()
            # einstein notation matrix multiplication
            x, y, z = np.einsum('ij,jklm->iklm', Rot.as_matrix(), [x, y, z])
            return x, y, z

        # define models
        models = {}
        def gaussian(x, y, z, sigma=0.5, center=(3,3,3), amplitude=1):
            """Gaussian distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                np.array: the gaussian distribution in 3D space
            """
            # normalized gaussian function - see 
            return amplitude * np.exp(-((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)/(2*sigma**2)) # * 1/(sigma**3 * (2*np.pi)**(3/2)) 
        models['gaussian'] = gaussian

        def dz2(x, y, z, sigma=0.5, center=(3,3,3), amplitude=1):
            """dz2 orbital distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            # normalized gaussian function - see
            r2 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
            return amplitude * (3*(z-center[2])**2 - r2)/r2 * np.exp(-(r2/(2*sigma**2)))
        models['dz2'] = dz2

        def dxy(x, y, z, sigma=0.5, center=(3,3,3), amplitude=1, theta0=np.pi/4, phi0=0):
            """dxy orbital distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """

            # center at site
            x, y, z = x-center[0], y-center[1], z-center[2]

            # Rot is a matrix that rotates the orbital in that way (the inv ensures that in fact)
            #   - extrinsic rotations 'zyz' along the laboratory coordinate system axes
            #   - intrinsic rotations 'ZYZ' along the orbital coordinate system axes
            Rot = R.from_euler('ZYZ', [phi0, theta0, 0], degrees=False).inv()
            
            # einstein notation matrix multiplication
            x, y, z = np.einsum('ij,jklm->iklm', Rot.as_matrix(), [x, y, z])

            # get the wave function
            r_sq = x**2 + y**2 + z**2
            return amplitude * x*y/r_sq * np.exp(-(r_sq/(2*sigma**2)))
        models['dxy'] = dxy

        def dx2y2(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-0.99290, phi0=-0.58594, Z_eff=1, C=None):
            """dxy orbital distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            n = 3  # principal number; 3 for d orbitals
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            r = np.sqrt(r_sq)
            return np.sqrt(amplitude) * (x**2 - y**2)/r_sq * (r/a0)**2 * np.exp(-(Z_eff*r/a0/n))
        models['dx2y2'] = dx2y2

        def dx2y2_normalized(x, y, z, sigma=None, center=(3,3,3), theta0=0, phi0=0, Z_eff=1):
            """dxy orbital distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.
_sq = (x**2 + y**2 + z**2)
            rho = 2 * Z_eff * np.sqrt(r_sq) / n
            C = 1/(2*np.sqrt(8*np.pi)) * Z_eff**(3/2)
            return  (C * (2 - rho) * np.exp(-rho/2) )**2
            Returns:
                _type_: _description_
            """
            n = 3  # principal number; 3 for d orbitals
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            r = np.sqrt(r_sq)
            C = 1/(9*np.sqrt(30)) * (2*Z_eff/n)**2 * Z_eff**(3/2) * np.sqrt(15/(16*np.pi))
            print('C2', C**2)
            return  C *  (x**2 - y**2)/r_sq * (r/a0)**2 * np.exp(-(Z_eff*r/a0/n))
        models['dx2y2_normalized'] = dx2y2_normalized

        def two_s(x, y, z, sigma=None, center=(3,3,3), theta0=0, phi0=0, Z_eff=1):
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2s/2s_equations.html
            n = 1  # principal number; 1 for s orbital (wrong, it should be 2 for n=2, but no point to correct now?)  
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            rho = 2 * Z_eff * np.sqrt(r_sq) / n
            C = 1/(2*np.sqrt(8*np.pi)) * Z_eff**(3/2)
            return  C * (2 - rho) * np.exp(-rho/2)
        models['two_s'] = two_s

        def two_px(x, y, z, sigma=None, center=(3,3,3), theta0=0, phi0=0, Z_eff=1):
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            n = 2
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            rho = 2 * Z_eff * np.sqrt(r_sq) / n
            C = 1/(2*np.sqrt(8*np.pi)) * Z_eff**(3/2)
            return  C * x/np.sqrt(r_sq) * rho * np.exp(-rho/2)
        models['two_px'] = two_px

        def two_spx(x, y, z, sigma=None, center=(3,3,3), amplitude=0.136746, theta0=-1.006, phi0=-0.5933, Z_eff=8.333754, Z_eff_s=None, C=0.48):
            """spx hybrid where C is the weight of the s orbital
                
            """
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            orbital = np.sqrt(1-C**2) * two_px(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff) +\
                                 C    *  two_s(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff)
            return amplitude * orbital
        models['two_spx'] = two_spx

        def two_s_correct(x, y, z, sigma=None, center=(3,3,3), theta0=0, phi0=0, Z_eff=1):
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2s/2s_equations.html
            n = 2  # principal number
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            rho = 2 * Z_eff * np.sqrt(r_sq) / a0 / n
            return  (2 - rho) * np.exp(-rho/2)
        models['two_s_correct'] = two_s_correct

        def two_px_correct(x, y, z, sigma=None, center=(3,3,3), theta0=0, phi0=0, Z_eff=1):
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            n = 2  # principal number
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            rho = 2 * Z_eff * np.sqrt(r_sq) / a0 / n
            return  x/np.sqrt(r_sq) * rho * np.exp(-rho/2)
        models['two_px_correct'] = two_px_correct

        def two_spx_correct(x, y, z, sigma=None, center=(3,3,3), amplitude=0.3, theta0=-1.006, phi0=-0.5933, Z_eff=9.05, Z_eff_s=None, C=0.48):
            """spx hybrid where C is the weight of the s orbital
                
            """
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            orbital = np.sqrt(1-C**2) * two_px_correct(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff) +\
                                 C    *  two_s_correct(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff)
            return amplitude * orbital
        models['two_spx_correct'] = two_spx_correct     

        def four_s(x, y, z, sigma=None, center=(3,3,3), theta0=0, phi0=0, Z_eff=1):
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/4s/4s_equations.html
            n = 4  # principal number; 1 for s orbital   
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            rho = 2 * Z_eff * np.sqrt(r_sq) / n
            C = 1/(96*np.sqrt(4*np.pi)) * Z_eff**(3/2)
            return  C * (24 - 36*rho + 12*rho**2 - rho**3) * np.exp(-rho/2)
        models['four_s'] = four_s

        def dx2y2_with_four_s(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-1.011299, phi0=-0.59835726, Z_eff=12.8, Z_eff_s=10, C=0.2):
            """dxy orbital distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (_type_): Cartesian x coordinate in Angstrom.
                y (_type_): Cartesian y coordinate in Angstrom.
                z (_type_): Cartesian z coordinate in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            orbital = np.sqrt(1-C**2) * dx2y2(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff) +\
                                 C    *  four_s(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff_s)
            return np.sqrt(amplitude) * orbital
        models['dx2y2_with_four_s'] = dx2y2_with_four_s

        def dx2y2_neat(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-0.99290, phi0=-0.58594, Z_eff=1, C=None):
            """dxy orbital distribution in 3D space - https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function

            Args:
                x (_type_): Cartesian x coordinate in Angstrom.
                y (_type_): Cartesian y coordinate in Angstrom.
                z (_type_): Cartesian z coordinate in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            n = 3  # principal number; 3 for d orbitals
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            r = np.sqrt(r_sq)
            rho = 2 * Z_eff * r / n / a0
            return amplitude * (x**2 - y**2)/r_sq * rho**2 * np.exp(-rho/2)
        models['dx2y2_neat'] = dx2y2_neat

        def dz2(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-0.99290, phi0=-0.58594, Z_eff=1, C=None):
            """
            dz2 orbital https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/3d/3d_equations.html
            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            n = 3  # principal number; 3 for d orbitals
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            r = np.sqrt(r_sq)
            return np.sqrt(amplitude) * (3*z**2 - r_sq)/r_sq * (r/a0)**2 * np.exp(-(Z_eff*r/a0/n))
        models['dz2'] = dz2

        def dx2y2_with_dz2(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-1.011299, phi0=-0.59835726, Z_eff=12.8, Z_eff_s=10, C=0.2):
            """dx2y2 with dz2 orbitals  https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/3d/3d_equations.html
            (not normalized)

            Args:
                x (_type_): Cartesian x coordinate in Angstrom.
                y (_type_): Cartesian y coordinate in Angstrom.
                z (_type_): Cartesian z coordinate in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            orbital = np.sqrt(1-C**2) * dx2y2(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff) +\
                                    C    *  dz2(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff_s)
            return np.sqrt(amplitude) * orbital
        models['dx2y2_with_dz2'] = dx2y2_with_dz2

        def dsphere(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-0.99290, phi0=-0.58594, Z_eff=1, C=None):
            """
            dz2 orbital https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/3d/3d_equations.html
            Args:
                x (np.array): Array of cartesian x coordinates in Angstrom.
                y (np.array): Array of cartesian y coordinates in Angstrom.
                z (np.array): Array of cartesian z coordinates in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            n = 3  # principal number; 3 for d orbitals
            x, y, z = center_and_rotate(x, y, z, center=center, theta0=theta0, phi0=phi0, seq='yzy')
            r_sq = (x**2 + y**2 + z**2)
            r = np.sqrt(r_sq)
            rho = 2 * Z_eff * r / n / a0
            return amplitude * ((3*z**2 - r**2)/(np.sqrt(3)) + 2*x*y + 2*y*z + 2*x*z + (x**2-y**2))/r_sq * rho**2 * np.exp(-rho/2)
            return amplitude * (2*x*y)/r_sq * rho**2 * np.exp(-rho/2)
            
        models['dsphere'] = dsphere

        def dx2y2_neat_with_dsphere(x, y, z, sigma=None, center=(3,3,3), amplitude=1, theta0=-1.011299, phi0=-0.59835726, Z_eff=12.8, Z_eff_s=10, C=0.2):
            """dx2y2 with dz2 orbitals  https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/3d/3d_equations.html
            (not normalized)

            Args:
                x (_type_): Cartesian x coordinate in Angstrom.
                y (_type_): Cartesian y coordinate in Angstrom.
                z (_type_): Cartesian z coordinate in Angstrom.
                sigma (float, optional): _description_. Defaults to 0.5.
                center (tuple, optional): _description_. Defaults to (3,3,3).
                sign (int, optional): _description_. Defaults to 1.

            Returns:
                _type_: _description_
            """
            # https://winter.group.shef.ac.uk/orbitron/atomic_orbitals/2p/2p_equations.html
            orbital = np.sqrt(1-C**2) * dx2y2_neat(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff) +\
                                    C    *  dsphere(x, y, z, sigma=sigma, center=center, theta0=theta0, phi0=phi0, Z_eff=Z_eff_s)
            return amplitude * orbital
        models['dx2y2_neat_with_dsphere'] = dx2y2_neat_with_dsphere
        
        def d_from_coeffs(x, y, z, sigma=None, center=(3,3,3), amplitude=None, theta0=None, phi0=None, Z_eff=None, Z_eff_s=None, C=None):
            
            coeffs = [-0.622, 0.355, -0.329, -0.241, 0.566]
            
            psi = make_wavefunction_from_coeffs(coeffs=coeffs, center=center)
            return psi(x, y, z)
        models['d_from_coeffs'] = d_from_coeffs

        # check plot
        # x = np.linspace(-1, 1, 101)
        # y = np.linspace(-1, 1, 101)
        # z = np.linspace(-1, 1, 101)
        # X, Y, Z = np.meshgrid(x, y, z)
        # f = models['dxy']
        # model_density = (f(X, Y, Z, sigma=0.5, center=(0,0,0), sign=1, amplitude=1, theta0=0, phi0=0))**2
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot = ax.scatter(X, Y, Z, c=model_density.flatten(), cmap='viridis')
        # ax.set_aspect('equal', adjustable='box')
        # plt.colorbar(plot)
        # plt.tight_layout()
        # plt.savefig('model_dxy.png')
        # exit()

        # get the optional other parameters for the model
        #    - either as the final ones if no fitting, or starting values for fitting
        if 'fit_params_init_all' in parameters:
            fit_params_init_all = parameters['fit_params_init_all']
        else:
            fit_params_init_all = {}

        def construct_model_density(fit_params_init_all):
            """Convenience function returning the model density depending on the parameters.

            Args:
                fit_params_init_all (dict): dictionary with the parameters of the model

            Returns:
                np.array: the model density (or wave function) in 3D space
            """
            model_wavefunction_spinor = [np.zeros_like(self.array), np.zeros_like(self.array)]

            # place all the site-centered models in the space
            for i in range(len(parameters['type'])):
                # create a function in 3D space that gives a Gaussian density distribution around point centers[i] with standard deviation sigmas[i]
                # the sign of the density is given by signs[i]
                sigma = parameters['sigmas'][i]
                center = parameters['centers'][i]
                
                spin_down_orbital = False if 'spin_down_orbital_all' not in parameters else parameters['spin_down_orbital_all'][i] # if not provided, flip_sign will be False
                spinor_idx = 1 if spin_down_orbital else 0
                    
                if fit_params_init_all:
                    fit_params_init = {key:value[i] for key, value in fit_params_init_all.items()}
                else:
                    fit_params_init = {}

                # choose the 3D scalar field function from models
                f = models[parameters['type'][i]]
                model_wavefunction_spinor[spinor_idx] += f(self.x_cart_mesh, self.y_cart_mesh, self.z_cart_mesh, sigma=sigma, center=center, **fit_params_init)

            if leave_as_wavefunction:
                # !!!!! returns the sum of spin-up and spin-down orbitals by default !!!!!!
                return model_wavefunction_spinor[0] + model_wavefunction_spinor[1]
            else:
                rho_up = np.multiply(model_wavefunction_spinor[0].conj(), model_wavefunction_spinor[0])
                rho_down = np.multiply(model_wavefunction_spinor[1].conj(), model_wavefunction_spinor[1])
                return rho_up - rho_down
            
        if fit:
            def dict_to_list_and_flatten(dict_in):
                list_out = []
                for value in dict_in.values():
                    for val in value:
                        list_out.append(val)
                return list_out
            
            def list_to_dict(list_in, N_for_each_key, keys):
                dict_out = {}
                for i, key in enumerate(keys):
                    dict_out[key] = list_in[N_for_each_key*i:N_for_each_key*(i+1)]
                return dict_out
            
            N_for_each_key = len(fit_params_init_all[list(fit_params_init_all.keys())[0]])
            keys = fit_params_init_all.keys()
            
            global loss_function_counter 
            loss_function_counter = 0
            self.SStot_array = np.sum((self.array[self.mask] - np.mean(self.array[self.mask]))**2)
            def loss_function(fit_params_all_as_list):
                # convert the dictionary of list values to a single list - need to feed the loss function with a single list
                # count number of iterations

                global loss_function_counter 
                loss_function_counter += 1
                fit_params_init_all = list_to_dict(fit_params_all_as_list, N_for_each_key, keys)
                model_density = construct_model_density(fit_params_init_all)

                # if leave_sites is given, mask the model density with in the same way as the original data
                if leave_sites is not None:
                    model_density = self.mask_except_sites(leave_sites, density_to_mask=model_density)
                # make loss function the 1-R2 value
                R2 = 1 - np.sum((self.array[self.mask] - model_density[self.mask])**2) / self.SStot_array
                loss_function_value = 1 - R2
                print(f'call {loss_function_counter}:   params {fit_params_all_as_list}      R^2 {R2:.6f}')
                return loss_function_value
            
            # convert the initial dictionary of list values to a single list - need to feed the loss function with a single list 
            fit_params_init_all_as_list = dict_to_list_and_flatten(fit_params_init_all)

            # fit
            res = minimize(loss_function, x0=fit_params_init_all_as_list, method='Nelder-Mead', options={'disp': True}, tol=5e-4)
            
            print(res)

            fit_params_final_all = list_to_dict(res.x, N_for_each_key, keys)
        else:
            fit_params_final_all = fit_params_init_all

        # replace the scalar field in the numpy array by the (possibly fitted) model
        self.array = construct_model_density(fit_params_final_all)


    def coordinate_permutation(self, cube_data, permutation=[2,1,0]):
        """Swap in a cyclic way (x,y,z) -> (y,z,x) -> (z,x,y) depending on number of steps (1 or 2).

        Args:
            cube_data (tuple): Tuple with the numpy array and the metadata.
            permutation (list, optional): List of integers for the permutation. Defaults to [2,1,0].

        Returns:
            tuple: Tuple with the numpy array and the metadata.
        """
        array = cube_data[0]
        xvec = cube_data[1]['xvec']
        yvec = cube_data[1]['yvec']
        zvec = cube_data[1]['zvec']

        array = np.moveaxis(array, [0, 1, 2], permutation)
        
        xyz_vec = list(np.array([xvec, yvec, zvec], dtype=object)[permutation])
    
        cube_data[1]['xvec'] = xyz_vec[0]
        cube_data[1]['yvec'] = xyz_vec[1]
        cube_data[1]['zvec'] = xyz_vec[2]

        return (array, cube_data[1])
        

    def plot_cube_file_outer_surface(self, fout_name='rho_sz.png'):
        """Plot the outer surface of the scalar field in a 3D plot.
        
        Args:
            fout_name (str, optional): Name of the output file. Defaults to 'rho_sz.png'.

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D grid
        x = np.arange(self.na)
        y = np.arange(self.nb)
        z = np.arange(self.nc)
        X, Y, Z = np.meshgrid(x, y, z)

        # Plot the scalar field
        plot = ax.scatter(X, Y, Z, c=self.array.flatten(), cmap='viridis')
        ax.set_aspect('equal', adjustable='box')
        # plot colorbar the colorbar
        plt.colorbar(plot)
        plt.tight_layout()
        plt.savefig(fout_name)
        plt.close()


    # def plot_cube_file_original(self, c_idx_arr=[0,1,-1], fout_name='rho_sz.png', alpha=0.2, figsize=(8.0, 6), dpi=300, zeros_transparent=True,
    #                    xlims=None, ylims=None, zlims=None, show_plot=False):
    #     """THE ORIGINAL
        
    #     For an array of indices, plot a 2D map as contourf at that z index of the 3D scalar field into a 3D plot at the height given by the z value.

    #     Args:
    #         c_idx_arr (list, optional): show cuts at these indeces. Defaults to [0,1,-1].
    #         fout_name (str, optional): _description_. Defaults to 'rho_sz.png'.
    #     """
    #     scale_down_data = 0.02

    #     if zeros_transparent:
    #         transparent_sigma = 0.15
    #         alpha_baseline = 0.50
    #         print(f"Plotting with transparency near the middle of the colormap: alpha = {alpha_baseline:.3f} (1 - exp(-(x/sigma)^2) with sigma={transparent_sigma:.3f})")
    #         x = np.linspace(-0.5, 0.5, pylab.cm.coolwarm.N)
    #         alpha = alpha_baseline*(1 - np.exp(-(x/transparent_sigma)**2))
    #         plt.plot(x,alpha)
    #         plt.xlabel('Transparency')
    #         plt.ylabel('rho_sz')
    #         plt.savefig('/'.join(fout_name.split('/')[:-1]) + '/transparency_profile_rho_sz_all-in-one.png', dpi=400)
    #         plt.close()
    #     else:
    #         # uniform 
    #         alpha = np.ones(pylab.cm.coolwarm.N)*alpha

    #     my_cmap = pylab.cm.coolwarm(np.arange(pylab.cm.coolwarm.N))
    #     my_cmap[:,-1] = alpha
    #     cmap = ListedColormap(my_cmap)

    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Create a 3D grid
    #     X = self.x_cart_mesh[:,:,0] # !!!(1)
    #     Y = self.y_cart_mesh[:,:,0]# !!!(2)
    #     z_cart = np.arange(self.nc)/self.nc * self.c[2] # !!!(3)

    #     z_max_abs_unscaled = np.max(np.abs(self.array))
    #     z_max_abs = z_max_abs_unscaled*scale_down_data

    #     # Plot the scalar field
    #     for c_idx in c_idx_arr:
    #         z_curr = z_cart[c_idx]
    #         Z_arr = self.array[:, :, c_idx]
            
    #         # get levels
    #         min_Z_arr = np.min(Z_arr)
    #         max_Z_arr = np.max(Z_arr)
    #         if abs(min_Z_arr - max_Z_arr) < 1e-10:
    #             levels = np.linspace(min_Z_arr-1e-10, max_Z_arr+1e-10, 100)*scale_down_data
    #         else:
    #             levels = np.linspace(min_Z_arr, max_Z_arr, 100)*scale_down_data
    #         ax.contourf(X, Y, z_curr+Z_arr*scale_down_data, cmap=cmap, zdir='z', levels=z_curr+levels, vmin=z_curr-z_max_abs, vmax=z_curr+z_max_abs)

    #     fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-z_max_abs_unscaled, z_max_abs_unscaled), cmap=cmap),
    #          ax=ax, orientation='vertical', label='spin density')
    #     # manually make a colorbar with limits -z_max_abs, z_max_abs and cmap coolwarm
        
    #     # cbar.set_clim(-z_max_abs, z_max_abs)
    #     # plt.colorbar(plot)

    #     # color
        
    #     # margin = 0.05
    #     # plt.xlim(0, np.max(self.x_cart_mesh*(1+margin)))
    #     # plt.ylim(0, np.max(self.y_cart_mesh*(1+margin)))

    #     plt.title(f'Spin density ranging from {np.min(self.array):.3f} to {np.max(self.array):.3f}')

    #     ax.set_zlim(min(0, self.c[2]), max(0, self.c[2]))

    #     ax.set_xlabel(r'$x$ ($\mathrm{\AA}$)', fontsize=11)
    #     ax.set_ylabel(r'$y$ ($\mathrm{\AA}$)', fontsize=11)
    #     ax.set_zlabel(r'$z$ ($\mathrm{\AA}$)', fontsize=11)

    #     if xlims:
    #         ax.set_xlim(xlims)
    #     if ylims:
    #         ax.set_ylim(ylims)
    #     if zlims:
    #         ax.set_zlim(zlims)

    #     ax.set_aspect('equal', adjustable='box')
    #     # plot colorbar the colorbar
    #     plt.tight_layout()

    #     if show_plot:
    #         plt.show()
    #     else:
    #         plt.savefig(fout_name, dpi=dpi)
    #         plt.close()


    def plot_cube_file_general(self, X, Y, z_levels_cart, scalar3D_data, c_idx_arr=[0,1,-1], fout_name='rho_sz.png', alpha=0.2, figsize=(8.0, 6), dpi=300, zeros_transparent=True,
                       xlims=None, ylims=None, zlims=None, show_plot=False, xlabel=None, ylabel=None, zlabel=None, colors_centered=True, cmap='coolwarm', alpha_baseline = 0.50, transparent_sigma=0.15, 
                       colorbar_label='spin density'):
        """For an array of indices, plot a 2D map as contourf at that z index of the 3D scalar field into a 3D plot at the height given by the z value.

        Args:
            c_idx_arr (list, optional): show cuts at these indeces. Defaults to [0,1,-1].
            fout_name (str, optional): _description_. Defaults to 'rho_sz.png'.

        Returns:
            None
        """
        scale_down_data = 0.02 * 1/np.max(np.abs(scalar3D_data))

        cmap = plt.get_cmap(cmap)
        if zeros_transparent:        
            if colors_centered:
                t = np.linspace(-0.5, 0.5, cmap.N)
                alpha = alpha_baseline*(1 - np.exp(-(t/transparent_sigma)**2))
                print(f"Plotting with transparency near the middle of the colormap: alpha = {alpha_baseline:.3f} (1 - exp(-(x/sigma)^2) with sigma={transparent_sigma:.3f})")
            else:
                t = np.linspace(0, 1, cmap.N)
                alpha = alpha_baseline * np.exp(-((1-t)/transparent_sigma)**3)
                print(f"Plotting with transparency everywhere except around largest values: alpha = {alpha_baseline:.3f} exp(-((1-x)/sigma)^3) with sigma={transparent_sigma:.3f})")
            
            plt.plot(t,alpha)
            plt.xlabel('Transparency')
            plt.ylabel('rho_sz')
            fsplit = fout_name.split('/')
            transparency_fout = '/'.join(fsplit[:-1]) + '/transparency_profile_' + fsplit[-1]
            plt.savefig(transparency_fout, dpi=400)
            plt.close()
        else:
            # uniform 
            alpha = np.ones(cmap.N)*alpha

        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,-1] = alpha
        cmap = ListedColormap(my_cmap)

        fig = plt.figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D grid

        z_max_abs_unscaled = np.max(np.abs(scalar3D_data))
        z_max_abs = z_max_abs_unscaled*scale_down_data

        # Plot the scalar field
        for c_idx in c_idx_arr:
            z_curr = z_levels_cart[c_idx]
            Z_arr = scalar3D_data[:, :, c_idx]
            
            # get levels
            min_Z_arr = np.min(Z_arr)
            max_Z_arr = np.max(Z_arr)
            if abs(min_Z_arr - max_Z_arr) < 1e-10:
                # if all levels in the array are close to a single value (zero typically)
                levels = np.linspace(min_Z_arr-(1e-10/scale_down_data), max_Z_arr+(1e-10/scale_down_data), 100)*scale_down_data
            else:
                levels = np.linspace(min_Z_arr, max_Z_arr, 100)*scale_down_data
            
            if np.max(levels) < 1e-10:
                levels *= 1e-10 / np.max(np.abs(levels))

            ax.contourf(X, Y, z_curr+Z_arr*scale_down_data, cmap=cmap, zdir='z', levels=z_curr+levels, vmin=z_curr-z_max_abs, vmax=z_curr+z_max_abs)

        if colors_centered:
            colorbar_min = -z_max_abs_unscaled
            colorbar_max = z_max_abs_unscaled
        else:
            colorbar_min = 0
            colorbar_max = z_max_abs_unscaled

            # add colorbar that doesn't overlap

    # cbar_kwargs={
    #     "label":"MLD in m ",
    #     "shrink": 0.8,
    #     "aspect": 30,
    #     "fraction": 0.5,
    #     "anchor": (1.0,0.5),
    #     "location": "right",
            
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(colorbar_min, colorbar_max), cmap=cmap),
             ax=ax, orientation='vertical', label=colorbar_label, shrink=0.5)
        # manually make a colorbar with limits -z_max_abs, z_max_abs and cmap coolwarm
        
        # cbar.set_clim(-z_max_abs, z_max_abs)
        # plt.colorbar(plot)

        # color
        
        # margin = 0.05
        # plt.xlim(0, np.max(self.x_cart_mesh*(1+margin)))
        # plt.ylim(0, np.max(self.y_cart_mesh*(1+margin)))

        plt.title(f'{colorbar_label} ranging from {np.min(scalar3D_data):.3f} to {np.max(scalar3D_data):.3f}')

        ax.set_zlim(min(0, self.c[2]), max(0, self.c[2]))  #!!!

        # axes labels
        if not xlabel:
            xlabel = r'$x$ ($\mathrm{\AA}$)'
        if not ylabel:
            ylabel = r'$y$ ($\mathrm{\AA}$)'
        if not zlabel:
            zlabel = r'$z$ ($\mathrm{\AA}$)'

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_zlabel(zlabel, fontsize=11)


        if xlims:
            ax.set_xlim(min(xlims), max(xlims))
        if ylims:
            ax.set_ylim(min(ylims), max(ylims))
        if zlims:
            ax.set_zlim(min(zlims), max(zlims))

        ax.set_aspect('equal', adjustable='box')
    
        plt.savefig(fout_name, dpi=dpi)
        if show_plot:
            plt.show()

    def plot_cube_rho_sz(self, c_idx_arr=[0,1,-1], fout_name='rho_sz.png', alpha=0.2, figsize=(8.0, 6), dpi=300, zeros_transparent=True,
                       xlims=None, ylims=None, zlims=None, show_plot=False, output_folder=None):
        """Concrete use of plot_cube_file_general for spin density files.

        Args:
            c_idx_arr (list, optional): show cuts at these indeces. Defaults to [0,1,-1].
            fout_name (str, optional): _description_. Defaults to 'rho_sz.png'.
            alpha (float, optional): transparency. Defaults to 0.2.
            figsize (tuple, optional): _description_. Defaults to (8.0, 6).
            dpi (int, optional): _description_. Defaults to 300.
            zeros_transparent (bool, optional): _description_. Defaults to True.
            xlims (list, optional): _description_. Defaults to None.
            ylims (list, optional): _description_. Defaults to None.
            zlims (list, optional): _description_. Defaults to None.
            show_plot (bool, optional): _description_. Defaults to False.
            output_folder (str, optional): _description_. Defaults to None.
        """

        if output_folder is None:
            output_folder = self.output_folder
        fout_name = os.path.join(output_folder, fout_name)
        
        X = self.x_cart_mesh[:,:,0]
        Y = self.y_cart_mesh[:,:,0]
        z_levels_cart = np.arange(self.nc)/self.nc * self.c[2]
        scalar3D_data = self.array

        xlabel = r'$x$ ($\mathrm{\AA}$)'
        ylabel = r'$y$ ($\mathrm{\AA}$)'
        zlabel = r'$z$ ($\mathrm{\AA}$)'

        if not zlims:
            zlims = [min(0, self.c[2]), max(0, self.c[2])]

        self.plot_cube_file_general(X, Y, z_levels_cart, scalar3D_data, c_idx_arr=c_idx_arr, fout_name=fout_name, alpha=alpha, figsize=figsize, dpi=dpi, 
                                    zeros_transparent=zeros_transparent, xlims=xlims, ylims=ylims, zlims=zlims, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
        

    def plot_cube_fft(self, c_idx_arr=[0,1,-1], fout_name='rho_sz.png', alpha=0.2, figsize=(8.0, 6), dpi=300, zeros_transparent=True,
                       xlims=None, ylims=None, zlims=None, show_plot=False, output_folder=None):
        """Concrete use of plot_cube_file_general for spin density files.

        Args:
            c_idx_arr (list, optional): show cuts at these indeces. Defaults to [0,1,-1].
            fout_name (str, optional): _description_. Defaults to 'rho_sz.png'.
            alpha (float, optional): transparency. Defaults to 0.2.
            figsize (tuple, optional): _description_. Defaults to (8.0, 6).
            dpi (int, optional): _description_. Defaults to 300.
            zeros_transparent (bool, optional): _description_. Defaults to True.
            xlims (list, optional): _description_. Defaults to None.
            ylims (list, optional): _description_. Defaults to None.
            zlims (list, optional): _description_. Defaults to None.
            show_plot (bool, optional): _description_. Defaults to False.
            output_folder (str, optional): _description_. Defaults to None.

        Returns:
            None
        """

        if output_folder is None:
            output_folder = self.output_folder
        fout_name = output_folder + '/' + fout_name
        
        # 2D grid with correct units but no dimensionality
        i_vals = (np.arange(self.nka)-self.nka//2) / self.nka
        j_vals = (np.arange(self.nkb)-self.nkb//2) / self.nkb
        I, J = np.meshgrid(i_vals, j_vals, indexing='ij')

        X = I * self.ka[0] + J * self.kb[0]
        Y = I * self.ka[1] + J * self.kb[1]

        scalar3D_data = self.F_abs_sq

        z_levels_cart = (np.arange(self.nkc)/self.nkc - 0.5) * self.kc[2]
        xlabel = r'$k_x$ ($\mathrm{\AA}^{-1}$)'
        ylabel = r'$k_y$ ($\mathrm{\AA}^{-1}$)'
        zlabel = r'$k_z$ ($\mathrm{\AA}^{-1}$)'

        if not zlims:
            zlims = (min(z_levels_cart), max(z_levels_cart))

        self.plot_cube_file_general(X, Y, z_levels_cart, scalar3D_data, c_idx_arr=c_idx_arr, fout_name=fout_name, alpha=alpha, figsize=figsize, dpi=dpi, 
                                    zeros_transparent=zeros_transparent, xlims=xlims, ylims=ylims, zlims=zlims, show_plot=show_plot, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, 
                                    colors_centered=False, cmap='viridis', 
                                    alpha_baseline=0.5,
                                    transparent_sigma=0.25, 
                                    colorbar_label=r'$|F|^2$')

    def FFT(self, verbose=True, normalized=True):
        """Perform the 3D FFT of the scalar field.

        Args:
            verbose (bool, optional): Print information about the FFT. Defaults to True.
            normalized (bool, optional): Normalize the squared FFT. Defaults to True.
        """
        # norm='backward' means no prefactor applied

        # pad array with zeros if scale_factor > 1 (because then self.nka > self.na)s
        if self.nka != self.na:
            array = np.pad(self.array, ((self.pad_x, self.pad_x), (self.pad_y, self.pad_y), (self.pad_z, self.pad_z)), 'constant', constant_values=(0, 0))
        else:
            array = self.array

        self.F = fft.fftshift(fft.fftn(array, norm='backward') )
        self.F_abs_sq = np.square(np.abs(self.F))

        # NORMALIZATION (normalize by the FFT value at the first stripes maxima (self.kx_for_norm, self.ky_for_norm) at plane defined by kz_for_norm)
        kx_center = (np.max(self.kx_cart_mesh) + np.min(self.kx_cart_mesh)) / 2
        ky_center = (np.max(self.ky_cart_mesh) + np.min(self.ky_cart_mesh)) / 2
        kz_center = (np.max(self.kz_cart_mesh) + np.min(self.kz_cart_mesh)) / 2
        
        kz_for_norm = kz_center # 1/Angstrom
        i_kz = self.get_i_kz(kz_for_norm)

        kx_data = self.kx_cart_mesh[:,:,i_kz] - kx_center
        ky_data = self.ky_cart_mesh[:,:,i_kz] - ky_center
        F_abs_sq_cut = self.F_abs_sq[:,:,i_kz]
        interp = LinearNDInterpolator(np.vstack((kx_data.flatten(), ky_data.flatten())).T, F_abs_sq_cut.flatten())
        self.F_abs_sq_normalization_constant = interp(self.kx_for_norm, self.ky_for_norm)

        # normalize squared FFT
        if normalized:
            self.F_abs_sq_max = np.max(self.F_abs_sq)
            self.F_abs_sq /= self.F_abs_sq_normalization_constant
            np.savetxt(os.path.join(self.output_folder,'normalization_constant_FFT_squared.txt'), np.array([self.F_abs_sq_normalization_constant, self.F_abs_sq_max]), delimiter='\t', header='Normalization constant for the squared FFT\tMaximum of FFT', fmt='%.8e')

    def get_i_kz(self, kz_target):
        """Get the index of the kz plane closest to the target value.

        Args:
            kz_target (float): Target value of kz.

        Returns:
            int: Index of the kz plane.
        """

                #    SIMPLE FIRST: just assume c is along z and sum along c axis
        # sum along c
        # !!!!! stupid coordinate system of MnGeO4 -- need to sum along x axis

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!1 sum along z for the next
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # find a momentum along z !!! 
        # along a for now change <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        kc_array_max = abs(self.kc[2])
        print(f'- by the way, kz_target must be between 0.00 and {kc_array_max:.6f} 1/Angstrom')
        if kz_target > kc_array_max:
            raise ValueError(f'kz_target must be between 0.00 and {kc_array_max:.6f} 1/Angstrom')
        kc_array = np.linspace(0, kc_array_max, self.nkc)
        i_kz = np.argmin(np.abs(kc_array - kz_target))
        print(f'i_kz({kz_target})', i_kz)
        return i_kz


    def create_cmap_with_cap(self, base_cmap_name="viridis", threshold=0.7, num_colors=1024):
        """Creates a colormap where colors remain constant beyond a specified threshold.

        Args:
            base_cmap_name (str, optional): Name of the base colormap. Defaults to "viridis".
            threshold (float, optional): Threshold value. Defaults to 0.7.
            num_colors (int, optional): Number of colors in the colormap. Defaults to 1024.

        Returns:
            matplotlib.colors.ListedColormap: New colormap.
        """
        # Get the base colormap
        base_cmap = plt.get_cmap(base_cmap_name)
        
        # Generate color array from the base colormap
        new_colors = base_cmap(np.linspace(0, 1, num_colors))

        # Determine threshold index
        threshold_index = int(threshold * num_colors)

        # Set all colors below the threshold to the base colormap, rescaled by the threshold
        new_colors[:threshold_index] = base_cmap(np.linspace(0, 1, threshold_index))

        # Set all colors above the threshold to the max color
        new_colors[threshold_index:] = new_colors[-1]  # Use the last color

        # Create and return new colormap
        return mcolors.ListedColormap(new_colors)


    def plot_fft_2D(self, i_kz=None, fft_as_log=False, k1_idx=0, k2_idx=1, 
                    fout_name='fft_2D_out.png', 
                    verbose=True, 
                    figsize=(6.0, 6.0), 
                    dpi=500,
                    fixed_z_scale=True, 
                    xlims=None, ylims=None,zlims=None,
                    plot_line_cut=False, 
                    kx_arr_along=None, ky_arr_along=None,
                    kx_arr_perp=None, ky_arr_perp=None,
                    kx_arr_along_opposite=None, ky_arr_along_opposite=None,
                    cut_along='both',
                    normalized=True, 
                    cax_saturation=None,
                    output_folder=None,
                    show_plot=False, 
                    data_to_plot=None,
                    quantity_name=r'$|F|^2$', 
                    X=None, 
                    Y=None,
                    xlabel=None,
                    ylabel=None,
                    title=None,
                    plot_rec_latt_vectors=True,
                    save_2D_map_data=False,
                    ):
        """Plot the 2D FFT of the scalar field.

        Args:
            i_kz (int): Index of the kz plane.
            fft_as_log (bool, optional): Plot the log of the FFT. Defaults to False.
            k1_idx (int, optional): Index of the first lattice vector. Defaults to 0.
            k2_idx (int, optional): Index of the second lattice vector. Defaults to 1.
            fout_name (str, optional): Name of the output file. Defaults to 'fft_2D_out.png'.
            verbose (bool, optional): Print information about the plot. Defaults to True.
            figsize (tuple, optional): Size of the figure. Defaults to (8.0, 6.0).
            dpi (int, optional): Dots per inch. Defaults to 500.
            fixed_z_scale (bool, optional): Fix the z scale. Defaults to True.
            xlims (list, optional): Limits of the x axis. Defaults to None.
            ylims (list, optional): Limits of the y axis. Defaults to None.
            zlims (list, optional): Limits of the z axis. Defaults to None.
            plot_line_cut (bool, optional): Plot the line cuts. Defaults to False.
            kx_arr_along (list, optional): List of kx values for the line cut along the lattice vector. Defaults to None.
            ky_arr_along (list, optional): List of ky values for the line cut along the lattice vector. Defaults to None.
            kx_arr_perp (list, optional): List of kx values for the line cut perpendicular to the lattice vector. Defaults to None.
            ky_arr_perp (list, optional): List of ky values for the line cut perpendicular to the lattice vector. Defaults to None.
            cut_along (str, optional): Direction of the line cut. Defaults to 'both'.
            normalized (bool, optional): Normalize the FFT. Defaults to True.
            cax_saturation (float, optional): Saturation value for the colorbar. Defaults to None.
            output_folder (str, optional): Output folder. Defaults to None.
            show_plot (bool, optional): Show the plot. Defaults to False.
            data_to_plot (np.ndarray, optional): Data to plot. Defaults to None, in which case the data will be a cut of np.abs(F_abs_sq) at index i_kz.
            quantity_name (str, optional): Name of the quantity to plot. Defaults to r'$|F|^2$'.
            X (2D np.ndarray, optional): X coordinates for the plot. Defaults to None, in which case they are the cartesian X coordinates of the reciprocal unit cell.
            Y (2D np.ndarray, optional): Y coordinates for the plot. Defaults to None, in which case they are the cartesian Y coordinates of the reciprocal unit cell.
            xlabel (str, optional): Label for the x axis. Defaults to None, in which case default string is provided.
            ylabel (str, optional): Label for the y axis. Defaults to None, in which case default string is provided.
            title (str, optional): Title of the plot. Defaults to None, in which case default string is provided.
            plot_rec_latt_vectors (bool, optional): Plot the reciprocal lattice vectors. Defaults to True.
        """
        
        if output_folder is None:
            output_folder = self.output_folder
        fout_name = os.path.join(output_folder, fout_name)

        # ----------------- RECIPROCAL SPACE PLOTTING -----------------
        # sum all projections into plane (defined by a vector normal to the plane)
        # n_vec_plane = np.array([0, 0, 1])

        n1 = np.array((self.nka, self.nkb, self.nkc))[k1_idx]
        n2 = np.array((self.nka, self.nkb, self.nkc))[k2_idx]

        take_idx = [0, 1, 2]
        take_idx.remove(k1_idx)
        take_idx.remove(k2_idx)
        take_idx = take_idx[0]
        print('take_idx', take_idx)

        # PREPARE 2D array 
        #    - sum?
        # F_abs_sq_sum_a = np.sum(F_abs_sq, axis=take_idx)

        #    - cut
        if i_kz is not None:
            F_abs_sq_cut = self.F_abs_sq.take(i_kz, axis=take_idx)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        k1 = [self.ka, self.kb, self.kc][k1_idx]
        k2 = [self.ka, self.kb, self.kc][k2_idx]

        if verbose:
            print(k1, k2)
            print(self.array.shape)

        # Compute the actual coordinates in 2D space - only if they have not been provided
        if X is None and Y is None:
            # 2D grid with correct units but no dimensionality
            i_vals = (np.arange(n1)-n1//2) / n1
            j_vals = (np.arange(n2)-n2//2) / n2
            I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
            X = I * k1[0] + J * k2[0]
            Y = I * k1[1] + J * k2[1]

        if data_to_plot is None:
            plot_array = np.abs(F_abs_sq_cut)
            if fft_as_log:
                plot_array = np.log(plot_array)
        else:
            plot_array = data_to_plot

        # 'SATURATED' colormap
        #     - custom cmap that will have 'viridis' from 0 to cax_lim and the max color of 'viridis' from cax_lim to 1.0
        if cax_saturation:
            cmap = self.create_cmap_with_cap(base_cmap_name="viridis", threshold=cax_saturation)
        else:
            cmap = 'viridis'

        plt.pcolormesh(X, Y, plot_array, shading='auto', cmap=cmap)

        # colorbar
        label = 'log of '+quantity_name if fft_as_log else quantity_name #r'$\mathrm{log}\(|F|^2\)_{xy}$'
        def fmt(x, pos): 
            base, exponent = f"{x:2.1e}".split('e')
            exponent = f"{int(exponent):+01d}"  # Format exponent with a sign and 3 digits
            return f"{base}e{exponent}"
        # format string which keeps fixed length of the number, scientific format
        format = None if normalized else FuncFormatter(fmt)
        cbar = plt.colorbar(label=label, format=format, shrink=0.7)

        if fixed_z_scale and not normalized:
            plt.clim(0, np.max(self.F_abs_sq))
        elif normalized:
            if zlims is None:
                plt.clim(0, 1.0)
            else:
                plt.clim(zlims[0], zlims[1])

        # Overlay grid points
        # plt.scatter(X, Y, color='black', s=1)

        # plot lattice vectors
        if plot_rec_latt_vectors:
            head_width = 0.04*np.linalg.norm(k1)
            head_length = 0.07*np.linalg.norm(k1)
            arrow_line_color = 'k'
            linestyle = (5, (5, 5))
            ax.arrow(0, 0, k1[0], k1[1], head_width=head_width, head_length=head_length, fc=arrow_line_color, ec=arrow_line_color)
            ax.arrow(0, 0, k2[0], k2[1], head_width=head_width, head_length=head_length, fc=arrow_line_color, ec=arrow_line_color)
            ax.arrow(-k1[0]/2-k2[0]/2, -k1[1]/2-k2[1]/2, k1[0], k1[1], head_width=0, head_length=0, fc=arrow_line_color, ec=arrow_line_color, linestyle=linestyle)
            ax.arrow(-k1[0]/2-k2[0]/2, -k1[1]/2-k2[1]/2, k2[0], k2[1], head_width=0, head_length=0, fc=arrow_line_color, ec=arrow_line_color, linestyle=linestyle)
            ax.arrow(k1[0]/2-k2[0]/2, k1[1]/2-k2[1]/2, k2[0], k2[1], head_width=0, head_length=0, fc=arrow_line_color, ec=arrow_line_color, linestyle=linestyle)
            ax.arrow(-k1[0]/2+k2[0]/2, -k1[1]/2+k2[1]/2, k1[0], k1[1], head_width=0, head_length=0, fc=arrow_line_color, ec=arrow_line_color, linestyle=linestyle)

        if plot_line_cut:
            # plot line cut
            linecolor_along = '#00b0f0'
            linecolor_perp = '#dc005a'
            linecolor_along_opposite = '#0058f0'
            if cut_along == 'both' or cut_along == 'along_stripes':
                if kx_arr_along is not None and ky_arr_along is not None:
                    ax.arrow(kx_arr_along[0], ky_arr_along[0], kx_arr_along[-1]-kx_arr_along[0], ky_arr_along[-1]-ky_arr_along[0], 
                                head_width=None, head_length=None, fc=linecolor_along, ec=linecolor_along, linestyle=':', linewidth=2.0, color=linecolor_along)
                    print('KX_ARR_ALONG', kx_arr_along)
                    print('KY_ARR_ALONG', ky_arr_along)
                    # dkx = kx_arr_along[len(kx_arr_along)//2]
                    # dky = ky_arr_along[len(ky_arr_along)//2]
                    # for n in range(-10, 10):
                    #     ax.arrow(kx_arr_along[0]+n*dkx, ky_arr_along[0]+n*dky, kx_arr_along[-1]-kx_arr_along[0], ky_arr_along[-1]-ky_arr_along[0], 
                    #             head_width=None, head_length=None, fc=linecolor_along, ec=linecolor_along, linestyle=':', linewidth=0.5, color=linecolor_along)
            if cut_along == 'both' or cut_along == 'perpendicular_to_stripes':
                if kx_arr_perp is not None and ky_arr_perp is not None:
                    ax.arrow(kx_arr_perp[0], ky_arr_perp[0], kx_arr_perp[-1]-kx_arr_perp[0], ky_arr_perp[-1]-ky_arr_perp[0], 
                             head_width=None, head_length=None, fc=linecolor_perp, ec=linecolor_perp, linestyle=':', linewidth=2.0, color=linecolor_perp)
                    ax.arrow(-kx_arr_along[len(kx_arr_along)//2], -ky_arr_along[len(ky_arr_along)//2], kx_arr_along[len(kx_arr_along)//2], ky_arr_along[len(ky_arr_along)//2], head_width=None, head_length=None, fc='k', ec='k', linestyle='-', linewidth=1.0, color=linecolor_perp)
                print('KX_ARR_PERP', kx_arr_perp)
                print('KY_ARR_PERP', ky_arr_perp)
            
            if cut_along == 'along_opposite_stripes':
                if kx_arr_along is not None and ky_arr_along is not None:
                    ax.arrow(kx_arr_along_opposite[0], ky_arr_along_opposite[0], kx_arr_along_opposite[-1]-kx_arr_along_opposite[0], ky_arr_along_opposite[-1]-ky_arr_along_opposite[0], 
                             head_width=None, head_length=None, fc=linecolor_along_opposite, ec=linecolor_along_opposite, linestyle=':', linewidth=2.0, color=linecolor_along_opposite)
                
        # Formatting
        if xlabel is None:
            xlabel = r"$k_x$ ($\mathrm{\AA}^{-1}$)"
        if ylabel is None:
            ylabel = r"$k_y$ ($\mathrm{\AA}^{-1}$)"
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        if title is None:
            title = quantity_name+r"($k_x$, $k_y$; $k_z$ = " + f'{self.get_kz_at_index(i_kz):.4f} '+r'$\mathrm{\AA}^{-1})$'
        plt.title(title, fontsize=12)

        ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio

        if xlims:
            plt.xlim(xlims)
        if ylims:
            plt.ylim(ylims)
        if zlims and not normalized:
            # first, add one tick with the colorbar maximum to the colorbar ticks
            # like this, it is clear what is the colorbar scale and can be easily compared with other plots
            if not zlims[1] in cbar.get_ticks():
                cbar.set_ticks(list(cbar.get_ticks()) + [zlims[1]])
            plt.clim(zlims)

        plt.tight_layout()
        if fixed_z_scale:
            # add appendix to name (while keeping original file format)
            fsplit = fout_name.split('.')
            fout_name = '.'.join(fsplit[:-1]) + '_fix-scale.' + fsplit[-1]
        plt.savefig(windows_long_path(fout_name), dpi=dpi)

        if save_2D_map_data is True:
            import pickle
            data_to_save = {}
            data_to_save['data'] = np.abs(F_abs_sq_cut)
            data_to_save['kx'] = X
            data_to_save['ky'] = Y
            with open(f'2D_map_data_kz{self.get_kz_at_index(i_kz):.4f}.pickle', 'wb') as fw:
                pickle.dump(data_to_save, fw)

        if show_plot:
            plt.show()

    def write_cube_file_rho_sz(self, fout='rho_sz_modified.cube', output_folder=None):
        """Write out the modified rho_sz to a cube file.

        Args:
            fout (str, optional): _description_. Defaults to 'rho_sz_modified.cube'.
            output_folder (str, optional): _description_. Defaults to None.
        """
        if output_folder is None:
            output_folder = self.output_folder
        fout = os.path.join(output_folder, fout)
        write_cube(self.array, self.metadata_orig, fout)

    def write_cube_file_fft(self, fout='rho_sz_fft.cube', output_folder=None):
        """Write out the modified rho_sz to a cube file.

        Args:
            fout (str, optional): _description_. Defaults to 'rho_sz_modified.cube'.
            output_folder (str, optional): _description_. Defaults to None.
        """
        if output_folder is None:
            output_folder = self.output_folder
        fout = os.path.join(output_folder, fout)
        meta_fft = deepcopy(self.metadata_orig)
        meta_fft['xvec'] = self.ka
        meta_fft['yvec'] = self.kb
        meta_fft['zvec'] = self.kc
        write_cube(self.F_abs_sq, meta_fft, fout)

    def integrate_cube_file(self, data_array=None, volume=None, verbose=True, fft=False):
        """Integrate the density in a cube file.

        Args:
            data_array (nd array, optional): cube_file data (either provided or taken as the saved array upon loading). Defaults to None.
            volume (float, optional): volume of the cell in Bohr^3 !!!. If None, taken as  Defaults to None.
            verbose (bool, optional): Print out the progress. Defaults to True.

        Returns:
            (float, float): total charge in the volume, total absolute charge in the volume
        """
        if not data_array:
            if fft:
                data_array = self.F_abs_sq
            else:
                data_array = self.array
        if not volume:
            Angstrom = 1e-10
            a_Bohr = physical_constants['Bohr radius'][0]
            # get volume in units of Bohr^3
            #   conversion factor because a, b,c are in Angstrom 
            volume = np.abs(np.dot(np.cross(self.a, self.b), self.c)) * (Angstrom/a_Bohr)**3
        rho_tot = np.sum(data_array) * volume / data_array.size
        abs_rho_tot = np.sum(np.abs(data_array)) * volume / data_array.size
        if verbose:
            print(f'\nTotal (spin) charge in the volume: {rho_tot:.6f} e')
            print(f'Total absolute (spin) charge in the volume: {abs_rho_tot:.6f} e\n')
        return rho_tot, abs_rho_tot
    
    def plot_fft_along_line(self, i_kz=None, cut_along='along_stripes', kx_ky_fun=None, k_dist_lim=15, 
                            kx_0_along=None, ky_0_along=None, kx_0_perp=None, ky_0_perp=None, 
                            N_points=3001, fout_name='test_1D_plot_along.png',
                            normalized=True, figsize=(4.5, 3.5), 
                            ylim=1.4, cax_saturation=None):
        """Plot the line cut of the FFT along the stripes.

        Args:
            i_kz (int, optional): Index of the kz plane. Defaults to None.
            cut_along (str, optional): Direction of the line cut. Defaults to 'along_stripes'.
            kx_ky_fun (function, optional): Function to calculate kx, ky for the line cut. Defaults to None.
            k_dist_lim (int, optional): Limit of the distance from the center. Defaults to 15.
            kx_0_along (float, optional): Initial kx for the line cut along the stripes. Defaults to None.
            ky_0_along (float, optional): Initial ky for the line cut along the stripes. Defaults to None.
            kx_0_perp (float, optional): Initial kx for the line cut perpendicular to the stripes. Defaults to None.
            ky_0_perp (float, optional): Initial ky for the line cut perpendicular to the stripes. Defaults to None.
            N_points (int, optional): Number of points in the line cut. Defaults to 3001.
            fout_name (str, optional): Name of the output file. Defaults to 'test_1D_plot_along.png'.
            normalized (bool, optional): Normalize the FFT. Defaults to True.
            figsize (tuple, optional): Size of the figure. Defaults to (4.5, 3.5).
            ylim (float, optional): Limit of the y axis. Defaults to 1.4.
            cax_saturation (float, optional): Saturation value for the colorbar. Defaults to None.

        Returns:
            tuple: kx_along, ky_along, F_abs_sq_interp_along, kx_perp, ky_perp, F_abs_sq_interp_per
        """
        # --- INTERPOLATION ---
        # input arrays
        if i_kz is None:
            i_kz = self.get_i_kz(0)

        kx_center = (np.max(self.kx_cart_mesh[:,:,i_kz]) + np.min(self.kx_cart_mesh[:,:,i_kz])) / 2
        ky_center = (np.max(self.ky_cart_mesh[:,:,i_kz]) + np.min(self.ky_cart_mesh[:,:,i_kz])) / 2

        print('kx_center before', np.max(self.kx_cart_mesh[:,:,i_kz]) + np.min(self.kx_cart_mesh[:,:,i_kz]) / 2)
        print('ky_center before', np.max(self.ky_cart_mesh[:,:,i_kz]) + np.min(self.ky_cart_mesh[:,:,i_kz]) / 2)

        if ky_0_along is None:
            kx_0_along = self.kx_for_norm
            ky_0_along = self.ky_for_norm
        if kx_0_perp is None:
            kx_0_perp = 0
        if ky_0_perp is None:
            ky_0_perp = 0

        self.kx_0_along = kx_0_along
        self.ky_0_along = ky_0_along

        if kx_ky_fun is None:
            def kx_ky_fun(k_dist):
                # direction cosines
                gamma_x = self.R_vec[0] / self.R_xy
                gamma_y = self.R_vec[1] / self.R_xy
                # 90 deg rotated --> perpendicular the stripes
                kx_along = kx_center + kx_0_along - gamma_y * k_dist
                ky_along = ky_center + ky_0_along + gamma_x * k_dist
                # at opposite stripe
                kx_along_opposite = kx_center - kx_0_along - gamma_y * k_dist
                ky_along_opposite = ky_center - ky_0_along + gamma_x * k_dist
                # R is perpendicular to stripes
                kx_perp = kx_center + kx_0_perp + gamma_x * k_dist
                ky_perp = ky_center + ky_0_perp + gamma_y * k_dist
                return kx_along, ky_along, kx_perp, ky_perp, kx_along_opposite, ky_along_opposite

        k_dist_arr = np.linspace(-k_dist_lim, k_dist_lim, N_points)
        kx_along_arr, ky_along_arr, kx_perp_arr, ky_perp_arr, kx_along_opposite_arr, ky_along_opposite_arr = kx_ky_fun(k_dist_arr)

        # interpolate along the line
        kx_data = self.kx_cart_mesh[:,:,i_kz]
        ky_data = self.ky_cart_mesh[:,:,i_kz]
        if not 'F_abs_sq' in dir(self):
            self.FFT()
        F_abs_sq_cut = self.F_abs_sq[:,:,i_kz]

        interp = LinearNDInterpolator(np.vstack((kx_data.flatten(), ky_data.flatten())).T, F_abs_sq_cut.flatten())
        F_abs_sq_interp_along = interp(kx_along_arr, ky_along_arr)
        F_abs_sq_interp_perp = interp(kx_perp_arr, ky_perp_arr)
        F_abs_sq_interp_along_opposite = interp(kx_along_opposite_arr, ky_along_opposite_arr)

        print('MAXIMUM OF FFT', np.max(self.F_abs_sq))
        print('NORMALIZATION FACTOR FFT', self.F_abs_sq_normalization_constant )
            
        # --- PLOTTING ---
        linecolor_along = '#00b0f0'
        linecolor_perp = '#dc005a'
        linecolor_along_opposite = '#0058f0'
        if fout_name is not None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            if cut_along == 'along_stripes' or cut_along == 'both':
                ax.plot(k_dist_arr, F_abs_sq_interp_along, '-', color=linecolor_along)
            if cut_along == 'perpendicular_to_stripes' or cut_along == 'both':
                ax.plot(k_dist_arr, F_abs_sq_interp_perp, '-', color=linecolor_perp)
            if cut_along == 'along_opposite_stripes':
                ax.plot(k_dist_arr, F_abs_sq_interp_along_opposite, '-', color=linecolor_along_opposite)
            ax.set_xlabel(r'$k$ ($\mathrm{\AA}^{-1}$)', fontsize=12)
            ax.set_ylabel(r'$|F|^2$', fontsize=12)
            title_appendix = ' ' + ' '.join(cut_along.split('_'))
            ax.set_title(r'$|F|^2$'+title_appendix)
            # xticks by 1.0 Angstrom^-1
            ax.xaxis.set_minor_locator(MultipleLocator(1.0))
            if normalized:
                ax.set_ylim(0, ylim)
            if cax_saturation:
                plt.axhline(y=cax_saturation, color='k', linestyle='--', linewidth=1.0)
            plt.tight_layout()
            plt.savefig(fout_name, dpi=400)
            plt.savefig('.'.join(fout_name.split('.')[:-1])+'.pdf', dpi=400)

        return (kx_along_arr-kx_center), (ky_along_arr-ky_center), F_abs_sq_interp_along, (kx_perp_arr-kx_center), (ky_perp_arr-ky_center), F_abs_sq_interp_perp, (kx_along_opposite_arr-kx_center), (ky_along_opposite_arr-ky_center), \
                F_abs_sq_interp_along_opposite

    def get_c_idx_at_z_coordinates(self, z_coordinates):
        """Get the indices of the z coordinates.

        Args:
            z_coordinates (list): List of z coordinates.

        Returns:
            list: List of indices.
        """
        # check if z_coordinates are in the range of (0, self.c)
        if not np.all(0 <= z <= self.c for z in z_coordinates):
            raise ValueError(f'z_coordinates must lie within the unit cell range (0, {self.c}) Angstrom.')
        
        # an array of all possible z levels
        z_levels_cart = np.arange(0, self.nc) * self.dc[2]
        idx_z = []
        for z in z_coordinates:
            idx_z.append(np.argmin(np.abs(z_levels_cart - z)))
        return idx_z
    
    def get_line_density(self, along_vector, center, N_points=101):
        """Get the line density along a vector.

        Args:
            along_vector (np.ndarray): The vector along which to get the density.
            center (np.ndarray): The center point of the line.
            N_points (int, optional): Number of points along the line. Defaults to 101.

        Returns:
            tuple: The 1D coordinates (Angstrom) and the line density (Angstrom^-1).
        """
        r_along_R_vec = np.array([self.x_cart_mesh.flatten() - center[0], self.y_cart_mesh.flatten() - center[1], self.z_cart_mesh.flatten() - center[2]]).T @ along_vector
        density_3D = self.array.flatten()
        density_1D = np.zeros(N_points)
        step = (r_along_R_vec.max() - r_along_R_vec.min()) / (N_points-1)  # Angstrom
        r_along_R_vec_min = r_along_R_vec.min()
        for i in range(N_points):
            in_interval = np.logical_and(r_along_R_vec_min + (i + 1) * step > r_along_R_vec, r_along_R_vec >= r_along_R_vec_min + i * step)
            density_1D[i] = np.sum(density_3D[in_interval])
        density_1D *= self.V/a0**3 / self.array.size / step # units of 1/Angstrom; (self.V is the unit cell volume and is converted from Angstrom^3 to Bohr^3 because 3D density is in Bohr^-3)
        r_along_R_vec = np.linspace(r_along_R_vec.min(), r_along_R_vec.max(), N_points)
        # ensure that step in r_along_R_vec is the same as step
        print('step (Angstrom)', step)
        print('r_along_R_vec step -- must be equal', r_along_R_vec[1] - r_along_R_vec[0])
        # test that integration is correct
        print('density_1D_sum * step', np.sum(density_1D) * step)
        # absolute
        print('abs(density_1D_sum * step)', np.sum(np.abs(density_1D) * step))
        return r_along_R_vec, density_1D

    
def get_XY_meshgrid_in_space(u=(1,0,0), v=(0,1,0), origin=(0,0,0), x_lim=1, y_lim=1, N_points=101):
    """
    Generate a meshgrid of points in the uv plane shifted to origin.

    Parameters:
    axis (tuple): The axis around which to rotate the XY plane (default is z-axis).
    origin (tuple): The origin point of the meshgrid (default is (0,0,0)).
    x_lim (float): The half-width of the meshgrid in the x direction.
    y_lim (float): The half-width of the meshgrid in the y direction.
    N_points (int): The number of points along each axis.

    Returns:
    tuple: A tuple containing the X,Y,Z 2D arrays of coordinates and the lattice vectors of the meshgrid.
    """

    print('orthogonality before norm check u and v', np.dot(u, v))

    print('u', u)
    print('v', v)

    u_norm = np.array(u) / np.linalg.norm(u)  # Normalize the u vector
    v_norm = np.array(v) / np.linalg.norm(v)  # Normalize the axis vector

    print('u_norm', u_norm)
    print('v_norm', v_norm)
    print('orthonormality check u and v', np.dot(u_norm, v_norm))

    # the direct meshgrid coordinates 
    i_u = np.linspace(-x_lim, x_lim, N_points)
    i_v = np.linspace(-y_lim, y_lim, N_points)
    X_direct, Y_direct = np.meshgrid(i_u, i_v, indexing='ij')

    # the cartesian meshgrid coordinates
    X_cart = X_direct * u_norm[0] + Y_direct * v_norm[0] + origin[0]
    Y_cart = X_direct * u_norm[1] + Y_direct * v_norm[1] + origin[1]
    Z_cart = X_direct * u_norm[2] + Y_direct * v_norm[2] + origin[2]

    return X_cart, Y_cart, Z_cart, X_direct, Y_direct, u_norm, v_norm


def test_shift():
    """Test the shift of a 3D array using numpy's fftshift function.
    """
    array_3D = np.zeros((9,9,9), dtype=np.float_)
    idx_3D = np.zeros((9,9,9), dtype=np.bool_)

    for i in range(9):
        for j in range(9):
            for k in range(9):
                idx_3D[i,j,k] = i > 4 and j > 4 and k > 4
    # print(array3D)

    array_3D[idx_3D] = 1
    array_3D[~idx_3D] = -1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create a 3D grid
    x = np.arange(9)
    y = np.arange(9)
    z = np.arange(9)
    X, Y, Z = np.meshgrid(x, y, z)
    # Plot the scalar field
    plot = ax.scatter(X, Y, Z, c=idx_3D.flatten(), cmap='coolwarm')
    ax.set_aspect('equal', adjustable='box')
    # plot colorbar the colorbar
    plt.colorbar(plot)
    plt.tight_layout()
    plt.savefig('fake_3D.png')
    plt.close()

    # shift
    array3D_shifted = np.fft.fftshift(idx_3D)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create a 3D grid
    x = np.arange(9)
    y = np.arange(9)
    z = np.arange(9)
    X, Y, Z = np.meshgrid(x, y, z)
    # Plot the scalar field
    plot = ax.scatter(X, Y, Z, c=array3D_shifted.flatten(), cmap='coolwarm')
    ax.set_aspect('equal', adjustable='box')
    # plot colorbar the colorbar
    plt.colorbar(plot)
    plt.tight_layout()
    plt.savefig('fake_3D_shifted.png')
    plt.close()

def get_interpolator(interpolator_name: str, X: np.ndarray, fX: np.ndarray, verbose=True) -> LinearNDInterpolator:
    """Create or load an interpolator for the given data.

    Args:
        interpolator_name (str): Name of the interpolator file.
        X (np.ndarray): Input data points.
        fX (np.ndarray): Function values at the input data points.
        verbose (bool, optional): Verbosity flag. Defaults to True.

    Returns:
        LinearNDInterpolator: The interpolator object.
    """
    if not os.path.exists(interpolator_name):
        if verbose: print('creating interpolator...')
        interpolator = LinearNDInterpolator(X, fX)
        with open(interpolator_name, 'wb') as f:
            pickle.dump(interpolator, f)
        if verbose: print('interpolator saved to a pickle file!')
    else:
        if verbose: print('interpolator pickle file exists! loading interpolator...')
        with open(interpolator_name, 'rb') as f:
            interpolator = pickle.load(f)
    return interpolator


def workflow(output_folder, site_idx, site_radii, replace_DFT_by_model, parameters_model, fit_model_to_DFT,
             use_saved_interpolated_data=True, save_interpolated_data=True):
    """Main workflow for the analysis of the 3D scalar field.

    Args:
        output_folder (str): Output folder.
        site_idx (list): List of site indices.
        site_radii (list): List of site radii.
        replace_DFT_by_model (bool): Replace DFT by a model.
        parameters_model (dict): Parameters of the model.
        fit_model_to_DFT (bool): Fit the model to the DFT.
    """
    # --- INPUT ----

    fname_cube_file = f'{base_path}Cu2AC4_rho_sz_0.0eV_to_1.5eV.cube' #'./cube_files/Mn2GeO4_rho_sz.cube'
    
    permutation = None #!! for Mn2GeO4 need to use [2,1,0] to swap x,y,z -> z,y,x

    # ---- CALCULATION CONTROL ----

    density_3D = False
    density_slices = False
    
    fft_3D = True
    full_range_fft_spectrum_cuts = False
    zoom_in_fft_spectrum_cuts = True
    fft_cut_planes_oblique = False

    write_cube_files = True

    # ---- PARAMETERS -----

    cax_saturation = 0.5

    # !!!! [0, 1.6e6] for a single site and [0, 6.4e6] for double !
    if not site_idx:
        fft_zlims = [0, 4*6.4e6]
    elif 0 in site_idx and 1 in site_idx:
        fft_zlims = [0, 6.4e6] 
    else:
        fft_zlims = [0, 1.6e6] # arb. units

    density_figsize = (6.0, 4.5)
    dpi_rho = 500
    density_slice_each_n_images = 4

    all_in_one_xlims = (1.5, 6.5) #None
    all_in_one_ylims = (3.5, 8.5) #None
    all_in_one_zlims = (1.0, 7.5) #None

    # all_in_one_xlims = None
    # all_in_one_ylims = None
    # all_in_one_zlims = None

    fft_figsize = (4.5, 4.5)
    fft_dpi = 400
    fft_xlims = [-19, 19] # 1/Angstrom
    fft_ylims = [-19, 19] # 1/Angstrom
    
    fft_as_log = False
    fft_slice_each_n_images = 4
    all_in_one_density_total_slices = 300

    # -- create folder if does not exist --
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- READ CUBE FILE -----
    density = Density(permutation=permutation, verbose=True, fname_cube_file=fname_cube_file, scale_factor=scale_factor, output_folder=output_folder)

    density.integrate_cube_file()

    # ---- MASKING -----

    # leave_sites = {'site_centers':[(0.5, 0.5, 0.5)], 'site_radii':[0.5]}
    if site_idx and site_radii:
        site_centers = density.get_sites_of_atoms(site_idx)
        print('site_centers', site_centers)
        leave_sites = {'site_centers':site_centers, 'site_radii':site_radii}
        density.mask_except_sites(leave_sites)

    print('After masking:')
    density.integrate_cube_file()

    # get kz at index
    # density.get_kz_at_index(80)
    # density.get_index_at_kz(-14.67785)
    # density.get_index_at_kz(14.67785)

    # ---- INSERT (/AND FIT) MODEL -----
    # parameters_model = {'type':'gaussian', 'sigmas':[0.2, 0.2], 'centers':[(-3.0, -3, -5), (-2.0, -3, -5)], 'amplitudes':[1, -1]}
    # parameters_model = {'type':'gaussian', 'sigmas':[0.3, 0.3], 'centers':site_centers, 'amplitudes':[1,-1]}

    # add centers automatically according to the sites

    if site_centers and site_radii:
        parameters_model['centers'] = site_centers

    if replace_DFT_by_model:
        density.replace_by_model(fit=fit_model_to_DFT, parameters=parameters_model, leave_sites=leave_sites)
    print('After replacing by model:')
    density.integrate_cube_file()


    # ---- VISUALIZE DENSITY -----
    if density_slices:
        for i in np.arange(0, density.nc, density_slice_each_n_images):
            c_idx_array = np.array([i, 0]) #np.array([i, -1]
            density.plot_cube_rho_sz(c_idx_arr=c_idx_array, fout_name=f'rho_sz_exploded_masked_{i}.jpg', alpha=0.8, figsize=density_figsize, dpi=dpi_rho, zeros_transparent=False)  # rho_sz_gauss_exploded

    if density_3D:
        c_idx_array = np.arange(0, density.nc, max(1, density.nc//all_in_one_density_total_slices))
        density.plot_cube_rho_sz(c_idx_arr=c_idx_array, fout_name=f'rho_sz_exploded_masked_all.jpg', alpha=0.05, figsize=(5.5,5.5), dpi=dpi_rho, zeros_transparent=True,
                            xlims=all_in_one_xlims, ylims=all_in_one_ylims, zlims=all_in_one_zlims, show_plot=False)  # rho_sz_gauss_exploded_all

    # single cut
        # kz = 30
    # i_kz = density.get_i_kz(kz_target=kz)
    # density.plot_2D_fft(i_kz=i_kz, k1_idx=k1_idx, k2_idx=k2_idx, fout_name=f'./test_fft.png')

    # ---- WRITE MODIFIED DENSITY TO CUBE FILE -----
    if write_cube_files:
        density.write_cube_file_rho_sz(fout=f'rho_sz_modified.cube')

    # ---- FFT -----
    density.FFT(verbose=True)

    # ---- WRITE MODIFIED FFT TO CUBE FILE -----
    if write_cube_files:
        density.write_cube_file_fft(fout=f'fft.cube')

    # ---- VISUALIZE FFT -----
    
    # 3D
    if fft_3D:
        c_idx_array = np.arange(0, density.nc, max(1, density.nc//all_in_one_density_total_slices))
        xlims = [-9, 9] #None
        ylims = xlims
        zlims = xlims
        density.plot_cube_fft(c_idx_arr=c_idx_array, fout_name=f'F_abs_sq_all.png', figsize=(5.5,5.5), dpi=dpi_rho, zeros_transparent=True,
                                xlims=xlims, ylims=ylims, zlims=zlims, show_plot=False)

    # (1) variable scale, full reciprocal space
    if full_range_fft_spectrum_cuts:
        for i_kz in range(0, density.nc, fft_slice_each_n_images):
            appendix = '_log' if fft_as_log else ''
            density.plot_fft_2D(i_kz=i_kz, fft_as_log=fft_as_log, 
                                fout_name=f'F_abs_sq{appendix}-scale_kz_at_idx_{i_kz}.png', 
                                figsize=fft_figsize,
                                dpi=fft_dpi, 
                                fixed_z_scale=False,
                                cax_saturation=cax_saturation,
                                xlims=None,
                                ylims=None)
        
    # (2) fixed scale, zoom-in
    if zoom_in_fft_spectrum_cuts:
        kz_arr = range(density.nkc//2, density.nkc//2 + 1)
        for i_kz in kz_arr:
            appendix = '_zoom_log' if fft_as_log else '_zoom'
            density.plot_fft_2D(i_kz=i_kz, fft_as_log=fft_as_log, 
                                fout_name=f'F_abs_sq{appendix}-scale_kz_at_idx_{i_kz}.png', 
                                figsize=(5.5, 4.5),
                                dpi=fft_dpi,
                                fixed_z_scale=True,
                                cax_saturation=cax_saturation,
                                xlims=fft_xlims,
                                ylims=fft_ylims, 
                                zlims=fft_zlims)
            
            # along stripes
            for cut_along in ['along_stripes', 'perpendicular_to_stripes', 'both', 'along_opposite_stripes']:
                kx_arr_along, ky_arr_along, F_abs_sq_interp_along, kx_arr_perp, ky_arr_perp, F_abs_sq_interp_perp, kx_arr_along_opposite, ky_arr_along_opposite, F_abs_sq_interp_along_opposite = density.plot_fft_along_line(i_kz=i_kz, cut_along=cut_along, kx_ky_fun=None, k_dist_lim=12, N_points=3001, fout_name=f'{output_folder}/cut_1D_{cut_along}.png', cax_saturation=cax_saturation,)
                density.plot_fft_2D(i_kz=i_kz, fft_as_log=fft_as_log, 
                            fout_name=f'F_abs_sq{appendix}-scale_kz_at_idx_{i_kz}_cut_{cut_along}.png', 
                            figsize=(5.5, 4.5),
                            dpi=fft_dpi,
                            fixed_z_scale=True,
                            cax_saturation=cax_saturation,
                            xlims=fft_xlims,
                            ylims=fft_ylims, 
                            zlims=fft_zlims,
                            plot_line_cut=True, 
                            kx_arr_along=kx_arr_along, ky_arr_along=ky_arr_along,
                            kx_arr_perp=kx_arr_perp, ky_arr_perp=ky_arr_perp,
                            kx_arr_along_opposite=kx_arr_along_opposite, ky_arr_along_opposite=ky_arr_along_opposite,
                            cut_along=cut_along)
                np.savetxt(os.path.join(density.output_folder, 'cut_1D_both.txt'), np.array([kx_arr_along, ky_arr_along, F_abs_sq_interp_along, kx_arr_perp, ky_arr_perp, F_abs_sq_interp_perp, kx_arr_along_opposite, ky_arr_along_opposite, F_abs_sq_interp_along_opposite]).T, delimiter='\t', fmt='%.8e', header='kx_along\tky_along\tF_abs_sq_along\tkx_perp\tky_perp\tF_abs_sq_perp\tkx_along_opposite\tky_along_opposite\tF_abs_sq_along_opposite')
        
        # for the middle i_kz, plot also line cuts
        i_kz = density.nkc//2
        density.plot_fft_2D(i_kz=i_kz, fft_as_log=fft_as_log, 
                    fout_name=f'F_abs_sq{appendix}-scale_kz_at_idx_{i_kz}.png', 
                    figsize=(5.5, 4.5),
                    dpi=fft_dpi,
                    fixed_z_scale=True,
                    cax_saturation=cax_saturation,
                    xlims=fft_xlims,
                    ylims=fft_ylims, 
                    zlims=fft_zlims,
                    save_2D_map_data=True)
        
    def fft_perform_cut_planes_oblique(u_axis, v_axis, center=(0.0, 0.0, 0.0), seedname_perp_cut='perp_to_axis'):

        # create the interpolator
        #   kx_cart_mesh etc. are 3D arrays of kx coordinates etc.

        xy_lim = 6.0 # Angstrom^-1

        kx_flat = density.kx_cart_mesh_centered.flatten()
        ky_flat = density.ky_cart_mesh_centered.flatten()
        kz_flat = density.kz_cart_mesh_centered.flatten()

        print('np.min(density.kx_cart_mesh)', np.min(density.kx_cart_mesh_centered))
        print('np.max(density.kx_cart_mesh)', np.max(density.kx_cart_mesh_centered))


        # axis = density.R_vec #(0,0,1) #
        # center = -density.R_vec/np.linalg.norm(density.R_vec)*1.201 #(0.0, 0.0, 0.0) #

        interpolator_name = os.path.join(output_folder, f'interpolator_512_scale-factor_{scale_factor:.1f}_xy-lim_{xy_lim:.2f}_invA_center_{center[0]:.2f}_{center[1]:.2f}_{center[2]:.2f}_invA.pickle')

        interpolator_cutoff = xy_lim*np.sqrt(3) # Angstrom^-1
        include_points = np.where(np.sqrt((kx_flat-center[0])**2 + (ky_flat-center[1])**2 + (kz_flat-center[2])**2) < interpolator_cutoff)
        print('kx_cart_mesh shape', density.kx_cart_mesh_centered.shape)
        print('kx_flat[include_points] shape', kx_flat[include_points].shape)
        
        # ------- GET (or load) THE INTERPOLATOR ------
            # coordinates
        X = np.vstack((kx_flat[include_points], ky_flat[include_points], kz_flat[include_points])).T
            # function values
        fX = density.F_abs_sq.flatten()[include_points]
            # create (or load / save) the interpolator
        interpolator = get_interpolator(interpolator_name, X, fX, verbose=True)
         
        N_points = 101
        cax_saturation = None
        normalized = True
        zlims = (0.0, 1.0)
        fft_as_log = False

        title = f'FFT centered at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})'+r'$\mathrm{\AA}^{-1}$'
        cax_saturation_str = f'{cax_saturation:.1f}' if cax_saturation else 'None'
        fout_name = f'fft_2D_scale-factor_{scale_factor:.1f}_{seedname_perp_cut}_{v_axis[0]:.2f}_{v_axis[1]:.2f}_{v_axis[2]:.2f}_center_{center[0]:.2f}_{center[1]:.2f}_{center[2]:.2f}_Nk_{N_points}_caxsat_{cax_saturation_str}_normalized_{normalized}_log-{fft_as_log}.png'

        # X_cart, Y_cart, Z_cart, X_direct, Y_direct are all 2D arrays of coordinates
        X_cart, Y_cart, Z_cart, X_direct, Y_direct, u, v = get_XY_meshgrid_in_space(u=u_axis, v=v_axis, origin=center, x_lim=xy_lim, y_lim=xy_lim, N_points=N_points)

        # check if the interpolation has been done already
        interp_data_fout_pickle_name = f'fft_2D_DATA_scale-factor_{scale_factor:.1f}_{seedname_perp_cut}_{v_axis[0]:.2f}_{v_axis[1]:.2f}_{v_axis[2]:.2f}_center_{center[0]:.2f}_{center[1]:.2f}_{center[2]:.2f}_Nk_{N_points}.pickle'
        interp_data_fout_pickle_name = os.path.join(density.output_folder, interp_data_fout_pickle_name)

        if use_saved_interpolated_data and os.path.exists(interp_data_fout_pickle_name):
            print('Interpolated data already exists! Loading from pickle file...')
            with open(interp_data_fout_pickle_name, 'rb') as f:
                data_to_load = pickle.load(f)
                X_direct = data_to_load['X_direct']
                Y_direct = data_to_load['Y_direct']
                fft_XYZ = data_to_load['fft_XYZ']
        else:
            # need to perform the interpolation (takes time)
            #   get 2D array of the FFT values:
            print('interpolating in 2D plane')
            fft_XYZ = interpolator(X_cart, Y_cart, Z_cart)

            if save_interpolated_data:
                data_to_save = {}
                data_to_save['X_direct'] = X_direct
                data_to_save['Y_direct'] = Y_direct
                data_to_save['fft_XYZ'] = fft_XYZ
                with open(interp_data_fout_pickle_name, 'wb') as f:
                    pickle.dump(data_to_save, f)


        xlabel = r'$k_u$' + f" [({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})" + r'$\mathrm{\AA}^{-1}$]' + f"+ ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})" + r'$\mathrm{\AA}^{-1}$'
        ylabel = r'$k_v$' + f" [({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})" + r'$\mathrm{\AA}^{-1}$]' + f"+ ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})" + r'$\mathrm{\AA}^{-1}$'

        density.plot_fft_2D(
                            fout_name=fout_name,
                            data_to_plot=fft_XYZ, 
                            X=X_direct, 
                            Y=Y_direct,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            title=title,
                            cax_saturation=cax_saturation,
                            xlims=None, #(-xy_lim, xy_lim),
                            ylims=None, #(-xy_lim, xy_lim),
                            zlims=zlims,
                            normalized=normalized,
                            plot_rec_latt_vectors=False,
                            fft_as_log=fft_as_log,
                            )


    if fft_cut_planes_oblique:
        r_Ox9 = np.array([3.64655, 3.70753, 3.52220])
        r_Ox2 = np.array([5.77542, 7.04040, 3.76543])
        r_Ox1 = np.array([2.10020, 4.69340, 4.79607])
        r_Ox10 = np.array([4.22907, 8.02627, 5.03930])

        r_Ox33 = np.array([3.89759, 5.95690, 2.01717])
        r_Ox18 = np.array([5.54428, 4.75066, 5.31609])
        r_Ox34 = np.array([3.97804, 5.77690, 6.54433])
        r_Ox17 = np.array([2.33134, 6.98314, 3.24541])
        
        r_Cu1 = np.array([4.85991, 5.28091, 3.56158])
        r_Cu2 = np.array([3.01571, 6.45289, 4.99992])
        
        # right-handed system m-n-l, where m is along Ox-Ox (y-axis), n is along Ox-Ox (x-axis) and l is along Cu-Cu (z-axis)  
        R_m = ( (r_Ox18 - r_Ox33) + (r_Ox34 - r_Ox17) ) / 2 # average the equivalent Ox-Ox connections on the two copper sites
        R_n = ( (r_Ox9 - r_Ox2) + (r_Ox1 - r_Ox10) ) / 2   # average the equivalent Ox-Ox connections on the two copper sites
        R_l = r_Cu2 - r_Cu1 # Cu1 - Cu2 vector
        
        print('orthogonality check R_m R_n', np.dot(R_m, R_n) / (np.linalg.norm(R_m) * np.linalg.norm(R_n)))
        print('orthogonality check R_m R_l', np.dot(R_m, R_l) / (np.linalg.norm(R_m) * np.linalg.norm(R_l)))
        print('orthogonality check R_n R_l', np.dot(R_n, R_l) / (np.linalg.norm(R_n) * np.linalg.norm(R_l)))

        center_nl = (0.0, 0.0, 0.0)
        center_nm = (0.0, 0.0, 0.0) #density.R_vec / np.linalg.norm(density.R_vec) * 1.201 # Cu1 - Cu2 vector scaled to 1.201 Angstrom^-1
        center_lm = (0.0, 0.0, 0.0)

        fft_perform_cut_planes_oblique(u_axis=R_n, v_axis=R_l, center=center_nl, seedname_perp_cut='CuCu')
        fft_perform_cut_planes_oblique(u_axis=R_n, v_axis=R_m, center=center_nm, seedname_perp_cut='CuOx4')
        fft_perform_cut_planes_oblique(u_axis=R_l, v_axis=R_m, center=center_lm, seedname_perp_cut='CuOx2CuOx2')

    # test_shift()
    # exit()

    # test plotting
    # twoD_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # plot_2D_fft(twoD_data)


def workflow_density_vs_cutoff_radius(site_idx=[0], site_radii_all=[[i] for i in np.arange(0.1, 2.0, 0.05)], plot=True, save_data=True):
    """Workflow for the analysis of the density vs. cutoff-radius.

    Args:
        site_idx (list, optional): List of site indices. Defaults to [0].
        site_radii_all (list, optional): List of site radii. Defaults to [[i] for i in np.arange(0.1, 2.0, 0.05)].
        plot (bool, optional): Plot the results. Defaults to True.
        save_data (bool, optional): Save the data. Defaults to True.

    Returns:
        tuple: Total charge in the unit cell, total absolute charge in the unit cell
    """
    

# ===================== WORKFLOW DENSITY vs. cutoff-radius =====================
    fname_cube_file = './cube_files/Cu2AC4_rho_sz_512.cube' #'./cube_files/Mn2GeO4_rho_sz.cube'
    
    permutation = None #!! for Mn2GeO4 need to use [2,1,0] to swap x,y,z -> z,y,x
    
    rho_tot_all = []
    rho_abs_tot_all = []

    for site_radii in site_radii_all:
            # ---- READ CUBE FILE -----
        density = Density(permutation=permutation, verbose=True, fname_cube_file=fname_cube_file, scale_factor=scale_factor,)

        rho_tot_unitcell, rho_abs_tot_unitcell = density.integrate_cube_file()


        # ---- MASKING -----

        # leave_sites = {'site_centers':[(0.5, 0.5, 0.5)], 'site_radii':[0.5]}
        if site_idx and site_radii:
            site_centers = density.get_sites_of_atoms(site_idx)
            print('site_centers', site_centers)
            leave_sites = {'site_centers':site_centers, 'site_radii':site_radii}
            density.mask_except_sites(leave_sites)

        print('After masking:')
        rho_tot, rho_abs_tot = density.integrate_cube_file()

        rho_tot_all.append(rho_tot)
        rho_abs_tot_all.append(rho_abs_tot)

    # plotting
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(site_radii_all, rho_tot_all, 's-', label='rho_tot', markerfacecolor='none')
        plt.plot(site_radii_all, rho_abs_tot_all, 'o-', label='rho_abs_tot', markerfacecolor='none')
        plt.xlabel('site_radii (Angstrom)')
        plt.ylabel('charge')
        plt.title(f'Total charge in the unit cell {rho_tot_unitcell:.4f} e.\nTotal absolute charge in the unit cell {rho_abs_tot_unitcell:.4f} e.')
        plt.legend()
        plt.savefig('rho_vs_cutoff_radius.pdf')
        plt.close()

    if save_data:
        data_all = np.array([np.array(site_radii_all)[:,0], rho_tot_all, rho_abs_tot_all])
        np.savetxt('rho_vs_cutoff_radius.txt', data_all, delimiter='\t', header=f'rho_tot_unitcell {rho_tot_unitcell:.6e} e, rho_abs_tot_unitcell {rho_abs_tot_unitcell:.6e} e\nsite_radii\trho_tot\trho_abs_tot')

    return rho_tot_all, rho_abs_tot_all


def workflow_autocorrelation_term(parameters_model, scale_R_array=[1.0], output_folder='.', write_cube_files=False, site_idx=[0]):
    """Workflow for the analysis of the autocorrelation term.

    Args:
        parameters_model (dict): Parameters of the model.
        scale_R_array (list, optional): List of scaling factors. Defaults to [1.0].
        output_folder (str, optional): Output folder. Defaults to '.'.
        write_cube_files (bool, optional): Write cube files. Defaults to False.
        site_idx (list, optional): List of site indices. Defaults to [0].

    Returns:
        tuple: Form factor, overlap term, E_perp term
    """
    R_base = np.array([3.01571, 6.45289, 4.99992]) - np.array([4.85991, 5.28091, 3.56158]) # Cu1 - Cu0 = (-1.8442, 1.17198, 1.43834)
    R_array=[f*R_base for f in scale_R_array]

    fname_cube_file = './cube_files/Cu2AC4_rho_sz_256.cube' #'./cube_files/Mn2GeO4_rho_sz.cube'
    permutation = None #!! for Mn2GeO4 need to use [2,1,0] to swap x,y,z -> z,y,x
    # output_folder = './outputs/Cu2AC4/E_perp/case_23' #_and_oxygens' # 'Mn2GeO4_kz_tomography_64' #'./gaussian/sigma_0.3_distance_1.0' # Mn2GeO4_kz_tomography_64

    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- READ CUBE FILE -----
    orbital = Density(permutation=permutation, verbose=True, fname_cube_file=fname_cube_file, output_folder=output_folder, scale_factor=scale_factor, )
    # orbital.plot_fft_along_line(i_kz=orbital.nc//2, cut_along='stripes', kx_ky_fun=None, k_dist_lim=15, N_points=3001, fout_name=f'{output_folder}/test_1D_plot_along.png')


    # copy to a new orbital object
    orbital_shifted_plus = deepcopy(orbital)
    orbital_shifted_minus = deepcopy(orbital)

    # ---- REPLACE BY MODEL -----
    site_centers = orbital.get_sites_of_atoms(site_idx=site_idx)
    parameters_model['centers'] = site_centers
    orbital.replace_by_model(parameters=parameters_model, leave_as_wavefunction=True)
    
    # if write_cube_files:
    #     orbital.write_cube_file_rho_sz(fout='orbital.cube')  #- beware - after writing out the cube file, some data is maybe missing in the object

    # conjugate it
    orbital.conjugate()
    # orbital_conj_array = deepcopy(orbital.array)
    # orbital.write_cube_file_rho_sz(fout=folder_out + '/orbital_conjugate.cube')

    # make a density out of it
    density = deepcopy(orbital)
    density.square_data()
    # density.integrate_cube_file()

    if write_cube_files:
        density.write_cube_file_rho_sz(fout='density.cube')

    # get its density's FFT (the form factor)
    density.FFT()
    form_factor = np.abs(density.F)

    form_factor_term_sq_integrated_all = []
    overlap_term_sq_integrated_all = []
    E_perp_sq_integrated_all = []

    for scale_R, R in zip(scale_R_array, R_array):
        appendix = f'scale-R_{scale_R:.2f}_scale-factor_{scale_factor:.2f}'
        # ---- REPLACE 2 BY MODEL -----
        # add the displacement R to all site centers
        site_centers_plus = [tuple(np.array(r) + np.array(R)) for r in site_centers]
        site_centers_minus = [tuple(np.array(r) - np.array(R)) for r in site_centers]

        # add inversion! of sites as well! - fake, not working so well
        order = [0, 2, 1, 4, 3] # switch oxygen orbitals in a centrosymmetric fashion
        site_centers_plus = [site_centers_plus[i] for i in order]
        site_centers_minus = [site_centers_minus[i] for i in order]

        parameters_model['centers'] = site_centers_plus
        orbital_shifted_plus.replace_by_model(parameters=parameters_model, leave_as_wavefunction=True)

        orbital_shifted_plus.square_data()
        if write_cube_files:
            orbital_shifted_plus.write_cube_file_rho_sz(f'density_shifted_plus_{appendix}.cube')  #- beware - after writing out the cube file, some data is maybe missing in the object

        # if write_cube_files:
        #     orbital_shifted_plus.write_cube_file_rho_sz(fout=f'orbital_shifted_plus_{appendix}.cube')  #- beware - after writing out the cube file, some data is maybe missing in the object
        # update density_2 by multiplying with density (in-place)
        orbital_shifted_plus.multiply_with(orbital)
        # if write_cube_files:
        #     orbital_shifted_plus.write_cube_file_rho_sz(fout=f'orbital_shifted_plus_times_orbital_{appendix}.cube')  #- beware - after writing out the cube file, some data is maybe missing in the object
        
        parameters_model['centers'] = site_centers_minus
        orbital_shifted_minus.replace_by_model(parameters=parameters_model, leave_as_wavefunction=True)
        if write_cube_files:
            orbital_shifted_minus.write_cube_file_rho_sz(fout=f'orbital_shifted_minus_{appendix}.cube')  #- beware - after writing out the cube file, some data is maybe missing in the object
        # update density_2 by multiplying with density (in-place)   
        orbital_shifted_minus.multiply_with(orbital)

        # scalar
        R_phiphi_plus, _ = orbital_shifted_plus.integrate_cube_file()
        print('R_phiphi_plus', R_phiphi_plus)
        R_phiphi_minus, _ = orbital_shifted_minus.integrate_cube_file()
        print('R_phiphi_minus', R_phiphi_minus)

        # FFT of R_phiphi
        orbital_shifted_plus.FFT()
        orbital_shifted_minus.FFT()

        R_tilde_phiphi_plus = orbital_shifted_plus.F
        R_tilde_phiphi_minus = orbital_shifted_minus.F

        cos_prefactor = np.sqrt( (1-np.cos(R[0]*orbital.kx_cart_mesh + R[1]*orbital.ky_cart_mesh + R[2]*orbital.kz_cart_mesh)) / 2)
        # density.F_abs_sq = np.abs(cos_prefactor)**2
        # density.write_cube_file_fft('cos_prefactor_squared.cube')   # density here is just a surrogate to save the relevant data

        form_factor_term = np.multiply(cos_prefactor, form_factor)
        density.F_abs_sq = np.abs(form_factor_term)**2
        if write_cube_files:
            density.write_cube_file_fft(f'cos_prefactor_times_form_factor_squared_{appendix}.cube')   # density here is just a surrogate to save the relevant data
        
        form_factor_term_sq_integrated, _ = density.integrate_cube_file(fft=True)
        form_factor_term_sq_integrated_all.append(form_factor_term_sq_integrated)

        overlap_term = R_phiphi_plus * R_tilde_phiphi_minus + \
                - R_phiphi_minus * R_tilde_phiphi_plus
        density.F_abs_sq = np.abs(overlap_term)**2
        if write_cube_files:
            density.write_cube_file_fft(f'overlap_term_squared_{appendix}.cube')   # density here is just a surrogate to save the relevant data
        overlap_term_sq_integrated, _ = density.integrate_cube_file(fft=True)
        overlap_term_sq_integrated_all.append(overlap_term_sq_integrated)


        E_perp = form_factor_term + overlap_term
        density.F_abs_sq = np.abs(E_perp)**2
        if write_cube_files:
            density.write_cube_file_fft(f'E_perp_squared_{appendix}.cube')   # density here is just a surrogate to save the relevant data
        E_perp_sq_integrated, _ = density.integrate_cube_file(fft=True)
        E_perp_sq_integrated_all.append(E_perp_sq_integrated)

        # colormaps
        # plot the 2D FFT
        fft_figsize = (4.5, 4.5)
        fft_dpi = 400
        kmax = 19
        fft_xlims = list( kmax * np.array([-1,1]) ) # 1/Angstrom
        fft_ylims = list( kmax * np.array([-1,1]) ) # 1/Angstrom

        i_kz_center = density.nkc//2 # density.get_i_kz(kz_target=density.nkc/2)

        i_kz_array =  [ i_kz_center + i for i in range(-4,5) ] # + i for i in range(-10, 10, 2) ] #range(density.nkc)
        
        for i_kz in i_kz_array:

            cut_along = 'along_stripes' # 'perpendicular_to_stripes' # 'both'
            cax_saturation = None
            N_kpoints = 101 # 3001

            kx_arr_along, ky_arr_along, F_abs_sq_interp_along, kx_arr_perp, ky_arr_perp, F_abs_sq_interp_perp, kx_arr_along_opposite, ky_arr_along_opposite, F_abs_sq_interp_along_opposite  = density.plot_fft_along_line(i_kz=i_kz, cut_along=cut_along, kx_ky_fun=None, k_dist_lim=12, N_points=N_kpoints, fout_name=f'{density.output_folder}/cut_1D_E_perp_{cut_along}.png', cax_saturation=cax_saturation,)
            
            density.plot_fft_2D(i_kz=i_kz, 
                                fout_name=f'F_abs_sq_{appendix}_overlap-term_kz_{density.get_kz_at_index(i_kz):.4f}.png', 
                                figsize=fft_figsize,
                                dpi=fft_dpi, 
                                fixed_z_scale=False,
                                xlims=fft_xlims,
                                ylims=fft_ylims,
                                cut_along=cut_along,
                                plot_line_cut=True, kx_arr_along=kx_arr_along, ky_arr_along=ky_arr_along,
                                kx_arr_perp=kx_arr_perp, ky_arr_perp=ky_arr_perp,
                                data_to_plot=(np.abs(overlap_term)**2)[:,:,i_kz],
                                normalized=False,
                                quantity_name=r'$|E_\mathrm{overlap}|^2$',
                                )
            
            density.plot_fft_2D(i_kz=i_kz, 
                                fout_name=f'F_abs_sq_{appendix}_form-factor_term_kz_{density.get_kz_at_index(i_kz):.4f}.png', 
                                figsize=fft_figsize,
                                dpi=fft_dpi, 
                                fixed_z_scale=False,
                                xlims=fft_xlims,
                                ylims=fft_ylims,
                                data_to_plot=(np.abs(form_factor_term)**2)[:,:,i_kz],
                                normalized=False,
                                quantity_name=r'$|F|^2$',
                                )


    # save data
    with open(f'{output_folder}/E_perp_sq_vs_scale_R.txt', 'w+') as fw:
        np.savetxt(fw, np.vstack([scale_R_array, form_factor_term_sq_integrated_all, overlap_term_sq_integrated_all, E_perp_sq_integrated_all]).T, \
                        header='r/R\t|E_ff|^2\t|E_overlap|^2\t|E_perp|^2', delimiter='\t')

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(4., 3.))
    # semilogy
    plt.semilogy()
    plt.plot(scale_R_array, form_factor_term_sq_integrated_all, 's-', label=r'$|E_\mathrm{ff}|^2$', markerfacecolor='none')
    plt.plot(scale_R_array, overlap_term_sq_integrated_all, 'o-', label=r'$|E_\mathrm{overlap}|^2$', markerfacecolor='none')
    plt.plot(scale_R_array, E_perp_sq_integrated_all, '^-', label=r'$|E_\perp|^2$', markerfacecolor='none')
    plt.xlabel(r'r/R')
    plt.ylabel(r'$|E_i|^2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/E_perp_sq_vs_scale_R.png', dpi=400)


site_idx_all = [
    [0], #0            
    [1], #1
    [0,1], #2
    [0, 16, 25, 9, 40], #3
    [1, 41, 8, 24, 17], #4
    [0, 16, 25, 9, 40, 1, 41, 8, 24, 17,], #5
    None, #6
    [0], #7
    [0,1], #8
    [0,1], #9
    [0], #10
    [0], #11    
    [0], #12
    [1,], #13
    [25,], #14
    [25,], #15
    [40], #16
    [9], #17
    [16], #18
    [0, 25, 40, 9, 16], #19
    [0], #20
    [0, 25, 40, 9, 16], #21
    [0, 25, 40, 9, 16], #22
    [0, 25, 40, 9, 16, 1, 41, 24, 17, 8], #23
    [25], #24
    [40], #25
    [9], #26
    [16], #27
    [0], #28
    [0, 25, 40, 9, 16], #29
    [0, 25, 40, 9, 16, 1, 41, 24, 17, 8], #30
    [0, 25, 40, 9, 16, 1, 41, 24, 17, 8], #31
    [0, 25, 40, 9, 16], #32
    [0, 25, 40, 9, 16], #33
    [0], #34
]

site_radii_all = [
    [r_mt_Cu], #0
    [r_mt_Cu], #1
    [r_mt_Cu]*2, #2
    [r_mt_Cu] + 4*[r_mt_O], #3
    [r_mt_Cu] + 4*[r_mt_O], #4
    [r_mt_Cu] + 4*[r_mt_O] + [r_mt_Cu] + 4*[r_mt_O], #5
    None, #6
    [r_mt_Cu], #7
    [r_mt_Cu]*2, #8
    [r_mt_Cu]*2, #9
    [r_mt_Cu]*1, #10
    [r_mt_Cu]*1, #11
    [r_mt_Cu*1.1]*1, #12
    [r_mt_Cu]*1, #13
    [r_mt_O], #14
    [r_mt_O], #15
    [r_mt_O], #16
    [r_mt_O], #17
    [r_mt_O], #18
    [r_mt_Cu]+[r_mt_O]*4, #19
    [r_mt_Cu], #20
    [r_mt_Cu]+[r_mt_O]*4, #21
    [r_mt_Cu]+[r_mt_O]*4, #22
    [r_mt_Cu]+[r_mt_O]*4+[r_mt_Cu]+[r_mt_O]*4, #23
    [r_mt_O], #24
    [r_mt_O], #25
    [r_mt_O], #26
    [r_mt_O], #27
    [r_mt_Cu], #28
    [r_mt_Cu]+[r_mt_O]*4, #29
    [r_mt_Cu]+[r_mt_O]*4+[r_mt_Cu]+[r_mt_O]*4, #30
    [r_mt_Cu]+[r_mt_O]*4+[r_mt_Cu]+[r_mt_O]*4, #31
    [r_mt_Cu]+[r_mt_O]*4, #32
    [r_mt_Cu]+[r_mt_O]*4, #33
    [r_mt_Cu], #34
]


output_folders_all = [
    base_path+f'masked_Cu0_scale-factor_{scale_factor:.2f}_norm', #0
    base_path+'masked_Cu1', #1
    base_path+'masked_Cu0-1', #2
    base_path+f'masked_Cu0_and_oxygens_scale-factor_{scale_factor:.2f}_norm', #3
    base_path+'masked_Cu1_and_oxygens', #4
    base_path+f'masked_Cu0-1_and_oxygens_scale-factor_{scale_factor:.2f}_norm', #5
    base_path+'unmasked_unit-cell', #6
    base_path+'masked_0_gaussian_sigma_0.3', #7
    base_path+'masked_0_1_gaussians_sigma_0.3', #8
    base_path+'masked_0_1_gaussians_sigma_0.3_same-sign', #9
    base_path+'masked_0_dxy_rotated_test', #10
    base_path+'masked_0_dx2y2_rotated_fit', #11
    base_path+'masked_0_dx2y2_rotated_normalized_fit', #12
    base_path+'masked_0_two_s_rotated_fit', #13
    base_path+'masked_0_two_px_rotated_fit', #14
    base_path+'masked_0_two_spx_rotated_25', #15
    base_path+'masked_0_two_spx_rotated_40', #16
    base_path+'masked_0_two_spx_rotated_9', #17
    base_path+'masked_0_two_spx_rotated_16', #18
    base_path+'masked_model_Cu0_and_oxygens', #19
    base_path+'masked_model_Cu0_s-dx2y2', #20
    base_path+'masked_model_Cu0_and_oxygens_s-dx2y2', #21
    base_path+f'masked_model_Cu0_and_oxygens_purely_bonding_{scale_factor:.2f}', #22
    base_path+f'masked_model_Cu0-1_and_oxygens_purely_bonding_exact_copies_0_and_1_{scale_factor:.2f}_norm', #23
    base_path+'masked_0_two_spx_correct_rotated_25', #24 - like 15 but with correct n=2 for 2s orbital
    base_path+'masked_0_two_spx_correct_rotated_40', #25 - like 16 but with correct n=2 for 2s orbital
    base_path+'masked_0_two_spx_correct_rotated_9', #26 - like 17 but with correct n=2 for 2s orbital
    base_path+'masked_0_two_spx_correct_rotated_16', #27 - like 18 but with correct n=2 for 2s orbital
    base_path+'masked_model_Cu0_dx2y2_neat', #28 - like 11 but expression slightly revamped - just change of parameters
    base_path+'masked_model_Cu0_and_oxygens_purely_bonding_spx_correct', #29 - like 21 but with correct n=2 for 2s orbital
    base_path+'masked_model_Cu0-1_and_oxygens_purely_bonding_spx_correct', #30
    base_path+'masked_model_Cu0-1_and_oxygens_purely_bonding_spx_correct', #31
    base_path+'masked_model_Cu0_and_oxygens_dx2y2_dz2', #32
    base_path+'masked_model_Cu0_and_oxygens_dx2y2_dsphere', #33
    base_path+'masked_model_Cu0_d_from_occ_matrix_coeffs', #34
]

replace_DFT_by_model_all = [
    False, #0
    False, #1
    False, #2
    False, #3
    False, #4
    False, #5
    False, #6
    True, #7
    True, #8
    True, #9
    True, #10
    True, #11
    True, #12
    True, #13
    True, #14
    True, #15
    True, #16
    True, #17
    True, #18
    True, #19
    True, #20
    True, #21
    True, #22
    True, #23
    True, #24
    True, #25
    True, #26
    True, #27
    True, #28
    True, #29
    True, #30
    True, #31
    True, #32
    True, #33
    True, #34
]

fit_model_to_DFT_all = [
    False, #0
    False, #1
    False, #2
    False, #3
    False, #4
    False, #5
    False, #6
    False, #7
    False, #8
    False, #9
    False, #10
    True, #11
    True, #12
    False, #13
    False, #14
    True, #15
    True, #16
    True, #17
    True, #18
    False, #19
    False, #20
    False, #21
    False, #22
    False, #23
    True, #24
    True, #25
    True, #26
    True, #27
    True, #28
    True, #29
    False, #30
    False, #31
    False, #32
    False, #33
    False, #34
]

parameters_model_all = [{}]*7 + [
    {'type':['gaussian'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[1]}}, #7
    {'type':['gaussian']*2, 'sigmas':[0.3, 0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[1,-1]}}, #8
    {'type':['gaussian']*2, 'sigmas':[0.3, 0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[1,1]}}, #9
    {'type':['dxy'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[1]}}, #10
    {'type':['dx2y2'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[700.256342], 'theta0':[-1.011299], 'phi0':[-0.59835726], 'Z_eff':[12.85111],}}, #11 (old Copper)
    {'type':['dx2y2_normalized'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'theta0':[-1.01], 'phi0':[-0.6001], 'Z_eff':[9.7],}}, #12
    {'type':['two_s'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'theta0':[-1.006], 'phi0':[-0.5933], 'Z_eff':[20.5],}}, #13
    {'type':['two_px'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'theta0':[-1.006], 'phi0':[-0.5933], 'Z_eff':[10],}}, #14
    {'type':['two_spx'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.12095979], 'theta0':[-0.82363378], 'phi0':[-0.59752522], 'Z_eff':[8.5545368], 'C':[0.50774442]}}, #15 (atom 25) <-------- Oxygen 1/4
    {'type':['two_spx'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.1264227], 'theta0':[1.18166768], 'phi0':[2.55285232], 'Z_eff':[8.5995384], 'C':[0.46376494]}}, #16 (atom 40) <-------- Oxygen 2/4
    {'type':['two_spx'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.123107364], 'theta0':[0.0003331], 'phi0':[0.829267292], 'Z_eff':[8.53154405], 'C':[0.543582469]}}, #17 (atom 9) <-------- Oxygen 3/4
    {'type':['two_spx'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.12409948], 'theta0':[0.17039612], 'phi0':[-2.0329591 ], 'Z_eff':[8.57672478], 'C':[0.46376494]}}, #18 (atom 16) <-------- Oxygen 4/4
    {'type':['dx2y2']+4*['two_spx'], 'sigmas':[0.3]*5, 'centers':[], 'fit_params_init_all':{'amplitude':[509.65056, 0.12095979, 0.1264227, 0.123107364, 0.12409948], 
                                                                            'theta0':[-0.99290,-0.82363378, 1.18166768, 0.0003331, 0.17039612], 
                                                                            'phi0':[-0.58594, -0.5933, 2.55285232, 0.829267292, -2.0329591 ], 
                                                                            'Z_eff':[12.2132868, 8.5545368, 8.5995384, 8.53154405, 8.57672478],
                                                                            'C':[0.000, 0.50774442, 0.46376494, 0.543582469, 0.50554664]}}, #19 <-------- Copper (d only) + 4 Oxygens -- mixed bonding and anti-bonding!!
    {'type':['dx2y2_with_four_s'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[691.803173], 'theta0':[-1.01204317], 'phi0':[-0.599982000], 'Z_eff':[12.8281], 'Z_eff_s':[3.59397636], 'C':[0.0057885]}}, #20 <------ Copper 1/1 
    {'type':['dx2y2_with_four_s']+4*['two_spx'], 'sigmas':[0.3]*5, 'centers':[], 'fit_params_init_all':{'amplitude':[691.803173, 0.12095979, 0.1264227, 0.123107364, 0.12409948], 
                                                                    'theta0':[-1.01204317,-0.82363378, 1.18166768, 0.0003331, 0.17039612], 
                                                                    'phi0':[-0.599982, -0.5933, 2.55285232, 0.829267292, -2.0329591 ], 
                                                                    'Z_eff':[12.8281, 8.5545368, 8.5995384, 8.53154405, 8.57672478],
                                                                    'Z_eff_s':[3.59397636, None, None, None, None],
                                                                    'C':[0.0057885, 0.50774442, 0.46376494, 0.543582469, 0.50554664]}}, #21 <-------- (  Copper (sd) + 4 Oxygens --- not much better) 
    {'type':['dx2y2']+4*['two_spx'], 'sigmas':[0.3]*5, 'centers':[], 'fit_params_init_all':{'amplitude':[509.65056, -0.12095979, -0.1264227, 0.123107364, 0.12409948], 
                                                                            'theta0':[-0.99290,-0.82363378, 1.18166768, 0.0003331, 0.17039612], 
                                                                            'phi0':[-0.58594, -0.5933, 2.55285232, 0.829267292, -2.0329591 ], 
                                                                            'Z_eff':[12.2132868, 8.5545368, 8.5995384, 8.53154405, 8.57672478],
                                                                            'C':[0.000, 0.50774442, 0.46376494, 0.543582469, 0.50554664]}}, #22 <-------- purely bonding version of 19
    {'type':['dx2y2']+4*['two_spx']+['dx2y2']+4*['two_spx'], 'sigmas':[0.3]*10, 'centers':[], 
        'spin_down_orbital_all':[False]*5 + [True]*5,
                                                                        'fit_params_init_all':{'amplitude':[509.65056, -0.12095979, -0.1264227, 0.123107364, 0.12409948, 509.65056, -0.12095979, -0.1264227, 0.123107364, 0.12409948], 
                                                                            'theta0':[-0.99290,-0.82363378, 1.18166768, 0.0003331, 0.17039612, -0.99290,-0.82363378, 1.18166768, 0.0003331, 0.17039612], 
                                                                            'phi0':[-0.58594, -0.5933, 2.55285232, 0.829267292, -2.0329591, -0.58594, -0.5933, 2.55285232, 0.829267292, -2.0329591,], 
                                                                            'Z_eff':[12.2132868, 8.5545368, 8.5995384, 8.53154405, 8.57672478, 12.2132868, 8.5545368, 8.5995384, 8.53154405, 8.57672478],
                                                                            'C':[0.000, 0.50774442, 0.46376494, 0.543582469, 0.50554664, 0.000, 0.50774442, 0.46376494, 0.543582469, 0.50554664]}}, #23 <-------- both Cu0 and Cu1 populated with model 22 (Cu1 with spin down - controlled by 'spin_down_orbital_all' key in parameters_model)
    {'type':['two_spx_correct'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[-0.2926634], 'theta0':[-0.82051673], 'phi0':[-0.5980457], 'Z_eff':[5.01517706], 'C':[0.25951331]}}, #24 (atom 25) - like 15 but correct orbitals <-------- Oxygen 1/4,         --->  R^2 0.853924
    {'type':['two_spx_correct'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[-0.30968292], 'theta0':[1.18435744], 'phi0':[2.55194631], 'Z_eff':[4.94533815], 'C':[0.22725085]}}, #25 (atom 40) - like 16 but correct orbitals <-------- Oxygen 2/4  --->  R^2 0.865949
    {'type':['two_spx_correct'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.295919999], 'theta0':[0.00034809], 'phi0':[0.8265912], 'Z_eff':[5.10058135], 'C':[0.291142454]}}, #26 (atom 9) - like 17 but correct orbitals <-------- Oxygen 3/4   --->  R^2 0.842962
    {'type':['two_spx_correct'], 'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.30207433], 'theta0':[0.18104707], 'phi0':[-2.03391158], 'Z_eff':[5.04343411], 'C':[0.28161311]}}, #27 (atom 16) - like 18 but correct orbitals <-------- Oxygen 4/4          --->  R^2 0.852591
    {'type':['dx2y2_neat'],      'sigmas':[0.3], 'centers':[], 'fit_params_init_all':{'amplitude':[0.360453056], 'theta0':[-1.011437], 'phi0':[-0.59855408], 'Z_eff':[12.8481725], 'C':[0.00]}}, #28 - like 11 but more neately defined parameters  <------ copper                --->  R^2 0.784286
    {'type':['dx2y2_neat']+4*['two_spx_correct'], 'sigmas':[0.3]*5, 'centers':[], 'fit_params_init_all':{'amplitude':[0.360453056, -0.2926634, -0.30968292, 0.295919999, 0.30207433], 
                                                                            'theta0':[-1.011437, -0.82051673, 1.18435744, 0.00034809, 0.18104707], 
                                                                            'phi0':[-0.59855408, -0.5980457, 2.55194631, 0.8265912, -2.03391158], 
                                                                            'Z_eff':[12.8481725, 5.01517706, 4.94533815, 5.10058135, 5.04343411],
                                                                            'C':[0.000, 0.25951331, 0.22725085, 0.291142454, 0.28161311]}}, #29 - like 22 but correct sp orbitals - for fitting
    # 30 will be like 23 but with parameters from #29
    {'type':['dx2y2_neat']+4*['two_spx_correct']+['dx2y2_neat']+4*['two_spx_correct'], 'sigmas':[0.3]*10, 'centers':[],
        'spin_down_orbital_all':[False]*5 + [True]*5,
                                                                        'fit_params_init_all':{'amplitude':[0.360453056, -0.2926634, -0.30968292, 0.295919999, 0.30207433, 0.360453056, -0.2926634, -0.30968292, 0.295919999, 0.30207433], 
                                                                                'theta0':[-1.011437, -0.82051673, 1.18435744, 0.00034809, 0.18104707, -1.011437, -0.82051673, 1.18435744, 0.00034809, 0.18104707], 
                                                                                'phi0':[-0.59855408, -0.5980457, 2.55194631, 0.8265912, -2.03391158, -0.59855408, -0.5980457, 2.55194631, 0.8265912, -2.03391158], 
                                                                                'Z_eff':[12.8481725, 5.01517706, 4.94533815, 5.10058135, 5.04343411, 12.8481725, 5.01517706, 4.94533815, 5.10058135, 5.04343411],
                                                                                'C':[0.000, 0.25951331, 0.22725085, 0.291142454, 0.28161311, 0.000, 0.25951331, 0.22725085, 0.291142454, 0.28161311]}}, #30 <-------- both Cu0 and Cu1 populated with model 29 (Cu1 with spin down - controlled by 'spin_down_orbital_all' key in parameters_model)     
    # 31 is like 30 but all orbitals are spin up
    {'type':['dx2y2_neat']+4*['two_spx_correct']+['dx2y2_neat']+4*['two_spx_correct'], 'sigmas':[0.3]*10, 'centers':[],
        'spin_down_orbital_all':[False]*5 + [False]*5,
                                                                        'fit_params_init_all':{'amplitude':[0.360453056, -0.2926634, -0.30968292, 0.295919999, 0.30207433, 0.360453056, -0.2926634, -0.30968292, 0.295919999, 0.30207433], 
                                                                                'theta0':[-1.011437, -0.82051673, 1.18435744, 0.00034809, 0.18104707, -1.011437, -0.82051673, 1.18435744, 0.00034809, 0.18104707], 
                                                                                'phi0':[-0.59855408, -0.5980457, 2.55194631, 0.8265912, -2.03391158, -0.59855408, -0.5980457, 2.55194631, 0.8265912, -2.03391158], 
                                                                                'Z_eff':[12.8481725, 5.01517706, 4.94533815, 5.10058135, 5.04343411, 12.8481725, 5.01517706, 4.94533815, 5.10058135, 5.04343411],
                                                                                'C':[0.000, 0.25951331, 0.22725085, 0.291142454, 0.28161311, 0.000, 0.25951331, 0.22725085, 0.291142454, 0.28161311]}}, #31 <-------- like 30 but all orbitals are spin up
       # add dz2: improving the decay rate of the model (density in-between copper dimers)
       {'type':['dx2y2_with_dz2']+4*['two_spx'], 'sigmas':[0.3]*5, 'centers':[], 'fit_params_init_all':{'amplitude':[691.803173, 0.12095979, 0.1264227, 0.123107364, 0.12409948], 
                                                                    'theta0':[-1.01204317,-0.82363378, 1.18166768, 0.0003331, 0.17039612], 
                                                                    'phi0':[-0.599982, -0.5933, 2.55285232, 0.829267292, -2.0329591 ], 
                                                                    'Z_eff':[12.8281, 8.5545368, 8.5995384, 8.53154405, 8.57672478],
                                                                    'Z_eff_s':[12.8281, None, None, None, None],
                                                                    'C':[0.2, 0.50774442, 0.46376494, 0.543582469, 0.50554664]}}, # 32 Copper (dx2y2-dz2) + 4 Oxygens
       # add dsphere: improving the decay rate of the model (density in-between copper dimers)
        {'type':['dx2y2_neat_with_dsphere']+4*['two_spx_correct'], 'sigmas':[0.3]*5, 'centers':[], 'fit_params_init_all':{'amplitude':[0.360453056, 0.2926634, 0.30968292, -0.295919999, -0.30207433], 
                                                                            'theta0':[-1.011437, -0.82051673, 1.18435744, 0.00034809, 0.18104707], 
                                                                            'phi0':[-0.59855408, -0.5980457, 2.55194631, 0.8265912, -2.03391158], 
                                                                            'Z_eff':[12.8481725, 5.01517706, 4.94533815, 5.10058135, 5.04343411],
                                                                            'Z_eff_s':[12.8481725, None, None, None, None],
                                                                            'C':[1.00, 0.25951331, 0.22725085, 0.291142454, 0.28161311]}}, # 33 Copper (dx2y2-ds) + 4 Oxygens              
        # d orbitals from occupation matrix coefficients
        {'type':['d_from_coeffs'], 'sigmas':[None], 'centers':[], 'fit_params_init_all':{'amplitude':[None], 
                                                                        'theta0':[None], 
                                                                        'phi0':[None], 
                                                                        'Z_eff':[None],
                                                                        'Z_eff_s':[None],
                                                                        'C':[None]}}, # 34 Copper d orbital from occupation matrix coefficients
    ]


# plot in both linear and log scale
def plot_density(r_dist, fX_interp, suffix, 
                log_scale=False, 
                xlims=None,
                ylims=None,
                ylabel=None,
                base_folder='./outputs/Cu2AC4/512/plot_along_line',
                line_name='Cu1Cu2'
                ):
    fig, ax = plt.subplots(1, 1, figsize=(4., 3.))
    if log_scale:
        plt.semilogy()
        suffix = 'log_abs_' + suffix
        fX_interp = np.abs(fX_interp)
    plt.plot(r_dist, fX_interp, '-o', markerfacecolor='none', markersize=0.2)
    plt.xlabel(r'$r$ ($\mathrm{\AA}$)')
    if ylabel is None:
        ylabel = r'$\rho_\mathrm{1D}$ ($\mathrm{\AA}^{-1}$)'
    plt.ylabel(ylabel)
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()  
    plt.savefig(os.path.join(base_folder, f'density_along_{line_name}_line_{suffix}.png'), dpi=400)
    plt.close()
    

def workflow_plot_density(suffix='rho_sz_up-down_512', replace_by_model_number=None, skip_interpolation=False, skip_projection=False,
                          R_vec=None, R_vec_center=None, line_name='Cu1Cu2'):

    print('workflow_plot_density')
    base_folder = './outputs/Cu2AC4/512/plot_along_line' # for hydrogen molecule: '/home/vojace_l/Documents/Cu(II)_acetate/hydrogen_molecule/gamma_only/3x_supercell/r_over_R_4.00/triplet/' #
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    fname_cube_file = f'./cube_files/diverse_calculations/Cu2AC4_{suffix}.cube'  # for hydrogen molecule: os.path.join(base_folder, 'H2_rho_sz.cube') # './cube_files/Mn2GeO4_rho_sz.cube'

    r_cutoff = 2.5 # Angstrom
    center = [3.93781, 5.8669, 4.28075] # for hydrogen molecule: [5.0746, 5.0746, 6.17542] #
    if R_vec is None:
        R_vec = R1 - R2
    if R_vec_center is None:
        R_vec_center = (R1 + R2) / 2

    Angstrom = 1e-10
    a_Bohr = physical_constants['Bohr radius'][0]

    if replace_by_model_number is not None:
        suffix += f'_model-{replace_by_model_number}'


    density = Density(fname_cube_file=fname_cube_file)

    central_radius = 5.0

    if replace_by_model_number is not None:
        print('ood')
        site_idx = site_idx_all[replace_by_model_number]
        site_radii = site_radii_all[replace_by_model_number]
        parameters_model = parameters_model_all[replace_by_model_number]
        fit_model_to_DFT = fit_model_to_DFT_all[replace_by_model_number]

        # ---- MASKING and REPLACING -----
        site_centers = density.get_sites_of_atoms(site_idx)
        leave_sites = {'site_centers':site_centers, 'site_radii':site_radii}
        density.mask_except_sites(leave_sites)

        print('After masking interated:')
        density.integrate_cube_file()
    
        parameters_model['centers'] = site_centers
        density.replace_by_model(fit=fit_model_to_DFT,
                                parameters=parameters_model,
                                leave_sites=leave_sites,
                                leave_as_wavefunction=False,
                                )
        
        # print('After replacing by model interated:')
        # density.integrate_cube_file() 

        # print('and saved file')
        #     # ---- WRITE MODIFIED DENSITY TO CUBE FILE -----
        # write_cube_files = True
        # if write_cube_files:
        #     density.write_cube_file_rho_sz(fout=os.path.join(base_folder, f'rho_sz_model_{suffix}.cube')) 


        # print('exiting now...')
        # exit()

    else:
        # mask 4 Angstrom radius from the origin
        # ---- MASKING -----
        leave_sites = {'site_centers':[tuple((R1+R2)/2)], 'site_radii':[central_radius]}
        density.mask_except_sites(leave_sites)

    if not skip_projection:
        # ---- ALL POINTS PROJECTED -----
        print('After masking integrated:')
        density.integrate_cube_file()
        # all points projected onto R_vec (the Cu1-Cu2 connection unit vector)
        R_vec_unit = R_vec / np.linalg.norm(R_vec)
        # distance of all points projected onto the Cu1-Cu2 unit vector
        r_along_R_vec, density_1D = density.get_line_density(R_vec_unit, R_vec_center, N_points=1001)

        # save projected data to files
        with open(os.path.join(base_folder, f'density_1D_proj_along_{line_name}_line_{suffix}.txt'), 'w+') as fw:
            np.savetxt(fw, np.vstack([r_along_R_vec, density_1D]).T, header='r_dist\trho_proj (Angstrom^-1)', delimiter='\t')
        suffix += '_projected'
        # plot and save
        plot_density(r_along_R_vec, density_1D, suffix, log_scale=False, xlims=(-central_radius, central_radius), base_folder=base_folder, line_name=line_name)
        plot_density(r_along_R_vec, density_1D, suffix, log_scale=True, xlims=(-central_radius, central_radius), ylims=(1e-13, 5.0), base_folder=base_folder, line_name=line_name)

    if not skip_interpolation:
        interpolator_name = os.path.join(base_folder, f'interpolator_{suffix}_along_{line_name}_line_r-cutoff_{r_cutoff:.2f}.pickle')

        if not os.path.exists(interpolator_name):

            x = density.x_cart_mesh.flatten()
            y = density.y_cart_mesh.flatten()
            z = density.z_cart_mesh.flatten()
            include = np.logical_and(np.logical_and(np.abs(x-center[0]) < r_cutoff, np.abs(y-center[1]) < r_cutoff), np.abs(z-center[2]) < r_cutoff)
            X = np.array([x[include], y[include], z[include]]).T
            fX = density.array.flatten()[include] #* (a_Bohr**3 / Angstrom**3) # convert to 1/Angstrom^3
        else:
            X = None
            fX = None

        interpolator = get_interpolator(interpolator_name=interpolator_name, X=X, fX=fX, verbose=True)

        r_dist = np.linspace(-r_cutoff*np.sqrt(2), r_cutoff*np.sqrt(2), 301)
        axis = R_vec / np.linalg.norm(R_vec)
        x = center[0] + axis[0] * r_dist
        y = center[1] + axis[1] * r_dist
        z = center[2] + axis[2] * r_dist
        X_interp = np.array([x, y, z]).T
        print('interpolating...')
        fX_interp = interpolator(X_interp)

        # save data
        with open(os.path.join(base_folder, f'density_along_{line_name}_line_{suffix}.txt'), 'w+') as fw:
            np.savetxt(fw, np.vstack([r_dist, fX_interp]).T, header='r_dist\trho (a_Bohr^-3)', delimiter='\t')

        ylabel = r'$\rho_\mathrm{3D}$ (a_Bohr$^{-3}$)'
        plot_density(r_dist, fX_interp, suffix, log_scale=False, ylabel=ylabel)
        plot_density(r_dist, fX_interp, suffix, log_scale=True, ylabel=ylabel)


def workflow_plot_all_densities_integrated():
        # PLOT THE ELECTRONIC AND SPIN DENSITY ALONG LINES ETC.
    replace_by_model_number = None # 30 # 
    skip_interpolation = True #True # 
    skip_projection = False # False #

            #suffix='rho_up-up_512',  #suffix='rho_up-down_512',   #suffix='rho_sz_up-up_512',   #suffix='rho_sz_up-down_512', 
    workflow_plot_density(suffix='rho_up-up_512',
                          replace_by_model_number=replace_by_model_number, 
                          skip_interpolation=skip_interpolation,
                          skip_projection=skip_projection
                          )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'
    
            #suffix='rho_up-up_512',  #suffix='rho_up-down_512',   #suffix='rho_sz_up-up_512',   #suffix='rho_sz_up-down_512', 
    workflow_plot_density(suffix='rho_up-down_512',
                          replace_by_model_number=replace_by_model_number, 
                          skip_interpolation=skip_interpolation,
                          skip_projection=skip_projection
                          )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'
    
            #suffix='rho_up-up_512',  #suffix='rho_up-down_512',   #suffix='rho_sz_up-up_512',   #suffix='rho_sz_up-down_512', 
    workflow_plot_density(suffix='rho_sz_up-up_512',
                          replace_by_model_number=replace_by_model_number, 
                          skip_interpolation=skip_interpolation,
                          skip_projection=skip_projection
                          )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'
    
            #suffix='rho_up-up_512',  #suffix='rho_up-down_512',   #suffix='rho_sz_up-up_512',   #suffix='rho_sz_up-down_512', 
    workflow_plot_density(suffix='rho_sz_up-down_512',
                          replace_by_model_number=replace_by_model_number, 
                          skip_interpolation=skip_interpolation,
                          skip_projection=skip_projection
                          )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'
    
            #suffix='rho_up-up_512',  #suffix='rho_up-down_512',   #suffix='rho_sz_up-up_512',   #suffix='rho_sz_up-down_512', 
    workflow_plot_density(suffix='rho_sz_up-down_512',
                          replace_by_model_number=30, 
                          skip_interpolation=skip_interpolation,
                          skip_projection=skip_projection
                          )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'
    
            #suffix='rho_up-up_512',  #suffix='rho_up-down_512',   #suffix='rho_sz_up-up_512',   #suffix='rho_sz_up-down_512', 
    workflow_plot_density(suffix='rho_sz_up-up_512',
                          replace_by_model_number=31, 
                          skip_interpolation=skip_interpolation,
                          skip_projection=skip_projection
                          )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'


def workflow_plot_all_densities_interpolated():

    # PLOT THE ELECTRONIC AND SPIN DENSITY ALONG LINES ETC.
    replace_by_model_number = None # 30 # 
    skip_interpolation = False #True #
    skip_projection = True # False #


    # 1. ============== oxygen 18-34 ==============
    R_vec = R_O18 - R_O34
    R_vec_center=(R_O18 + R_O34) / 2
    line_name = 'O18O34'

    # 2. ============== oxygen 9-1 ==============
    R_vec = R_O9 - R_O1
    R_vec_center=(R_O9 + R_O1) / 2
    line_name = 'O9O1'

    # calculate DFT rho_sz up-up, DFT rho_sz up-down, model rho_sz up-up, model rho_sz up-down, DFT rho up-up, DFT rho up-down 
    suffix_all = ['rho_sz_up-up_512', 'rho_sz_up-down_512', 
                   'rho_sz_up-up_512', 'rho_sz_up-down_512', 
                   'rho_up-up_512','rho_up-down_512']
    replace_by_model_all = [None, None, 31, 30, None, None]

    # for hydrogen molecule:
    # R_H1 = np.array([5.0746, 5.0746, 4.69105]) # H1
    # R_H2 = np.array([5.0746, 5.0746, 7.65979]) # H2
    # R_vec = R_H1 - R_H2
    # R_vec_center = (R_H1 + R_H2) / 2
    # suffix_all = ['rho_sz_up-up_512']
    # replace_by_model_all = [None]

    for suffix, replace_by_model in zip(suffix_all, replace_by_model_all):
        workflow_plot_density(suffix=suffix,
                            line_name=line_name,
                            replace_by_model_number=replace_by_model, 
                            skip_interpolation=skip_interpolation,
                            skip_projection=skip_projection,
                            R_vec=R_vec,
                            R_vec_center=R_vec_center
                            )   # suffix='rho_up-down_256'   # suffix='rho_sz_up-down_512'


def xsf_to_cube(xsf_name='Cu2AC4_00001.xsf', base_folder='.', normalize=True):
    """
    Convert an XSF file to a CUBE file.
    
    params:
        xsf_name: name of the XSF file to convert
    """
    xsf_path = os.path.join(base_folder, xsf_name)
    with open(xsf_path, 'r') as f:
        array, origin, span_vectors, atoms = read_xsf(f, read_data=True)

    if normalize is True:
        # condition -- for the density -- is that the array**2 summed and multiplied by the volume belonging to a single point (total unit cell volume / number of points) be equal to 1
        # get the unti cell size from atoms
        unit_cell = atoms.get_cell()
        norm_factor = (np.sum(array**2) * unit_cell.volume / a0**3 / array.size)
        array /= np.sqrt(norm_factor)

    with open(os.path.join(base_folder, xsf_name.split('.')[0]+'.cube'), 'w') as f:
        write_cube_ase(f, atoms, data=array, origin=origin, comment=None)
        # shift back from origin
        print('origin', origin)
        # print('array', array)
        print('atoms', atoms)


def workflow_wannier(base_folders=['.', '.'], xsf_name='Cu2AC4_00001.xsf', spins=['up', 'down'],
                     common_folder='.', scale_factor=1.0):
    """
    Workflow to convert an XSF file to a CUBE file and produce the density.
    params:
        xsf_name: name of the XSF file to convert
        spin: 'up' or 'down' for the spin density
    """

    def do_for_each(i_wannier):
        base_folder = base_folders[i_wannier]
        spin = spins[i_wannier]

        cube_name = xsf_name.split('.')[0] + '.cube'
        cube_path = os.path.join(base_folder, cube_name)
        if True: #not os.path.exists(cube_path):
            xsf_to_cube(xsf_name=xsf_name, base_folder=base_folder, normalize=True)

        orbital = Density(fname_cube_file=cube_path, output_folder=base_folder, scale_factor=scale_factor)

        density = deepcopy(orbital)
        density.square_data()
        
        # --- PRODUCE DENSITY FROM THE ORBITAL DATA ---
        density_cube_name = cube_name.split('.')[0] + '_rho_sz.cube'
        density_cube_path = os.path.join(base_folder, density_cube_name)

        if spin == 'up':
            pass
        elif spin == 'down':
            density.array = -density.array
        else:
            raise ValueError("'spin' must be 'up' or 'down'")
        
        if True: #not os.path.exists(density_cube_path):
            # write, but make a copy first
            density_to_write = deepcopy(density)
            density_to_write.write_cube_file_rho_sz(fout=density_cube_name, output_folder=base_folder)
        return orbital, density

    def plot_density_for_wannier(base_folder, density, suffix='wannier'):
        # --- PLOT THE DENSITY ALONG THE Cu0-Cu1 line (1D - projected) ---
        line_name = 'Cu0Cu1'
        central_radius = 5.0  # Angstrom
        # !!!!! the Wannier unit cell has shifted origin (all atomic positions and data values are shifted by the origin vector) !!!!!!
        # search for origin=origin in the code (it's about 0.05 in each direction)
        # this is because xcrysden file format xsf is working with a (1,1,1) supercell
        # there is no easy fix around this
        # you cannot specify positions for the data array
        # and exporting directly in cube format form wannier90 has it's own other quirks -- very difficult
        # !!!! ALSO NOTE THAT LABELING OF THE ATOMS IN THE NEW CUBE FILE IS DIFFERENT FROM THE ORIGINAL CUBE FILE FROM QUANTUM ESPRESSO !!!!!
        # AGAIN a quirk of wannier90
        R1_xsf = np.array((4.90913, 5.35425, 3.64720)) # np.array(list(self.metadata['atoms'][R_idx1][1])[1:])*physical_constants['Bohr radius'][0]*1e10
        R2_xsf = np.array((3.06493, 6.52622, 5.08553)) #np.array(list(self.metadata['atoms'][R_idx2][1])[1:])*physical_constants['Bohr radius'][0]*1e10
        R_vec = R1_xsf - R2_xsf
        R_vec_center = (R1_xsf + R2_xsf) / 2
        print('After masking integrated:')
        density.integrate_cube_file()
        # all points projected onto R_vec (the Cu1-Cu2 connection unit vector)
        R_vec_unit = R_vec / np.linalg.norm(R_vec)
        # distance of all points projected onto the Cu1-Cu2 unit vector
        r_along_R_vec, density_1D = density.get_line_density(R_vec_unit, R_vec_center, N_points=1001)

        # save projected data to files
        with open(os.path.join(base_folder, f'density_1D_proj_along_{line_name}_line_{suffix}.txt'), 'w+') as fw:
            np.savetxt(fw, np.vstack([r_along_R_vec, density_1D]).T, header='r_dist\trho_proj (Angstrom^-1)', delimiter='\t')
        suffix += '_projected'
        # plot and save
        plot_density(r_along_R_vec, density_1D, suffix, log_scale=False, xlims=(-central_radius, central_radius), base_folder=base_folder, line_name=line_name)
        plot_density(r_along_R_vec, density_1D, suffix, log_scale=True, xlims=(-central_radius, central_radius), ylims=(1e-13, 5.0), base_folder=base_folder, line_name=line_name)
    
    orbital_up, density_up = do_for_each(i_wannier=0)
    plot_density_for_wannier(base_folder=base_folders[0], density=density_up)
    orbital_down, density_down = do_for_each(i_wannier=1)
    plot_density_for_wannier(base_folder=base_folders[1], density=density_down)

    # --- COMBINE THE DENSITIES ---
    spin_density = deepcopy(density_up)
    spin_density.array = density_up.array + density_down.array
    # write cube file of the combined density
    spin_density.write_cube_file_rho_sz(fout=os.path.join(common_folder, f'Cu2AC4_combined_up_and_down_rho_sz.cube'))
    plot_density_for_wannier(base_folder=common_folder, density=spin_density)

    # --- PLOT their fourier transform (form-factor term) ---

    # for the middle i_kz, plot also line cuts
    i_kz = spin_density.nkc//2
    fft_as_log = False
    appendix = 'Wannier'
    fft_figsize = (4.5, 4.5)
    fft_dpi = 400
    fft_xlims = [-19, 19] # 1/Angstrom
    fft_ylims = [-19, 19] # 1/Angstrom
    fft_zlims = None #[0, 6.4e4] # 1/Angstrom
    normalized = True
    cax_saturation = 0.5
    # along stripes
    for cut_along in ['along_stripes', 'perpendicular_to_stripes', 'both', 'along_opposite_stripes']:
        kx_arr_along, ky_arr_along, F_abs_sq_interp_along, kx_arr_perp, ky_arr_perp, F_abs_sq_interp_perp, kx_arr_along_opposite, ky_arr_along_opposite, F_abs_sq_interp_along_opposite = spin_density.plot_fft_along_line(i_kz=i_kz, cut_along=cut_along, kx_ky_fun=None, k_dist_lim=12, N_points=3001, fout_name=f'{spin_density.output_folder}/cut_1D_{cut_along}.png', cax_saturation=cax_saturation, normalized=normalized,)
        spin_density.plot_fft_2D(i_kz=i_kz, fft_as_log=fft_as_log, 
                    fout_name=f'F_abs_sq{appendix}-scale_kz_at_idx_{i_kz}_cut_{cut_along}.png', 
                    figsize=(5.5, 4.5),
                    dpi=fft_dpi,
                    fixed_z_scale=True,
                    cax_saturation=cax_saturation,
                    xlims=fft_xlims,
                    ylims=fft_ylims, 
                    zlims=fft_zlims,
                    plot_line_cut=True, 
                    kx_arr_along=kx_arr_along, ky_arr_along=ky_arr_along,
                    kx_arr_perp=kx_arr_perp, ky_arr_perp=ky_arr_perp,
                    kx_arr_along_opposite=kx_arr_along_opposite, ky_arr_along_opposite=ky_arr_along_opposite,
                    cut_along=cut_along, 
                    normalized=normalized,)
        np.savetxt(os.path.join(spin_density.output_folder, 'cut_1D_both.txt'), np.array([kx_arr_along, ky_arr_along, F_abs_sq_interp_along, kx_arr_perp, ky_arr_perp, F_abs_sq_interp_perp, kx_arr_along_opposite, ky_arr_along_opposite, F_abs_sq_interp_along_opposite]).T, delimiter='\t', fmt='%.8e', header='kx_along\tky_along\tF_abs_sq_along\tkx_perp\tky_perp\tF_abs_sq_perp\tkx_along_opposite\tky_along_opposite\tF_abs_sq_along_opposite')


    # get the overlap density and calculate the overlap term
    overlap_density = deepcopy(orbital_up)
    overlap_density.array = np.conj(orbital_up.array) * orbital_down.array

    def E_overlap(rho_overlap):
        """
        Calculate the overlap energy term from the overlap density.
        """
        # integrate the overlap density
        rho_tot, abs_rho_tot = rho_overlap.integrate_cube_file()
        # calculate the energy
        E_overlap = np.sum(rho_overlap.array) * rho_overlap.unit_cell_volume / a0**3

def exchange_integral(rho1, rho2, r1, r2):
    """Calculate the exchange integral between two orbitals.

    Args:
        rho1 (np.ndarray): Wavefunction of the first orbital.
        rho2 (np.ndarray): Wavefunction of the second orbital. Same shape as rho1.
        r1 (np.ndarray): Position vector of the first orbital. It has the shape of (rho1.shape[0], 3) with x,y and z coordinates in dedicated columns.
        r2 (np.ndarray): Position vector of the second orbital. It has the shape of (rho2.shape[0], 3) with x,y and z coordinates in dedicated columns.
    """

    # calculate the exchange integral
    # This is a placeholder for the actual implementation
    # The actual implementation would involve calculating the integral over the product of the wavefunctions
    # at positions r1 and r2, taking into account the exchange symmetry.
    



def workflow_exchange_integrals(rho1, rho2):
    """Calculate the exchange integral between two orbitals.

    Args:
        rho1 (Density object): With leave_as_wavefunction=True.
        rho2 (Density object): With leave_as_wavefunction=True.
    """
    if rho1.leave_as_wavefunction is False or rho2.leave_as_wavefunction is False:
        raise ValueError("Both Density objects must have leave_as_wavefunction=True to calculate overlap integrals.")

    # calculate the exchange integral
    
    
if __name__ == '__main__':

    # ------ WANNNIER -----
    spins=['up', 'down'] # for Cu0 the wannierized orbital (unoccupied - hole) is spin down so the occupied net density is spin up  
    base_folders = ['/home/vojace_l/Documents/Cu(II)_acetate/from_daint_alps/Wannierization/singlet_spin_polarized/nscf_kpoints2x2x2/case_07_unoccupied_above_Fermi_Cu0_so_spin-down_1000_1000', \
                    '/home/vojace_l/Documents/Cu(II)_acetate/from_daint_alps/Wannierization/singlet_spin_polarized/nscf_kpoints2x2x2/case_08_unoccupied_above_Fermi_Cu1_so_spin-up_1000_1000']
    common_folder = '/home/vojace_l/Documents/Cu(II)_acetate/from_daint_alps/Wannierization/singlet_spin_polarized/nscf_kpoints2x2x2/joint_case_07_case_08'

    xsf_name='Cu2AC4_00001.xsf'
    # workflow_wannier(base_folders=base_folders, spins=spins, xsf_name=xsf_name, common_folder=common_folder, scale_factor=scale_factor)
    # exit()

    # workflow_density_vs_cutoff_radius(site_idx=[0], site_radii_all=[[i] for i in np.arange(0.5, 4.0, 0.5)], plot=True)
    # exit()

    # workflow_plot_all_densities_integrated()
    # workflow_plot_all_densities_interpolated()

    # ===== RUN a single case =====
    run_a_single_case = False
    if run_a_single_case:
        output_folder = './' #'./outputs/Cu2AC4/512/masked_Cu0_and_oxygens' #_and_oxygens' # 'Mn2GeO4_kz_tomography_64' #'./gaussian/sigma_0.3_distance_1.0' # Mn2GeO4_kz_tomography_64
        site_idx = [0, 25, 40, 9, 16, 1, 41, 24, 17, 8] #[0, 16, 25, 9, 40] #, 1] # 16, 25, 9, 40, 1, 41, 8, 24, 17] #  16, 25, 9, 40] #[0]# None #[0,  16, 25, 9, 40, 1, 41, 8, 24, 17] #[1, 41, 8, 24, 17, 0,  16, 25, 9, 40] #[0] #, 16, 25, 9, 40]  # [0] #,  16, 25, 9, 40] #
        site_radii = [r_mt_Cu] + 4*[r_mt_O] + [r_mt_Cu]+ 4*[r_mt_O]
        workflow(output_folder=output_folder, site_idx=site_idx, site_radii=site_radii, replace_DFT_by_model=False, parameters_model=parameters_model_all[29], fit_model_to_DFT=False)
        exit()

    # ===== RUN selected cases among the predefined ones =====

    run_cases = [34] #[33] #None #[5] #[0, 3, 5, 23] #, 3, 5, 23] #, 3, 5] # None

    # case = 29
    # scale_R_array = [1.0] #[1.00]
    # workflow_autocorrelation_term(parameters_model_all[case], 
    #                                 scale_R_array=scale_R_array, 
    #                                 output_folder=base_path+'masked_model_Cu0_and_oxygens_purely_bonding_spx_correct_overlap_map', 
    #                                 site_idx=site_idx_all[case],
    #                                 write_cube_files=False)
    # exit()                                                                                                

    # scale_R_array = [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0] #np.arange(0.5, 1.5, 0.05)
    # workflow_autocorrelation_term(parameters_model_all[case], 
    #                                 scale_R_array=scale_R_array, 
    #                                 output_folder=output_folders_all[case], 
    #                                 site_idx=site_idx_all[case],
    #                                 write_cube_files=True)
    # exit()

    if run_cases:
        for i in run_cases:
            site_idx = site_idx_all[i]
            site_radii = site_radii_all[i]
            output_folder = output_folders_all[i]
            replace_DFT_by_model = replace_DFT_by_model_all[i]
            parameters_model = parameters_model_all[i]
            fit_model_to_DFT = fit_model_to_DFT_all[i]
            workflow(site_idx=site_idx, site_radii=site_radii, output_folder=output_folder, replace_DFT_by_model=replace_DFT_by_model, parameters_model=parameters_model, fit_model_to_DFT=fit_model_to_DFT)

 