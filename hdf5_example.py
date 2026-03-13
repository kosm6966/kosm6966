#!/usr/bin/env python
#
# Author: Kori Smyser <kori.smyser@gmail.com>
#

'''
HDF5 file exploration, extraction, and I/O using HDF5Analyzer and
HDF5Container.

This example covers:
  1. Building a sample HDF5 file that mimics a PySCF quantum-chemistry output
  2. Exploratory analysis  — analyze(), summary(), list_datasets(),
     list_groups(), get_dataset_info()
  3. Data extraction       — extract_dataset() with optional slicing,
     extract_attribute(), extract_group_attributes(), HDF5Container
  4. File I/O              — save_dataset_to_file() in .npy and .csv formats
'''

import os
import shutil
import tempfile

import h5py
import numpy as np

from hdf5_utils import HDF5Analyzer, HDF5Container


def _make_sample_hdf5(filepath):
    '''Write a representative HDF5 file used throughout this example.'''
    rng = np.random.default_rng(42)
    with h5py.File(filepath, 'w') as f:
        f.attrs['program'] = 'PySCF'
        f.attrs['version'] = '2.4.0'
        f.attrs['description'] = 'Multigrid ISDF demonstration'

        geo = f.create_group('geometry')
        geo.attrs['units'] = 'angstrom'
        geo.attrs['n_atoms'] = 4
        geo.create_dataset('atomic_numbers', data=np.array([6, 6, 1, 1], dtype=np.int32))
        geo.create_dataset('coordinates', data=rng.standard_normal((4, 3)).astype(np.float64))

        basis = f.create_group('basis')
        basis.attrs['name'] = 'cc-pVTZ'
        basis.attrs['n_basis'] = 120
        coeffs = basis.create_group('coefficients')
        coeffs.attrs['ordering'] = 'spherical'
        coeffs.create_dataset('alpha',
                              data=rng.standard_normal(120).astype(np.float64),
                              compression='gzip', chunks=(60,))
        coeffs.create_dataset('contraction',
                              data=rng.standard_normal((120, 120)).astype(np.float64),
                              compression='gzip', chunks=(30, 30))

        res = f.create_group('results')
        res.attrs['method'] = 'RKS'
        res.attrs['converged'] = True
        ene = res.create_dataset('orbital_energies',
                                 data=np.sort(rng.standard_normal(60)).astype(np.float64))
        ene.attrs['units'] = 'Hartree'
        ene.attrs['n_occ'] = 20
        mo = res.create_dataset('mo_coefficients',
                                data=rng.standard_normal((120, 60)).astype(np.float64))
        mo.attrs['shape_note'] = 'n_basis x n_mo'
        res.create_dataset('density_matrix',
                           data=rng.standard_normal((120, 120)).astype(np.float64))


tmpdir = tempfile.mkdtemp()
h5_file = os.path.join(tmpdir, 'pyscf_isdf.h5')
_make_sample_hdf5(h5_file)

#
# Part 1: Exploratory analysis
#
analyzer = HDF5Analyzer()
file_info = analyzer.analyze(h5_file)

print(file_info.summary())
print('Total datasets : %d' % file_info.num_datasets)
print('Total groups   : %d' % file_info.num_groups)

print('\nDataset paths:')
for path in file_info.list_datasets():
    print('  ' + path)

print('\nGroup paths:')
for path in file_info.list_groups():
    print('  ' + path)

print(analyzer.get_dataset_info('results/orbital_energies'))

print('\nRoot attributes:')
for key, val in file_info.root_attributes.items():
    print('  %s: %s' % (key, val))

#
# Part 2: Data extraction
#
orbital_energies = analyzer.extract_dataset('results/orbital_energies')
print('\nOrbital energies (%d values):\n  %s' % (orbital_energies.size, orbital_energies))

mo_slice = analyzer.extract_dataset('results/mo_coefficients',
                                    slicing=(slice(None), slice(0, 5)))
print('\nMO coefficients, first 5 columns — shape: %s' % str(mo_slice.shape))

units = analyzer.extract_attribute('results/orbital_energies', 'units')
n_occ = analyzer.extract_attribute('results/orbital_energies', 'n_occ')
print('\nOrbital energy units   : %s' % units)
print('Number of occupied MOs : %d' % n_occ)

basis_attrs = analyzer.extract_group_attributes('basis')
print('\nBasis group attributes:')
for key, val in basis_attrs.items():
    print('  %s: %s' % (key, val))

# HDF5Container provides dot-notation traversal of groups and datasets
container = HDF5Container(h5_file)
coords = container.geometry.coordinates
print('\nAtomic coordinates via HDF5Container — shape: %s\n  %s' % (coords.shape, coords))

alpha_coeffs = container.basis.coefficients.alpha
print('\nAlpha coefficients (first 10): %s' % alpha_coeffs[:10])

#
# Part 3: File I/O
#
npy_out = os.path.join(tmpdir, 'orbital_energies.npy')
analyzer.save_dataset_to_file('results/orbital_energies', npy_out, format='npy')
reloaded = np.load(npy_out)
print('\nReloaded from .npy — shape: %s, matches original: %s'
      % (reloaded.shape, np.allclose(reloaded, orbital_energies)))

csv_out = os.path.join(tmpdir, 'mo_coefficients.csv')
analyzer.save_dataset_to_file('results/mo_coefficients', csv_out, format='csv')
reloaded_csv = np.loadtxt(csv_out, delimiter=',')
mo_full = analyzer.extract_dataset('results/mo_coefficients')
print('Reloaded from .csv  — shape: %s, matches original: %s'
      % (reloaded_csv.shape, np.allclose(reloaded_csv, mo_full)))

# Save only the occupied-orbital subset
sliced_npy = os.path.join(tmpdir, 'orbital_energies_occ.npy')
analyzer.save_dataset_to_file('results/orbital_energies', sliced_npy,
                              slicing=(slice(0, n_occ),), format='npy')
occ_energies = np.load(sliced_npy)
print('\nOccupied orbital energies — shape: %s\n  %s' % (occ_energies.shape, occ_energies))

shutil.rmtree(tmpdir)
