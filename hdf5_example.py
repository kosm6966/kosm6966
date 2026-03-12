#!/usr/bin/env python
# Author: Kori Smyser <kori.smyser@gmail.com>
#
# Example: HDF5 file exploration, extraction, and I/O using HDF5Analyzer
# and HDF5Container.
#
# This script demonstrates:
#   1. Building a sample HDF5 file with groups, datasets, and attributes
#   2. Exploratory functionality  — HDF5Analyzer.analyze(), summary(),
#      list_datasets(), list_groups(), get_dataset_info()
#   3. Extraction functionality   — extract_dataset() with optional slicing,
#      extract_attribute(), extract_group_attributes(), HDF5Container
#   4. I/O functionality          — save_dataset_to_file() in .npy and .csv
#

import os
import tempfile

import h5py
import numpy as np

from hdf5_utils import HDF5Analyzer, HDF5Container

# ---------------------------------------------------------------------------
# Helper: build a temporary HDF5 file that mimics a quantum-chemistry output
# ---------------------------------------------------------------------------

def _make_sample_hdf5(filepath: str) -> None:
    """Write a representative HDF5 file used throughout this example."""
    rng = np.random.default_rng(42)

    with h5py.File(filepath, 'w') as f:
        # Root-level metadata
        f.attrs['program'] = 'PySCF'
        f.attrs['version'] = '2.4.0'
        f.attrs['description'] = 'Multigrid ISDF demonstration'

        # ------------------------------------------------------------------
        # Group: geometry
        # ------------------------------------------------------------------
        geo = f.create_group('geometry')
        geo.attrs['units'] = 'angstrom'
        geo.attrs['n_atoms'] = 4

        geo.create_dataset('atomic_numbers', data=np.array([6, 6, 1, 1], dtype=np.int32))
        coords = rng.standard_normal((4, 3)).astype(np.float64)
        geo.create_dataset('coordinates', data=coords)

        # ------------------------------------------------------------------
        # Group: basis / subgroup: coefficients
        # ------------------------------------------------------------------
        basis = f.create_group('basis')
        basis.attrs['name'] = 'cc-pVTZ'
        basis.attrs['n_basis'] = 120

        coeffs = basis.create_group('coefficients')
        coeffs.attrs['ordering'] = 'spherical'
        coeffs.create_dataset(
            'alpha',
            data=rng.standard_normal(120).astype(np.float64),
            compression='gzip',
            chunks=(60,),
        )
        coeffs.create_dataset(
            'contraction',
            data=rng.standard_normal((120, 120)).astype(np.float64),
            compression='gzip',
            chunks=(30, 30),
        )

        # ------------------------------------------------------------------
        # Group: results
        # ------------------------------------------------------------------
        res = f.create_group('results')
        res.attrs['method'] = 'RKS'
        res.attrs['converged'] = True

        energies = res.create_dataset(
            'orbital_energies',
            data=np.sort(rng.standard_normal(60)).astype(np.float64),
        )
        energies.attrs['units'] = 'Hartree'
        energies.attrs['n_occ'] = 20

        mo = res.create_dataset(
            'mo_coefficients',
            data=rng.standard_normal((120, 60)).astype(np.float64),
        )
        mo.attrs['shape_note'] = 'n_basis x n_mo'

        res.create_dataset(
            'density_matrix',
            data=rng.standard_normal((120, 120)).astype(np.float64),
        )


# ---------------------------------------------------------------------------
# 1. Build the sample file
# ---------------------------------------------------------------------------

tmpdir = tempfile.mkdtemp()
h5_file = os.path.join(tmpdir, 'pyscf_isdf.h5')
_make_sample_hdf5(h5_file)
print(f'Sample HDF5 file written to: {h5_file}\n')


# ===========================================================================
# SECTION 1 — EXPLORATORY FUNCTIONALITY
# ===========================================================================
print('=' * 60)
print('SECTION 1: Exploratory Functionality')
print('=' * 60)

analyzer = HDF5Analyzer()
file_info = analyzer.analyze(h5_file)

# --- Full structured summary --------------------------------------------------
print(file_info.summary())

# --- Quick counts -------------------------------------------------------------
print(f'Total datasets : {file_info.num_datasets}')
print(f'Total groups   : {file_info.num_groups}')
print()

# --- List all dataset and group paths -----------------------------------------
print('Dataset paths:')
for path in file_info.list_datasets():
    print(f'  {path}')

print('\nGroup paths:')
for path in file_info.list_groups():
    print(f'  {path}')
print()

# --- Detailed info for a single dataset ---------------------------------------
print(analyzer.get_dataset_info('results/orbital_energies'))
print()

# --- Root-level attributes ----------------------------------------------------
print('Root attributes:')
for name, value in file_info.root_attributes.items():
    print(f'  {name}: {value}')
print()


# ===========================================================================
# SECTION 2 — EXTRACTION FUNCTIONALITY
# ===========================================================================
print('=' * 60)
print('SECTION 2: Extraction Functionality')
print('=' * 60)

# --- Extract entire dataset ---------------------------------------------------
orbital_energies = analyzer.extract_dataset('results/orbital_energies')
print(f'Orbital energies (all {orbital_energies.size} values):')
print(f'  {orbital_energies}')
print()

# --- Extract a slice of a 2-D dataset (first 5 MOs, all basis functions) ------
mo_slice = analyzer.extract_dataset('results/mo_coefficients', slicing=(slice(None), slice(0, 5)))
print(f'MO coefficients, first 5 columns — shape: {mo_slice.shape}')
print(f'  {mo_slice[:3, :]}  ...')
print()

# --- Extract a single attribute from a dataset --------------------------------
units = analyzer.extract_attribute('results/orbital_energies', 'units')
n_occ = analyzer.extract_attribute('results/orbital_energies', 'n_occ')
print(f'Orbital energy units   : {units}')
print(f'Number of occupied MOs : {n_occ}')
print()

# --- Extract all attributes from a group --------------------------------------
basis_attrs = analyzer.extract_group_attributes('basis')
print('Basis group attributes:')
for name, value in basis_attrs.items():
    print(f'  {name}: {value}')
print()

# --- Use HDF5Container for attribute-style access to the whole file -----------
# HDF5Container maps every group / dataset to a Python attribute, enabling
# dot-notation traversal without explicit path strings.
container = HDF5Container(h5_file)

# Access nested data via attribute chaining
coords = container.geometry.coordinates
print(f'Atomic coordinates via HDF5Container — shape: {coords.shape}')
print(f'  {coords}')
print()

# Access a compressed, chunked dataset
alpha_coeffs = container.basis.coefficients.alpha
print(f'Alpha coefficients (first 10): {alpha_coeffs[:10]}')
print()


# ===========================================================================
# SECTION 3 — I/O FUNCTIONALITY
# ===========================================================================
print('=' * 60)
print('SECTION 3: I/O Functionality')
print('=' * 60)

# --- Save a dataset as a NumPy binary (.npy) file ----------------------------
npy_out = os.path.join(tmpdir, 'orbital_energies.npy')
analyzer.save_dataset_to_file('results/orbital_energies', npy_out, format='npy')

reloaded = np.load(npy_out)
print(f'Reloaded from .npy — shape: {reloaded.shape}, matches original: {np.allclose(reloaded, orbital_energies)}')
print()

# --- Save a 2-D dataset as a CSV file ----------------------------------------
csv_out = os.path.join(tmpdir, 'mo_coefficients.csv')
analyzer.save_dataset_to_file('results/mo_coefficients', csv_out, format='csv')

reloaded_csv = np.loadtxt(csv_out, delimiter=',')
mo_full = analyzer.extract_dataset('results/mo_coefficients')
print(f'Reloaded from .csv  — shape: {reloaded_csv.shape}, matches original: {np.allclose(reloaded_csv, mo_full)}')
print()

# --- Save a sliced dataset ---------------------------------------------------
sliced_npy_out = os.path.join(tmpdir, 'orbital_energies_occ.npy')
analyzer.save_dataset_to_file(
    'results/orbital_energies',
    sliced_npy_out,
    slicing=(slice(0, n_occ),),
    format='npy',
)
occ_energies = np.load(sliced_npy_out)
print(f'Saved occupied-only energies — shape: {occ_energies.shape}')
print(f'  {occ_energies}')
print()

# --- Clean up temporary files ------------------------------------------------
import shutil
shutil.rmtree(tmpdir)
print('Temporary files cleaned up.')
