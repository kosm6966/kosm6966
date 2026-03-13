#!/usr/bin/env python
#
# Author: Kori Smyser <kori.smyser@gmail.com>
#

'''
Ion-trap crystal composition using the Crystal class.

This example covers:
  1. Single-species crystals — nions, mass_vec, charge
  2. Mixed and isotope-tagged crystals — mass_vec array, average mass
  3. Charge-to-mass ratio and overriding the average mass with a fixed value
'''

import numpy as np
import scipy.constants as ct

from crystal import Crystal


#
# Part 1: Single-species crystals
#
# The crystal string uses one letter per ion: Y (Yb-171), B (Ba-138),
# S (Sr-88), or C (Ca-40).  ion_charge is the charge state in units of
# the elementary charge.
#

yb_chain = Crystal('YYY', ion_charge=1)
print('Yb-171 chain (3 ions, +1)')
print('  nions     : %d'   % yb_chain.nions)
print('  charge    : %.6e C' % yb_chain.charge)
print('  mass (avg): %.6e kg' % yb_chain.mass)
print('  mass_vec  : %s kg' % yb_chain.mass_vec)

ba138_chain = Crystal('BBB', ion_charge=1)
print('\nBa-138 chain (3 ions, +1)')
print('  nions     : %d'   % ba138_chain.nions)
print('  mass (avg): %.6e kg' % ba138_chain.mass)

ba137_single = Crystal('B137', ion_charge=2)
print('\nBa-137 single ion (+2)')
print('  nions     : %d'   % ba137_single.nions)
print('  charge    : %.6e C' % ba137_single.charge)
print('  mass (avg): %.6e kg' % ba137_single.mass)

sr_chain = Crystal('SSS', ion_charge=1)
print('\nSr-88 chain (3 ions, +1)')
print('  mass (avg): %.6e kg' % sr_chain.mass)

ca_chain = Crystal('CCC', ion_charge=1)
print('\nCa-40 chain (3 ions, +1)')
print('  mass (avg): %.6e kg' % ca_chain.mass)

#
# Part 2: Mixed and isotope-tagged crystals
#
# Append the mass number explicitly (e.g. 'B137') to select a specific
# isotope.  All ion tokens are parsed independently, so species can be
# freely interleaved.
#

mixed = Crystal('YBSBY', ion_charge=1)
print('\nMixed crystal YBSBY (Yb-Ba-Sr-Ba-Yb, +1)')
print('  nions     : %d'   % mixed.nions)
print('  mass_vec  : %s kg' % mixed.mass_vec)
print('  mass (avg): %.6e kg' % mixed.mass)

isotope_mixed = Crystal('YB137Y', ion_charge=1)
print('\nIsotope-tagged crystal YB137Y (Yb-Ba137-Yb, +1)')
print('  nions     : %d'   % isotope_mixed.nions)
print('  mass_vec  : %s kg' % isotope_mixed.mass_vec)
print('  mass (avg): %.6e kg' % isotope_mixed.mass)

# Explicit isotope tags and default shorthand give identical mass_vec
explicit = Crystal('Y171B138S88C40', ion_charge=1)
shorthand = Crystal('YBSC', ion_charge=1)
print('\nExplicit tags vs shorthand give same masses: %s'
      % np.allclose(explicit.mass_vec, shorthand.mass_vec))

#
# Part 3: Charge-to-mass ratio and overriding the average mass
#
# .ratio returns charge / mass, the key figure of merit for RF trap
# operation.  Passing mass= to the constructor replaces the per-species
# average with a fixed value (useful for lumped-mass models).
#

print('\nCharge-to-mass ratios (C/kg):')
for label, crys in [('Yb-171 +1', yb_chain),
                    ('Ba-138 +1', ba138_chain),
                    ('Ba-137 +2', ba137_single),
                    ('Sr-88  +1', sr_chain),
                    ('Ca-40  +1', ca_chain)]:
    print('  %-12s  %.6e' % (label, crys.ratio))

# Override mass for a lumped single-species model
lumped_mass = 170.0 * ct.atomic_mass
yb_lumped = Crystal('YYY', ion_charge=1, mass=lumped_mass)
print('\nYb-171 chain with lumped mass %.6e kg' % lumped_mass)
print('  mass (avg, lumped): %.6e kg' % yb_lumped.mass)
print('  ratio (lumped)    : %.6e C/kg' % yb_lumped.ratio)
print('  ratio (per-species): %.6e C/kg' % yb_chain.ratio)
