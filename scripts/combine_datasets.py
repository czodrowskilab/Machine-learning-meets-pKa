"""
This script combines multiple SDF files specified via commandline arguments
to one SDF file. It will be saved as "combined_training_datasets.sdf". The
script tries to preserve the information about the original dataset name by
splitting the file name at "_" and saving the first element of this split as
SDF file tag named "original_dataset".
"""

from sys import argv

from rdkit.Chem import SDWriter, SDMolSupplier

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.0.0'

sdw = SDWriter('combined_training_datasets.sdf')
for f in argv[1:]:
    dsname = f.split('_')[0]
    sdm = SDMolSupplier(f)
    for mol in sdm:
        mol.SetProp('original_dataset', dsname)
        sdw.write(mol)
sdw.close()
