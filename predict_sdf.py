import pickle as pkl
from argparse import ArgumentParser

import numpy as np
from rdkit.Chem import PandasTools, AllChem as Chem

from scripts.cv_regressor import CVRegressor
from scripts.gen_clean_mono_dataset import run_oe_tautomers, cleaning, filtering, run_molvs_tautomers

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020-2023'
__license__ = 'MIT'
__version__ = '1.1.0'


def valid_file(path: str) -> str:
    with open(path):
        pass
    return path


parser = ArgumentParser()
parser.add_argument('sdf', metavar='INPUT_SDF', type=valid_file)
parser.add_argument('out', metavar='OUTPUT_PATH')
parser.add_argument('--no-openeye', '-noe', action='store_true')
args = parser.parse_args()

print('Loading SDF...')
df = PandasTools.LoadSDF(args.sdf)
try:
    df.set_index('ID', inplace=True, verify_integrity=True)
except ValueError:
    print('Warning: Molblock names are not unique (or missing), adding an unique index')
    df.ID = list(range(len(df)))
    df.ID = df.ID.astype(str)
    df.set_index('ID', inplace=True)
    for ix, mol in df.ROMol.items():
        mol.SetProp('_Name', ix)
print(f'{len(df)} molecules loaded')

print('Loading model...')
with open('RF_CV_FMorgan3_pKa.pkl', 'rb') as f:
    model = pkl.load(f)

print('Start preparing dataset...')
df = cleaning(df, list(df.columns[df.columns != 'ROMol']))
print(f'After cleaning: {len(df)}')

df = filtering(df)
print(f'After filtering: {len(df)}')

if not args.no_openeye:
    print('Using OpenEye QuacPac for tautomer and charge standardization...')
    df = run_oe_tautomers(df)
    print(f'After QuacPac tautomers: {len(df)}')
else:
    print('Using RDKit MolVS for tautomer and charge standardization...')
    df = run_molvs_tautomers(df)
    print(f'After MolVS: {len(df)}')

print('Calculating fingerprints...')
fmorgan3 = []
for mol in df.ROMol:
    fmorgan3.append(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096, useFeatures=True))
fmorgan3 = np.array(fmorgan3)

print('Predicting...')
df['pKa_prediction'] = model.predict(fmorgan3)

print('Writing result file...')
PandasTools.WriteSDF(df, args.out, properties=df.columns, idName='RowID')
