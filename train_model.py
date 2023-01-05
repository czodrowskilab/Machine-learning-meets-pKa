import pickle as pkl
import pickletools as pkltools
from argparse import ArgumentParser

import numpy as np
from rdkit.Chem import PandasTools, AllChem as Chem
from sklearn.ensemble import RandomForestRegressor

from scripts.cv_regressor import CVRegressor

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020-2023'
__license__ = 'MIT'
__version__ = '1.1.0'

SEED = 24

parser = ArgumentParser()
parser.add_argument('--no-openeye', '-noe', action='store_true')
parser.add_argument('--num-processes', '-np', type=int, default=12)
args = parser.parse_args()

sdf_path = 'datasets/combined_training_datasets_unique.sdf'
if args.no_openeye:
    sdf_path = 'datasets/combined_training_datasets_unique_no_oe.sdf'

print('Loading training dataset...')
all_df = PandasTools.LoadSDF(sdf_path)

print('Calculating fingerprints...')
fmorgan3 = []
for mol in all_df.ROMol:
    fmorgan3.append(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096, useFeatures=True))
fmorgan3 = np.array(fmorgan3)

np = max(1, args.num_processes)
print(f'Training 5-fold CV Random Forest on {np} cores (this may take some time)...')
cvr = CVRegressor(est=RandomForestRegressor, params=dict(n_estimators=1000, n_jobs=np, random_state=SEED),
                  n_folds=5, shuffle=True)
cvr.fit(fmorgan3, all_df.pKa.astype(float, copy=False), random_state=SEED)

print('Saving model to file...')
with open('RF_CV_FMorgan3_pKa.pkl', 'wb') as f:
    f.write(pkltools.optimize(pkl.dumps(cvr, protocol=pkl.HIGHEST_PROTOCOL)))
