import pickle as pkl
import pickletools as pkltools

import numpy as np
from rdkit.Chem import PandasTools, AllChem as Chem
from sklearn.ensemble import RandomForestRegressor

from scripts.cv_regressor import CVRegressor

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.0.0'

SDF_PATH = 'datasets/combined_training_datasets_unique.sdf'
SEED = 24
EST_JOBS = 12

print('Loading training dataset...')
all_df = PandasTools.LoadSDF(SDF_PATH)

print('Calculating fingerprints...')
fmorgan3 = []
for mol in all_df.ROMol:
    fmorgan3.append(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096, useFeatures=True))
fmorgan3 = np.array(fmorgan3)

print(f'Training 5-fold CV Random Forest on {EST_JOBS} cores (this may take some time)...')
cvr = CVRegressor(est=RandomForestRegressor, params=dict(n_estimators=1000, n_jobs=EST_JOBS, random_state=SEED),
                  n_folds=5, shuffle=True)
cvr.fit(fmorgan3, all_df.pKa.astype(float, copy=False), random_state=SEED)

print('Saving model to file...')
with open('RF_CV_FMorgan3_pKa.pkl', 'wb') as f:
    f.write(pkltools.optimize(pkl.dumps(cvr, protocol=pkl.HIGHEST_PROTOCOL)))
