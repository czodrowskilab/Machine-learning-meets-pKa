"""
This script combines the pKa values of duplicated structures.
"""

from sys import argv
from typing import Optional

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.0.0'

PKA_LOWER_CUT = 2
PKA_UPPER_CUT = 12


def mean_except_outliers(var: pd.Series, m: int = 2) -> Optional[float]:
    """Combines a Pandas Series of values. All values outside of the
    specified pKa range ("PKA_[LOWER|UPPER]_CUT") are left out. Also
    outlier that deviate more than "m" standard deviations from mean
    are filtered. Finally the arithmetic mean is calculated and returned.

    Parameters
    ----------
    var : pd.Series
        Pandas Series containing the values
    m : int, optional
        Values that deviate more than "m" standard deviations from
        mean are filtered out (default is 2)

    Returns
    -------
    float
        Arithmetic mean except outliers and outer range values
    """

    varo = var.values
    var = []
    for v in varo:
        if PKA_LOWER_CUT <= v <= PKA_UPPER_CUT:
            var.append(v)
    if len(var) == 1:
        return var[0]
    if len(var) == 0:
        return None
    std = np.std(var)
    mean = np.mean(var)
    res = np.mean([x for x in var if mean - m * std <= x <= mean + m * std])
    return res


df = PandasTools.LoadSDF(argv[1], isomericSmiles=True, smilesName='ISO_SMILES')
df.pKa = pd.to_numeric(df.pKa)
print(f'Initial: {len(df)}')

grp = df.groupby('ISO_SMILES')
df2 = grp.first()
df2.pKa = grp.pKa.agg(mean_except_outliers)
try:
    df2.original_dataset = grp.original_dataset.agg(lambda x: list(set(x)))
except AttributeError:
    pass
df2.dropna(subset=['pKa'], inplace=True)

print(f'Unique: {len(df2)}')
PandasTools.WriteSDF(df2, argv[1].replace('.sdf', '_unique.sdf'), properties=df2.columns)
