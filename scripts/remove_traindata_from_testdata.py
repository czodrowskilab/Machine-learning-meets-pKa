"""
This script removes molecules from the specified test data set
if there are also contained in the specified training data set.
The molecules are compared by isomeric smiles.
"""

from argparse import ArgumentParser

from rdkit.Chem import PandasTools, SDMolSupplier, MolToSmiles

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.0.0'


def valid_file(path: str) -> str:
    with open(path):
        pass
    return path


parser = ArgumentParser()
parser.add_argument('train', metavar='TRAIN_SDF', type=valid_file)
parser.add_argument('test', metavar='TEST_SDF', type=valid_file, nargs='+')
args = parser.parse_args()

train_smi = [MolToSmiles(mol, isomericSmiles=True) for mol in SDMolSupplier(args.train)]

for tf in args.test:
    df = PandasTools.LoadSDF(tf, smilesName='ISO_SMI').set_index('ID', verify_integrity=True)
    drop_ix = []
    for i, smi in df.ISO_SMI.iteritems():
        if smi in train_smi:
            drop_ix.append(i)
    df.drop(index=drop_ix, inplace=True)
    PandasTools.WriteSDF(df, tf.split('/')[-1].replace('.sdf', '_notraindata.sdf'), properties=df.columns)
    print(f'Dropped {len(drop_ix)} for "{tf}"')
