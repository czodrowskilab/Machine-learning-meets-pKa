"""
This script prepares SDF files which can then be used for machine learning.
This includes sanitizing, filtering molecules with bad functional groups and
unwanted elements, removing salts, filtering by Lipinski's rule of five and
unify different tautomers.
"""

from argparse import ArgumentParser, Namespace
from io import StringIO
from subprocess import PIPE, Popen, DEVNULL, SubprocessError
from typing import Optional, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, SaltRemover, Descriptors, Lipinski, Crippen, Mol

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.0.0'

# Selenium, Silicon and Bor
BAD_ELEMENTS = ['Se', 'Si', 'B']
BAD_ELEM_QUERY = Chem.MolFromSmarts(f'[{",".join(BAD_ELEMENTS)}]')

BFG = [
    Chem.MolFromSmarts('[!#8][NX3+](=O)[O-]'),  # "Classical" nitro group
    Chem.MolFromSmarts('[$([NX3+]([O-])O),$([NX3+]([O-])[O-])]=[!#8]'),  # Nitro group in tautomer form
]

ADDITIONAL_SALTS = [
    Chem.MolFromSmarts('[H+]'),
    Chem.MolFromSmarts('[I,N][I,N]'),
    Chem.MolFromSmarts('[Cs+]'),
    Chem.MolFromSmarts('F[As,Sb,P](F)(F)(F)(F)F'),
    Chem.MolFromSmarts('[O-,OH][Cl+3]([O-,OH])([O-,OH])[O-,OH]')
]

PKA_LOWER_CUT = 2
PKA_UPPER_CUT = 12
MARVIN_LOG_CUT = 4

LIPINSKI_RULES = [
    (Descriptors.MolWt, 500),
    (Lipinski.NumHDonors, 5),
    (Lipinski.NumHAcceptors, 10),
    (Crippen.MolLogP, 5),
]


def count_bad_elements(mol: Mol) -> int:
    """
    Counts occurrences of bad elements
    specified in <BAD_ELEM_QUERY> for a molecule.

    Parameters
    ----------
    mol : Mol
        RDKit mol object

    Returns
    -------
    int
        Bad element count
    """

    return len(mol.GetSubstructMatches(BAD_ELEM_QUERY))


def count_bfg(mol: Mol) -> int:
    """
    Counts occurrences of bad functional groups
    specified in <BFG> for a molecule.

    Parameters
    ----------
    mol : Mol
        RDKit mol object

    Returns
    -------
    int
        Bad functional group count
    """

    n = 0
    for bfg in BFG:
        if mol.HasSubstructMatch(bfg):
            n += 1
    return n


ADDITIONAL_FILTER_RULES = [
    (count_bad_elements, 0),  # Are there any bad elements (more than zero)
    (count_bfg, 0),  # Are there any bad functional groups (more than zero)
]


def parse_args() -> Namespace:
    """
    Parses commandline parameters

    Returns
    -------
    Namespace
        Argparse Namespace object containing parsed commandline
        parameters
    """

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='INFILE')
    parser.add_argument('outfile', metavar='OUTFILE')
    parser.add_argument('--keep-props', '-kp', metavar='PROP1,PROP2,...', default=[], type=lambda x: x.split(','))
    return parser.parse_args()


def check_on_remaining_salts(mol: Mol) -> Optional[Mol]:
    """
    Checks if any salts are remaining in the given molecule.

    Parameters
    ----------
    mol : Mol

    Returns
    -------
    Mol, optional
        Input molecule if no salts were found, None otherwise
    """

    if len(Chem.GetMolFrags(mol)) == 1:
        return mol
    return None


def check_sanitization(mol: Mol) -> Optional[Mol]:
    """
    Checks if molecule is sanitizable.

    Parameters
    ----------
    mol : Mol
        RDKit mol object

    Returns
    -------
    Mol, optional
        Sanitized molecule if possible, None otherwise
    """

    try:
        Chem.SanitizeMol(mol)
        return mol
    except ValueError:
        return None


def cleaning(df: pd.DataFrame, keep_props: List[str]) -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing unwanted columns,
    removing salts from all molecules and sanitize the molecules.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a ROMol column with RDKit molecules
        and all columns specified in "keep_props"
    keep_props : List[str]
        Property names that should be kept through this script

    Returns
    -------
    DataFrame
        Cleaned DataFrame
    """

    df = df.loc[:, ['ROMol'] + keep_props]

    salt_rm = SaltRemover.SaltRemover()
    salt_rm.salts.extend(ADDITIONAL_SALTS)
    df.ROMol = df.ROMol.apply(salt_rm.StripMol)
    df.dropna(subset=['ROMol'], inplace=True)

    df.ROMol = df.ROMol.apply(check_on_remaining_salts)
    df.dropna(subset=['ROMol'], inplace=True)

    df.ROMol = df.ROMol.apply(check_sanitization)
    df.dropna(subset=['ROMol'], inplace=True)

    return df


def filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters DataFrame rows by molecules contained in column
    "ROMol" by Lipinski's rule of five, bad functional groups
    and unwanted elements.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a ROMol column with RDKit molecules

    Returns
    -------
    DataFrame
        Filtered DataFrame
    """

    del_ix = []
    lip = 0
    for ix, row in df.iterrows():
        violations = 0
        for func, thres in LIPINSKI_RULES:
            if func(row.ROMol) > thres:
                violations += 1
            if violations > 1:
                del_ix.append(ix)
                lip += 1
                break
        if lip > 0 and del_ix[-1] == ix:
            continue
        for func, thres in ADDITIONAL_FILTER_RULES:
            if func(row.ROMol) > thres:
                del_ix.append(ix)
                break
    print(f'Dropped {lip} mols because of more than one Lipinski rule violation')
    print(f'Dropped {len(del_ix) - lip} mols through additional filtering')
    return df.drop(index=del_ix)


def mols_to_sdbuffer(df: pd.DataFrame, props: List[str] = None) -> StringIO:
    """
    Writes a DataFrame containing a ROMol column in SD format
    to a StringIO buffer.

    Parameters
    ----------
    df : DataFrame
        DataFrame that should be written to a buffer
    props : List[str]
        List of column names that should also be written
        to the buffer

    Returns
    -------
    StringIO
        StringIO buffer containing data in SD format
    """

    buffer = StringIO()
    PandasTools.WriteSDF(df, buffer, properties=props)
    return buffer


def run_external(args: List[str], df: pd.DataFrame) -> str:
    """
    Calls an external program via subprocess and writes the given
    DataFrame in SD format to stdin of the program. It returns
    the stdout of the external program.

    Parameters
    ----------
    args : List[str]
        List of arguments including the call of the desired program
        that can be directly passed to the subprocess Popen constructor
    df : DataFrame
        DataFrame that should be piped to the external program in SD format

    Returns
    -------
    str
        Stdout of the external program

    Raises
    ------
    SubprocessError
        If the called program exits with a non-zero exit code
    """

    with mols_to_sdbuffer(df.reset_index(), ['ID']) as buffer:
        p = Popen(args, text=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)
        stdout, _ = p.communicate(buffer.getvalue())
    if p.returncode != 0:
        raise SubprocessError(f'{args[0]} ended with non-zero exit code {p.returncode}')
    return stdout


def run_marvin_pka(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates pKa values for configured pH range with ChemAxon Marvin
    and returns only monoprotic structures.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing RDKit molecules

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only monoprotic structures
    """

    cmd_call = ['cxcalc', '--id', 'ID', 'pka', '-i', str(PKA_LOWER_CUT), '-x', str(PKA_UPPER_CUT), '-T', '298.15']
    res_df = pd.read_csv(StringIO(run_external(cmd_call, df)), sep='\t').set_index('ID', verify_integrity=True)
    res_df.index = res_df.index.astype(str)
    df = df.merge(res_df, right_index=True, left_index=True)
    for ix in df.index:
        try:
            if np.isnan(df.loc[ix, 'atoms']):
                continue
        except TypeError:
            pass
        ci = 0
        for col in ['apKa1', 'apKa2', 'bpKa1', 'bpKa2']:
            val = df.loc[ix, col]
            if np.isnan(val):
                continue
            if val < PKA_LOWER_CUT or val > PKA_UPPER_CUT:
                df.loc[ix, col] = np.nan
                atoms = df.loc[ix, 'atoms'].split(',')
                if len(atoms) == 1:
                    df.loc[ix, 'atoms'] = np.nan
                else:
                    del atoms[ci]
                    df.loc[ix, 'atoms'] = ','.join(atoms)
                    ci -= 1
            ci += 1
    df.atoms = pd.to_numeric(df.atoms, errors='coerce')
    df.dropna(subset=['atoms'], inplace=True)
    df.atoms = df.atoms.astype(int, copy=False).apply(lambda x: x - 1)
    df['pKa_type'] = df.apKa1.apply(lambda x: 'basic' if x is None else 'acidic')
    df.apKa1 = df[['apKa1', 'bpKa1']].sum(axis=1)
    df.drop(columns=['bpKa1', 'apKa2', 'bpKa2'], inplace=True)  # 'count'
    return df


def run_oe_tautomers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifies different tautomers with OpenEye QUACPAC/Tautomers.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing RDKit molecules

    Returns
    -------
    DataFrame
        DataFrame with tautomer canonized structures
    """

    cmd_call = ['tautomers', '-maxtoreturn', '1', '-in', '.sdf', '-warts', 'false']
    mols, ix = [], []
    for mol in Chem.SmilesMolSupplierFromText(run_external(cmd_call, df), titleLine=False):
        mols.append(mol)
        ix.append(mol.GetProp('_Name'))
    ixs = set(ix)
    if len(ix) != len(ixs):
        print('WARNING: Duplicates in tautomers result')
    dropped = df.index.difference(ixs)
    df.drop(index=dropped, inplace=True)
    df.ROMol = mols
    return df


def filter_strong_outlier_by_marvin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters strong outliers that differ more than <MARVIN_LOG_CUT>
    log units from the corresponding ChemAxon Marvin prediction.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the columns "apKa1" (marvin prediction)
        and "pKa" (experimental value)

    Returns
    -------
    DataFrame
        DataFrame without the strong outliers
    """

    del_ix = []
    for ix, row in df.iterrows():
        if abs(row.apKa1 - float(row.pKa)) > MARVIN_LOG_CUT:
            del_ix.append(ix)
    df.drop(index=del_ix, inplace=True)
    return df


def main(args: Namespace) -> None:
    """
    Main function of this script

    Parameters
    ----------
    args : Namespace
        Namespace object containing the parsed commandline arguments
    """

    df = PandasTools.LoadSDF(args.infile).set_index('ID', verify_integrity=True)
    print(f'Initial: {len(df)}')

    df = cleaning(df, args.keep_props)
    print(f'After cleaning: {len(df)}')

    df = filtering(df)
    print(f'After filtering: {len(df)}')

    df = run_oe_tautomers(df)
    print(f'After QuacPac tautomers: {len(df)}')

    df = run_marvin_pka(df)
    print(f'After Marvin pKa: {len(df)}')

    df = filter_strong_outlier_by_marvin(df)
    print(f'After removing strong outlier: {len(df)}')

    df.columns = ['ROMol'] + args.keep_props + ['marvin_pKa', 'marvin_atom', 'marvin_pKa_type']

    PandasTools.WriteSDF(df, args.outfile, idName='RowID', properties=df.columns)


if __name__ == '__main__':
    main(parse_args())
