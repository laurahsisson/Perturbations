import rdkit
import rdkit.Chem.rdFingerprintGenerator
import pandas as pd

mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=2048)

# https://github.com/rdkit/rdkit/discussions/3863
def to_numpy(fp):
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')

def get_fingerprint(smiles):
    mol = rdkit.Chem.MolFromSmiles(d.smiles)
    fp = mfpgen.GetFingerprint(mol)
    return to_numpy(fp)

