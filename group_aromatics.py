import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import os
import sys


def check_aromatic(atom):
    if atom.IsInRing():
        atomic_num = atom.GetAtomicNum()
        len_neighbors = len(atom.GetNeighbors())
        if atomic_num == 6 and len_neighbors == 3:  # any sp2 carbon can participate in an aromatic system
            return True  # should this take the step of checking if it's also in a ring?
        elif atomic_num == 7 and (1 < len_neighbors < 4):  # exclude sp3 hybridized N in a ring
            return True
    return False


def eval_dihedrals(mol, atom1, atom2):
    # for aromatic neighbors, evaluate if the dihedral angle is within tolerance that aromaticity isn't broken
    # this function assumes atom1 and atom2 are in valid aromatic systems; filtering should be applied before function call
    a1_idx = atom1.GetIdx()
    a2_idx = atom2.GetIdx()
    n1 = [a for a in atom1.GetNeighbors() if check_aromatic(a) and a.GetIdx() != a2_idx]
    n2 = [a for a in atom2.GetNeighbors() if check_aromatic(a) and a.GetIdx() != a1_idx]
    mol_conf = mol.GetConformers()[0]  # assume there is only one conformer
    dihedrals = []
    for nd1 in n1:
        for nd2 in n2:
                dihedrals.append(rdMolTransforms.GetDihedralDeg(mol_conf, nd1.GetIdx(), a1_idx, a2_idx, nd2.GetIdx()))

    abs_diff_dihedrals = [np.abs(np.abs(np.abs(d)-90) - 90) for d in dihedrals]  # abs(-180) - 90 = 90. abs(0) - 90 = -90
    planar_mean = np.mean(abs_diff_dihedrals)
    if planar_mean > 20:
        return False
    else:
        return True


def enumerate_aromatic_size(mol, atom):
    
    visited = [False]*len(mol.GetAtoms())
    queue = []
    aromatic_size = 0

    if check_aromatic(atom):
        queue.append(atom)
        aromatic_size += 1
    visited[atom.GetIdx()] = True

    while queue:
        s = queue.pop()
        for neighbor in s.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if visited[n_idx] == False:
                visited[n_idx] = True
                if check_aromatic(neighbor):  # only add to queue if neighbor is aromatic and not twisted out of plane
                    if eval_dihedrals(mol, s, neighbor):
                        queue.append(neighbor)
                        aromatic_size += 1

    return aromatic_size


def embedded_aromatic(atom):
    # check if atom is aromatic, and whether all of its neighbors are also aromatic
    # Thermometer scale: non-aromatic = 0, aromatic-embedded = 1, aromatic-edge = 2
    if check_aromatic(atom):
        n_arom = []
        for neighbor in atom.GetNeighbors():
            n_arom.append(check_aromatic(neighbor))
        assert len(n_arom) > 1  # an aromatic atom is in a ring, and must have at least 2 neighbors
        assert len(n_arom) < 4  # an aromatic atom can only have 3 neighbors
        if sum(n_arom) > 2:
            return 1  # embedded in aromatic system
        else:
            return 2  # edge of aromatic system, can therefore flex out of plane and is more reactive
    else:
        return 0


if __name__ == "__main__":

    #filename = "/home/nricke/work/klodaya/notebooks_klodaya/allfiles/catalystonly-molfiles/sf96x0_optsp_a0m2.mol"
    #filename = "/home/nricke/work/klodaya/notebooks_klodaya/allfiles/catalystonly-molfiles/sf245x0_optsp_a0m2.mol"
    #filename = "/home/nricke/work/klodaya/notebooks_klodaya/allfiles/catalystonly-molfiles/sf260x0_optsp_c1m2.mol"
    #filename = "/home/nricke/work/klodaya/notebooks_klodaya/allfiles/catalystonly-molfiles/sf209x0_optsp_c1m2.mol"
    #filename = "/home/nricke/work/klodaya/notebooks_klodaya/allfiles/catalystonly-molfiles/sf95x0_optsp_c1m2.mol"
    #filename = "/home/nricke/work/klodaya/notebooks_klodaya/allfiles/catalystonly-molfiles/sf244x0_optsp_a0m2.mol"
    #filename = sys.argv[1]
    #m1 = Chem.MolFromMolFile(filename, removeHs=False)

    indir = sys.argv[1]

    
    aromatic_extent, ring_edge, atom_num_list, catalyst = [], [], [], []
    for molfile in os.listdir(indir):
        m = Chem.MolFromMolFile(os.path.join(indir, molfile), removeHs=False)
        catalyst_from_filename = molfile.split("_")[0]
        if m != None:
            for atom in m.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol == "C":
                    aromatic_extent.append(enumerate_aromatic_size(m, atom))
                    ring_edge.append(embedded_aromatic(atom))
                    atom_num_list.append(atom.GetIdx())
                    catalyst.append(catalyst_from_filename)

    df = pd.DataFrame({"aromatic_extent": aromatic_extent, "ring_edge": ring_edge, "Atom Number": atom_num_list, 
        "Catalyst Name": catalyst})
    print(df)
