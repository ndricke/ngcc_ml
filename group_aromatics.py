import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import os
import sys


def check_aromatic(atom):
    """
    aromatic systems assumed to: be in a ring, be carbon (3 neighbors) or nitrogen (2 or 3 neighbors)
    RDKit is unreliable for this because it doesn't handle nitrogen well, and double bond assignments in aromatic
    systems can be incorrect
    """
    if atom.IsInRing():
        atomic_num = atom.GetAtomicNum()
        len_neighbors = len(atom.GetNeighbors())
        if atomic_num == 6 and len_neighbors == 3:  # any sp2 carbon can participate in an aromatic system
            return True  # should this take the step of checking if it's also in a ring?
        elif atomic_num == 7 and (1 < len_neighbors < 4):  # exclude sp3 hybridized N in a ring
            return True
    return False


def eval_dihedrals(mol, atom1, atom2):
    """
    for aromatic neighbors, evaluate if the dihedral angle is within tolerance that aromaticity isn't broken
    this function assumes atom1 and atom2 are in valid aromatic systems; filtering should be applied before function call
    """
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


def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle * 180/np.pi


def calc_plane_deviation(atom, coords):
    """
    aromatic systems assumed to: be in a ring, be carbon (3 neighbors) or nitrogen (2 or 3 neighbors)


    """
    n_coords = []
    center_coord = coords[atom.GetIdx()]
    for neighbor in atom.GetNeighbors():  # Get coords for all neighbors
        n_coords.append(coords[neighbor.GetIdx()])
        
    # With 2 neighbors, the deviation is considered 0 since it should always be possible to make a plane
    if len(n_coords) == 2:
        plane_deviation = 0.0  
    else:  # handle case with 3 atoms
        assert len(n_coords) == 3  # if there are not 2 neighbors, then there ought to be 3
        angles = []
        for i,j in [[0,1],[0,2],[1,2]]:  # Calculate angle for the 3 unique pairs 0-1, 0-2, 1-2
            angles.append(calc_angle(n_coords[i], center_coord, n_coords[j]))
        plane_deviation = 360.0 - sum(angles)
        
    return plane_deviation


def enumerate_aromatic_properties(mol, atom):
    """
    Use breadth-first search to find the contiguous aromatic group size the input atom is a part of
    This assumes that only carbon and nitrogen can be aromatic, and assigns aromaticity as having either  2 or 3 bonds
    This also tries to catch the case of biphenyl or similar molecules, where two distinct aromatic systems are connected
    but should not be grouped together (as they are not in plane)

    This is also a convenient function to include counts of heteroatoms, number of aromatic hydrogen in a ring system, and possibly
    account distance to embedded aromatic nitrogen as well
    The coordinates should also be available here, so this could be an opportunity to add the other coordinate aromatic features
    """
    visited = [False]*len(mol.GetAtoms())
    queue = []
    aromatic_size = 0
    ring_nitrogens = 0
    atom_plane_deviation = 0
    ring_plane_deviation_list = []
    ring_plane_deviation = 0
    conformer = mol.GetConformers()[0]
    coords = conformer.GetPositions()

    if check_aromatic(atom):
        queue.append(atom)
        aromatic_size += 1
        atom_plane_deviation = calc_plane_deviation(atom, coords)
        ring_plane_deviation_list.append(atom_plane_deviation)
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
                        ring_plane_deviation_list.append(calc_plane_deviation(neighbor, coords))
                        if neighbor.GetAtomicNum() == 7:  # only after being sure it's in the same ring should we add N to count
                            ring_nitrogens += 1
    
    if len(ring_plane_deviation_list) > 0:
        ring_plane_deviation = np.mean(ring_plane_deviation_list)
    else:
        ring_plane_deviation = 0
    return aromatic_size, ring_nitrogens, atom_plane_deviation, ring_plane_deviation


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

    indir = sys.argv[1]
    outfile = sys.argv[2]
    
    aromatic_extent, ring_edge, atom_num_list, catalyst = [], [], [], []
    ring_nitrogen_list, atom_plane_deviation_list, ring_plane_deviation_list = [], [], []
    charge_list = []
    for molfile in os.listdir(indir):
        m = Chem.MolFromMolFile(os.path.join(indir, molfile), removeHs=False)
        catalyst_from_filename = molfile.split("_")[0]
        if m != None:
            for atom in m.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol == "C":
                    chmult = molfile.split(".")[0].split("_")[-1]
                    print(chmult)
                    charge_list.append(int(chmult[1]))
                    aromex, rnit, apd, rpd = enumerate_aromatic_properties(m, atom)
                    aromatic_extent.append(aromex)
                    ring_nitrogen_list.append(rnit)
                    atom_plane_deviation_list.append(apd)
                    ring_plane_deviation_list.append(rpd)
                    ring_edge.append(embedded_aromatic(atom))
                    atom_num_list.append(atom.GetIdx())
                    catalyst.append(catalyst_from_filename)

    df = pd.DataFrame({"aromatic_extent": aromatic_extent, "ring_edge": ring_edge, "Atom Number": atom_num_list, 
        "ring_nitrogens": ring_nitrogen_list, "atom_plane_deviation": atom_plane_deviation_list, "Catalyst Name": catalyst,
        "ring_plane_deviation": ring_plane_deviation_list, "charge": charge_list})
    print(df)
    df.to_json(outfile)
