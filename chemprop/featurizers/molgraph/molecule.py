from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from torch import Tensor

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import Featurizer, GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.utils.utils import is_cuikmolmaker_available
from collections import Counter

if is_cuikmolmaker_available():
    import cuik_molmaker


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

            E[i : i + 2] = x_e

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        # Create MolGraph with optional mol features (defaults to None)
        return MolGraph(V, E, edge_index, rev_edge_index, None, None)


@dataclass
class ThermoGNNMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    include_mol_features: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.atom_fdim = len(self.atom_featurizer) + 9
        self.bond_fdim = len(self.bond_featurizer) + 8

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
            E = np.empty((0, self.bond_fdim), dtype=np.single)
            edge_index = np.array([[], []], int)
            rev_edge_index = np.array([], int)
            if self.include_mol_features:
                f_mol = np.zeros(8, dtype=np.float32)
                f_mol_atom = [f_mol]
            else:
                f_mol = None
                f_mol_atom = None
            return MolGraph(V, E, edge_index, rev_edge_index, f_mol, f_mol_atom)

        V_base = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E_base = []
        edge_index = [[], []]

        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            E_base.append(x_e)
            E_base.append(x_e)
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

        if len(E_base) == 0:
            E_base = np.empty((0, len(self.bond_featurizer)), dtype=np.single)
        else:
            E_base = np.array(E_base, dtype=np.single)

        V_ring = np.zeros((n_atoms, 9), dtype=np.float32)
        E_ring = np.zeros((n_bonds, 8), dtype=np.float32)

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        bond_rings = ring_info.BondRings()

        atom_ring_counts = [Counter() for _ in range(n_atoms)]
        for ring in atom_rings:
            ring_size = len(ring)
            for atom_idx in ring:
                atom_ring_counts[atom_idx][ring_size] += 1

        size_range = list(range(3, 11))
        for atom_idx, ring_counts in enumerate(atom_ring_counts):
            for ring_size, count in ring_counts.items():
                if ring_size in size_range:
                    feature_idx = size_range.index(ring_size)
                    V_ring[atom_idx, feature_idx] = count
            V_ring[atom_idx, 8] = 1.0 if len(ring_counts) > 0 else 0.0

        bond_ring_counts = [Counter() for _ in range(n_bonds)]
        for ring in bond_rings:
            ring_size = len(ring)
            for bond_idx in ring:
                bond_ring_counts[bond_idx][ring_size] += 1

        for bond_idx, ring_counts in enumerate(bond_ring_counts):
            for ring_size, count in ring_counts.items():
                if ring_size in size_range:
                    feature_idx = size_range.index(ring_size)
                    E_ring[bond_idx, feature_idx] = count

        V_combined = np.hstack((V_base, V_ring))
        E_ring_expanded = np.repeat(E_ring, 2, axis=0)
        E_combined = np.hstack((E_base, E_ring_expanded))

        if self.include_mol_features:
            formal_charge = Chem.GetFormalCharge(mol)
            charge_encoding = [0] * 5
            if -2 <= formal_charge <= 2:
                charge_encoding[formal_charge + 2] = 1
            else:
                charge_encoding = [-1] * 5

            unpaired_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            spin_multiplicity = int(2 * (unpaired_electrons / 2) + 1)
            spin_encoding = [0] * 3
            if spin_multiplicity in [1, 2, 3]:
                spin_encoding[spin_multiplicity - 1] = 1

            f_mol = np.array(charge_encoding + spin_encoding, dtype=np.float32)
            f_mol_atom = [f_mol] * n_atoms
        else:
            f_mol = None
            f_mol_atom = None

        rev_edge_index = np.arange(len(E_combined)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V_combined, E_combined, edge_index, rev_edge_index, f_mol, f_mol_atom)


@dataclass(repr=False, eq=False, slots=True)
class BatchCuikMolGraph:
    V: Tensor
    """the atom feature matrix"""
    E: Tensor
    """the bond feature matrix"""
    edge_index: Tensor
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self):
        self.__size = self.batch[-1].item() + 1

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


@dataclass
class CuikmolmakerMolGraphFeaturizer(Featurizer[list[str], BatchCuikMolGraph]):
    """A :class:`CuikmolmakerMolGraphFeaturizer` featurizes a list of molecules at once instead of
    one molecule at a time for efficiency.

    Parameters
    ----------
    atom_featurizer_mode: str, default="V2"
        The mode of the atom featurizer (V1, V2, ORGANIC, RIGR) to use.
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    add_h: bool, default=False
        whether to add hydrogens to the `Chem.Mol` objects created from the input SMILES strings
    """

    atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2"
    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0
    add_h: bool = False

    atom_fdim: int = field(init=False)
    bond_fdim: int = field(init=False)

    def __post_init__(self):
        if not is_cuikmolmaker_available():
            raise ImportError(
                "CuikmolmakerMolGraphFeaturizer requires cuik-molmaker package to be installed. "
                f"Please install it using `python {Path(__file__).parents[2] / Path('scripts/check_and_install_cuik_molmaker.py')}`."
            )
        atom_props_float = ["aromatic", "mass"]
        bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
        self.bond_fdim = 14

        self.atom_featurizer_mode = self.atom_featurizer_mode.upper()
        if self.atom_featurizer_mode == "V1":
            atom_props_onehot = [
                "atomic-number",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization",
            ]
            self.atom_fdim = 133
        elif self.atom_featurizer_mode == "V2":
            atom_props_onehot = [
                "atomic-number-common",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-expanded",
            ]
            self.atom_fdim = 72
        elif self.atom_featurizer_mode == "ORGANIC":
            atom_props_onehot = [
                "atomic-number-organic",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-organic",
            ]
            self.atom_fdim = 44
        elif self.atom_featurizer_mode == "RIGR":
            atom_props_onehot = ["atomic-number-common", "total-degree", "num-hydrogens"]
            atom_props_float = ["mass"]
            bond_props = ["is-null", "in-ring"]
            self.atom_fdim = 52
            self.bond_fdim = 2
        else:
            raise ValueError(f"Invalid atom featurizer mode: {self.atom_featurizer_mode}")

        self.atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_tensor(
            atom_props_onehot
        )

        self.atom_property_list_float = cuik_molmaker.atom_float_feature_names_to_tensor(
            atom_props_float
        )

        self.bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(bond_props)

        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        smiles_list: list[str],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> BatchCuikMolGraph:
        offset_carbon, duplicate_edges, add_self_loop = False, True, False

        (
            atom_feats,
            bond_feats,
            edge_index,
            rev_edge_index,
            batch,
        ) = cuik_molmaker.batch_mol_featurizer(
            smiles_list,
            self.atom_property_list_onehot,
            self.atom_property_list_float,
            self.bond_property_list,
            self.add_h,
            offset_carbon,
            duplicate_edges,
            add_self_loop,
        )

        if atom_features_extra is not None:
            atom_features_extra = torch.tensor(atom_features_extra, dtype=torch.float32)
            atom_feats = torch.cat((atom_feats, atom_features_extra), dim=1)
        if bond_features_extra is not None:
            bond_features_extra = np.repeat(bond_features_extra, repeats=2, axis=0)
            bond_features_extra = torch.tensor(bond_features_extra, dtype=torch.float32)
            bond_feats = torch.cat((bond_feats, bond_features_extra), dim=1)

        return BatchCuikMolGraph(
            V=atom_feats,
            E=bond_feats,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            batch=batch,
        )
