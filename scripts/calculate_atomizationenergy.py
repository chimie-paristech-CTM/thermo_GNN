import argparse
import csv
import os
import random
import logging
from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoleculeProcessor:
    def __init__(self, accuracy_level, read_method):
        self.accuracy_level = accuracy_level
        self.read_method = read_method
        self.single_atom_energy = self.set_single_atom_energy()

    def set_single_atom_energy(self):
        energy_levels = {
            "Enthalpy_B3LYP/6-31G(2df,p)": {"C": -37.844411, "H": -0.497912, "O": -75.062219, "N": -54.581501, "F": -99.716370},
            "Enthalpy_M06-2X/def2-SVP": {"C": -37.8425055992, "H": -0.498138511711, "O": -75.0661577121, "N": -54.5871121790},
            # Add additional levels as needed
        }
        return energy_levels.get(self.accuracy_level, {})

    def count_elements_in_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            element_counts = defaultdict(int)
            for atom in mol.GetAtoms():
                element_counts[atom.GetSymbol()] += 1
            return dict(element_counts)
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return {}

    def calculate_molecule_energy(self, smiles, energy):
        element_count = self.count_elements_in_smiles(smiles)
        if not element_count:
            return None
        total_energy = sum(count * self.single_atom_energy.get(element, 0) for element, count in element_count.items())
        atomic_energy = (energy - total_energy) * 627.509
        return {
            'smiles': smiles,
            'energy': energy,
            'element_count': element_count,
            'total_energy': total_energy,
            'atomic_energy': atomic_energy
        }

    def process_csv(self, filename):
        molecules = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                smiles, energy = row[0], float(row[1])
                molecule_data = self.calculate_molecule_energy(smiles, energy)
                if molecule_data:
                    molecules.append(molecule_data)
        return molecules

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")

def main(args):
    processor = MoleculeProcessor(args.accuracy_level, args.read_method)
    molecules = processor.process_csv(args.file_path)
    output_filename = os.path.splitext(args.file_path)[0] + '_processed.csv'
    processor.save_to_csv(molecules, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy_level', default="Enthalpy_B3LYP/6-31G(2df,p)", help="Specify the accuracy level for calculations.")
    parser.add_argument('--read_method', default="rdkit", help="Method to read and process SMILES.")
    parser.add_argument('--file_path', default="./PC9_data/pc9.csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args)
