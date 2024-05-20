import argparse
import csv
import os
import random
import logging
import pandas as pd
from rdkit import Chem, RDLogger
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hatree2kcal = 627.509
class MoleculeProcessor:
    def __init__(self, accuracy_level, read_method):
        self.accuracy_level = accuracy_level
        self.read_method = read_method
        self.single_atom_energy = self.set_single_atom_energy()

    def set_single_atom_energy(self):
        energy_levels = {
            "Enthalpy_ωB97X-D/def2-SVP": {"C": -37.795443, "H": -0.4995215, "O": -74.9743032,"N": -54.5200622,"Br": -2573.8545726, "P":-341.1371926,"I":-341.0718256, "F": -99.6120933,
                                          "S": -397.9695617,"Cl": -459.9861458, "B":-24.5012682},
            "free_energy_ωB97X-D/def2-SVP": {"C": -37.81265, "H": -0.5125361, "O": -74.9916159,"N": -54.5374582, "Br": -2573.8737631, "P":-341.1557129,"I":-341.0896915,"F": -99.6292669,
                                           "S": -397.9878553, "Cl": -460.0041835, "B": -24.5298189},
            # Add additional levels as needed
        }
        return energy_levels.get(self.accuracy_level, {})

    def count_elements_in_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            element_counts = defaultdict(int)
            unrecognized_elements = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in self.single_atom_energy:
                    element_counts[symbol] += 1
                else:
                    unrecognized_elements.append(symbol)
            if unrecognized_elements:
                logger.warning(f"Unrecognized elements {set(unrecognized_elements)} in SMILES {smiles}")
                raise ValueError(f"Unrecognized element '{set(unrecognized_elements)}' in SMILES {smiles}")
            return dict(element_counts)

        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            raise ValueError(f"Error processing SMILES {smiles}: {e}")

    def calculate_atomization_energy(self, smiles, energy):
        element_count = self.count_elements_in_smiles(smiles)
        if not element_count:
            raise ValueError(f"Unrecognized element_count")
        total_energy = sum(count * self.single_atom_energy.get(element) for element, count in element_count.items())
        atomization_energy = (energy - total_energy * hatree2kcal)
        return {
            'smiles': smiles,
            'energy': energy,
            'element_count': element_count,
            'total_energy': total_energy,
            'atomization_energy': atomization_energy
        }

    def process_csv(self, filename):
        molecules = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                smiles, energy = row[0], float(row[1])
                molecule_data = self.calculate_atomization_energy(smiles, energy)
                if molecule_data:
                    molecules.append(molecule_data)
        return molecules

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")

    def split_dataset(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        random.shuffle(data)
        total_samples = len(data)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        return {
            'train': data[:train_size],
            'val': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }

def main(args):
    processor = MoleculeProcessor(args.accuracy_level, args.read_method)
    molecules = processor.process_csv(args.file_path)
    datasets = processor.split_dataset(molecules)
    for dataset_name, data in datasets.items():
        output_filename = os.path.splitext(args.file_path)[0] + f'_{dataset_name}.csv'
        processor.save_to_csv(data, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy_level', default="Enthalpy_ωB97X-D/def2-SVP", help="Specify the accuracy level for calculations.")
    parser.add_argument('--read_method', default="rdkit", help="Method to read and process SMILES.")
    parser.add_argument('--file_path', default="./lowest_enthalpy.csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args)
