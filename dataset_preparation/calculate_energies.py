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
            ###PC9
            "Atomization_Enthalpy_B3LYP/6-31G(d)": {"C": -37.8439195, "H": -0.4979123, "O": -75.058261,"Br": -2571.6545717,"P": -341.2557294, "F": -99.713175, "N": -54.5821289, "S": -398.1026323,
                                                  "Cl": -460.1338818, },
            "Formation_Enthalpy_B3LYP/6-31G(d)": {"C": -38.0962914, "H": -0.5810156, "O": -75.1564768, "Br":  -2571.6973513,
                                             "P": -341.3309732, "F": -99.746247, "N": -54.757613, "S":  -398.18514 , "Cl": -460.1725895,},
            ###qm9
            "Atomization_Enthalpy_B3LYP/6-31G(2df,p)": {"C": -37.8444107, "H": -0.4979123, "O": -75.0622174,"Br": -2571.8212326, "P": -341.255194, "F": -99.716369, "N": -54.581501,
                                                      "S": -398.1033955, "Cl": -460.1343255, },
            "Formation_Enthalpy_B3LYP/6-31G(2df,p)": {"C": -38.0977515, "H": -0.5825296, "O": -75.1626006, "Br":  -2571.8653686  ,
                                             "P": -341.33728, "F": -99.7508616, "N": -54.7601501, "S":  -398.18514 , "Cl": -460.1787121,},
            ###paton
            "Atomization_Enthalpy_M06-2X/def2-TZVP": {"C": -37.8401451, "H": -0.4950509, "O": -74.9702017,"Br": -2574.1479438, "P": -341.2389756, "I": -297.6094277, "F": -99.6058557,
                                                    "N": -54.5190721, "S": -397.9588298, "Cl": -460.1302234, },
            "Formation_Enthalpy_M06-2X/def2-TZVP": {"C": -38.096271, "H": -0.5774791, "O": -75.1599267, "Br":  -2574.1854855 ,
                                             "P": -341.3273055, "I": -297.6366371, "F": -99.7579281, "N": -54.7634213, "S":  -398.1749635 , "Cl": -460.177522,},
            ###qmug
            "Atomization_Enthalpy_ωB97X-D/def2-SVP": {"C": -37.795443, "H": -0.4995215, "O": -74.9743032, "Br": -2573.8545726, "P": -341.1371926, "I": -297.7481894, "F": -99.6120933, "N": -54.5200622, "S": -397.9695617, "Cl": -459.9861458 ,"B": -24.792447},
            "Formation_Enthalpy_ωB97X-D/def2-SVP": {"C": -38.0568914, "H": -0.5791978, "O": -75.0740582, "Br":  -2573.8917577 ,
                                             "P": -341.2167247, "I": -297.7802878,"F": -99.6410919, "N": -54.6950353, "S":  -398.04901 , "Cl": -460.0280681,
                                             "B": -24.792447},

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
        formation_energy = (energy - total_energy )* hatree2kcal
        return {
            'smiles': smiles,
            'energy': energy,
            'element_count': element_count,
            'total_energy': total_energy,
            'formation_energy': formation_energy
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
        # return {'data': data}

def main(args):
    processor = MoleculeProcessor(args.accuracy_level, args.read_method)
    molecules = processor.process_csv(args.file_path)
    datasets = processor.split_dataset(molecules)
    for dataset_name, data in datasets.items():
        output_filename = os.path.splitext(args.file_path)[0] + f'_{dataset_name}.csv'
        processor.save_to_csv(data, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy_level', default="Formation_enrgy_B3LYP/6-31G(d)", help="Specify the accuracy level for calculations.")
    parser.add_argument('--read_method', default="rdkit", help="Method to read and process SMILES.")
    parser.add_argument('--file_path', default="./pc9_processed_data.csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args)