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
            "Formation_EE_B3LYP/6-31G(d)": {"C": -38.10280868, "H": -0.5877412036, "O": -75.160020097,
                                            "F": -99.749126099, "N": -54.762064333, },
            "atomization_EE_B3LYP/6-31G(d)": {"C": -37.8462799747, "H": -0.500272784186, "O": -75.0606214291,
                                              "N": -54.5844893898, "F": -99.7155354580},
            ###qm9
            "Atomization_Enthalpy_B3LYP/6-31G(2df,p)": {"C": -37.8444107, "H": -0.4979123, "O": -75.0622174,"Br": -2571.8212326, "P": -341.255194, "F": -99.716369, "N": -54.581501,
                                                      "S": -398.1033955, "Cl": -460.1343255, },
            "Formation_Enthalpy_B3LYP/6-31G(2df,p)": {"C": -38.0977515, "H": -0.5825296, "O": -75.1626006, "Br":  -2571.8653686  ,
                                             "P": -341.33728, "F": -99.7508616, "N": -54.7601501, "S":  -398.18514 , "Cl": -460.1787121,},
            ###paton
            "Atomization_Enthalpy_M06-2X/def2-TZVP": {"C": -37.840145, "H": -0.495778, "O": -75.063797,"Br": -2574.147944, "P": -341.238976, "I": -297.609428, "F": -99.731367,
                                                    "N": -54.584752, "S": -398.091829, "Cl": -460.130223, },
            "Formation_Enthalpy_M06-2X/def2-TZVP": {"C": -38.096271, "H": -0.5774791, "O": -75.1599267, "Br": -2574.1854855,
                                                    "P": -341.360178, "I": -297.6366371, "F": -99.7579281,"N": -54.7634213, "S": -398.1749635, "Cl": -460.177522,
                                                    "B": -24.821267},
            "Atomization_EE_M06-2X/def2-TZVP": {"C": -37.8425055992, "H": -0.498138511711, "O": -75.0661577121,"Br": -2574.15030430, "P": -341.241336098, "I": -297.611788203, "F": -99.7337277474, "N": -54.5871121790, "S": -398.094189269, "Cl": -460.132583898, },

            "Formation_EE_M06-2X/def2-TZVP": { "C": -38.1028042723,"H": -0.58422991,"O": -75.16359090,"Br": -2574.18771415,"P": -341.363144482,"I": -297.63881001,"F": -99.76091706,
                    "N": -54.76795403,"S": -398.17753078,"Cl": -460.17991970 ,"B": -24.82619484},
            ###qmug
            "Atomization_Enthalpy_ωB97X-D/def2-SVP": {"C": -37.795443, "H": -0.4995215, "O": -74.9743032, "Br": -2573.8545726, "P": -341.1371926, "I": -297.7481894, "F": -99.6120933, "N": -54.5200622, "S": -397.9695617, "Cl": -459.9861458 ,"B": -24.792447},
            "Formation_Enthalpy_ωB97X-D/def2-SVP": {"C": -38.0568914, "H": -0.5791978, "O": -75.0740582, "Br":  -2573.8917577 ,
                                             "P": -341.245752, "I": -297.7802878,"F": -99.6410919, "N": -54.6950353, "S":  -398.04901 , "Cl": -460.0280681,
                                             "B": -24.792447},
            "Electronic_Formation_Enthalpy_ωB97X-D/def2-SVP": {
                "C": -38.0635871, "H": -0.58587727642, "O": -75.0777307,
                "Br": -2573.8939851, "P": -341.2486453, "I": -297.7824569,
                "F": -99.6439847, "N": -54.6995728, "S": -398.0515704,
                "Cl": -460.0304402, "B": -24.7974109},
            ###ωB97M-V/def2-TZVPD
            "Formation_Enthalpy_ωB97M-V/def2-TZVPDmlip": {"C": -38.10051377, "H": -0.5737195, "O": -75.165627, "Br": -2574.005573,
                                                    "P": -341.3568803, "I": -297.6969885, "F": -99.7733085,"N": -54.7717995, "S": -398.17277, "Cl": -460.1718495,
                                                    "B": -24.81969992},
            "Formation_Gibbs_ωB97M-V/def2-TZVPDmlip": {"C": -38.10156693, "H": -0.581114, "O": -75.165627, "Br": -2574.019476,
                                                    "P": -341.3651825, "I": -297.7117365, "F": -99.784771,"N": -54.782663, "S": -398.17277, "Cl": -460.184493,
                                                    "B": -24.82288967},
            "Formation_EE_ωB97M-V/def2-TZVPDmlip": {"C": -38.10727307, "H": -0.580451, "O": -75.169272, "Br": -2574.007799,
                                                    "P": -341.3598445, "I": -297.699161, "F": -99.7762295,"N": -54.776296, "S": -398.175349, "Cl": -460.1742485,
                                                    "B": -24.82454108},
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
    parser.add_argument('--file_path', default="./pc9.csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args)
