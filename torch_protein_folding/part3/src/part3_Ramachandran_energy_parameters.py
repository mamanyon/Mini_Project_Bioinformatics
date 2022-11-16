import torch
import matplotlib.pyplot as plt
import numpy as np

from part3_utils import typesData
from part3_utils import RamachandranEnergies

# Calculate phi/psi torsion angles and stores them in type specific data structures
from my_part2 import calculate_ramachandran_maps
from my_part3_ramachandran_energy_parameters import calculate_energy_parameters

def plot_energy(types_data, ramachandran_energy, aa_type, index):
    energy_map = ramachandran_energy.energies[aa_type].parameters
    n_bins = ramachandran_energy.n_bins
    plot_frame(types_data, aa_type, index)
    x = np.array(range(n_bins + 1)) * 360 / n_bins - 180
    x, y = np.meshgrid(x, x)
    energy_map = torch.concat((energy_map, energy_map[0, :].unsqueeze(0)))
    energy_map = torch.concat((energy_map, energy_map[:, 0].unsqueeze(1)), dim=1)
    plt.imshow(energy_map, extent=(-180, 180, 180, -180), cmap='jet', interpolation='bilinear', vmax=5)


def plot(types_data, aa_type, index):
    type_data = types_data.types_data[aa_type]
    phi = type_data.ramachandran['phi']
    psi = type_data.ramachandran['psi']
    plot_frame(types_data, aa_type, index)
    plt.scatter(phi, psi, s=1)


def plot_frame(types_data, aa_type, index):
    ax = plt.subplot(2, 4, index)
    type_data = types_data.types_data[aa_type]
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel('$\Phi$')  # Tex style
    plt.ylabel('$\Psi$')
    plt.title(type_data.name)
    ax.set_aspect('equal', adjustable='box')
    plt.xticks([-180, -90, 0, 90, 180])
    plt.yticks([-180, -90, 0, 90, 180])


def main(path):
    n_bins = 36
    types_data = typesData()
    n_coordinates = torch.load(path + "/CoordNNative.pt")
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    c_coordinates = torch.load(path + "/CoordCNative.pt")
    sequences = torch.load(path + "/sequences.pt")
    masks = torch.load(path + "/nativemask.pt")

    types_data = typesData()
    calculate_ramachandran_maps(types_data, sequences, masks, n_coordinates, ca_coordinates, c_coordinates)
    plot(types_data, 'A', 1)
    plot(types_data, 'W', 2)
    plot(types_data, 'G', 3)
    plot(types_data, 'P', 4)
    ramachandran_energies = RamachandranEnergies(n_bins)
    calculate_energy_parameters(ramachandran_energies, types_data)
    plot_energy(types_data, ramachandran_energies, 'A', 5)
    plot_energy(types_data, ramachandran_energies, 'W', 6)
    plot_energy(types_data, ramachandran_energies, 'G', 7)
    plot_energy(types_data, ramachandran_energies, 'P', 8)
    plt.tight_layout()
    plt.show()
    torch.save(ramachandran_energies, path+'/ramachandran_energy.pt')


if __name__ == '__main__':
    main('C:/Users/keasar/Documents/Work/teaching/mini_project/code/data/sampleData22.5.22')
    # Replace with the relevant path.
