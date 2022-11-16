import torch
import re
import numpy as np
from scipy.optimize import curve_fit


def torsionAngle(V1, V2, V3, V4):
    # V in 3xN
    A = V2 - V1
    B = V3 - V2
    C = V4 - V3

    Bsq = torch.relu(torch.sum(B * B, dim=0, keepdim=True))
    AC = torch.sum(A * C, dim=0, keepdim=True)
    AB = torch.sum(A * B, dim=0, keepdim=True)
    BC = torch.sum(B * C, dim=0, keepdim=True)
    x = -torch.sum(Bsq * AC, dim=0, keepdim=True) + torch.sum(AB * BC, dim=0, keepdim=True)

    absB = torch.sqrt(Bsq).sum(dim=0, keepdim=True)
    BxC = torch.cross(B, C)
    y = torch.sum((absB * A) * BxC, dim=0, keepdim=True)

    cosTheta = x / torch.sqrt(x ** 2 + y ** 2 + 1e-3)
    sinTheta = y / torch.sqrt(x ** 2 + y ** 2 + 1e-3)
    theta = torch.arccos(cosTheta)
    theta = theta * torch.sign(y)
    return 180 * theta / torch.pi, cosTheta, sinTheta


def get_ramachandran(n_coordinates, ca_coordinates, c_coordinates):
    c0_phi = c_coordinates[0:-2, :]
    n_phi = n_coordinates[1:-1, :]
    ca_phi = ca_coordinates[1:-1, :]
    c_phi = c_coordinates[1:-1, :]
    phi_tmp, cos_phi, sin_phi = torsionAngle(c0_phi.t(), n_phi.t(), ca_phi.t(), c_phi.t())
    phi = torch.zeros(1, phi_tmp.shape[1] + 2)  # Why?
    phi[0, 1:-1] = phi_tmp[0, :]
    n_psi = n_coordinates[1:-1, :]
    ca_psi = ca_coordinates[1:-1, :]
    c_psi = c_coordinates[1:-1, :]
    n1_psi = n_coordinates[2:, :]
    psi_tmp, cos_psi, sin_psi = torsionAngle(n_psi.t(), ca_psi.t(), c_psi.t(), n1_psi.t())
    psi = torch.zeros(phi.shape)
    psi[0, 1:-1] = psi_tmp[0, :]
    return phi, psi


class typeData:
    def __init__(self, name, name1, name3):
        self.name = name
        self.name1 = name1
        self.name3 = name3
        self.ramachandran = {'phi': torch.tensor([], dtype=torch.float32), 'psi': torch.tensor([], dtype=torch.float32)}


class typesData:
    def __init__(self):
        self.types_data = {
            'A': typeData('Alanine', 'Ala', 'A'),
            'C': typeData('Cysteine', 'Cys', 'A'),
            'D': typeData('Aspartic Acid', 'Asp', 'D'),
            'E': typeData('Glutamic Acid', 'Glu', 'E'),
            'F': typeData('Phenylalanine', 'Phe', 'F'),
            'G': typeData('Glycine', 'Gly', 'G'),
            'H': typeData('Histidine', 'His', 'H'),
            'I': typeData('Isoleucine', 'Ile', 'I'),
            'K': typeData('Lysine', 'Lys', 'K'),
            'L': typeData('Leucine', 'Leu', 'L'),
            'M': typeData('Methionine', 'Met', 'M'),
            'N': typeData('Asparagine', 'Asn', 'N'),
            'P': typeData('Proline', 'Pro', 'P'),
            'Q': typeData('Glutamione', 'Gln', 'Q'),
            'R': typeData('Arginine', 'Arg', 'R'),
            'S': typeData('Serine', 'Ser', 'S'),
            'T': typeData('Threonine', 'Thr', 'T'),
            'V': typeData('Valine', 'Val', 'V'),
            'W': typeData('Tryptophan', 'Trp', 'W'),
            'Y': typeData('Tyrosine', 'Tyr', 'Y')
        }


class RamachandranEnergies:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.alpha = 0.1 * n_bins / 360
        self.epsilon = 0.000001
        self.energies = {
            'A': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'C': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'D': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'E': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'F': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'G': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'H': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'I': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'K': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'L': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'M': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'N': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'P': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'Q': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'R': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'S': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'T': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'V': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'W': RamachandranEnergy(n_bins, self.alpha, self.epsilon),
            'Y': RamachandranEnergy(n_bins, self.alpha, self.epsilon)
        }


class RamachandranEnergy:
    def __init__(self, n_bins, alpha, epsilon):
        self.n_bins = n_bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.parameters = []


def get_mask(mask_string):
    return torch.tensor([c == '+' for c in mask_string])


def get_type_mask(sequence, mask, aa_type):
    mask = mask.unsqueeze(0)
    type_mask = torch.full([1, len(sequence)], False)
    for i in re.finditer(aa_type, sequence):
        type_mask[0, i.start()] = True
    return mask & type_mask


class Histogram:
    def __init__(self, min_value, max_value, n_bins, epsilon):
        self.min = min_value
        self.max = max_value
        self.n_bins = n_bins
        self.n_events = 0
        self.values = torch.ones(n_bins) * epsilon
        self.indices = torch.tensor(range(n_bins))
        self.bin_width = (self.max - self.min) / self.n_bins
        self.bin_values = torch.tensor(
            [0.5 * self.bin_width + (i - self.min) * self.bin_width for i in range(self.n_bins)])
        self.frequencies = torch.zeros(n_bins)

    def add(self, values):
        if values.nelement() != 0:
            if torch.min(values) < self.min or torch.max(values) > self.max:
                raise Exception(str(torch.min(values)) + ' < ' + str(self.min) + ' or ' + str(torch.max(values)) + ' > ' +
                                str(self.max))
            h = torch.histogram(values, bins=torch.tensor([self.min + i * self.bin_width for i in range(self.n_bins + 1)]))
            self.values[self.indices] = self.values[self.indices] + h.hist
            self.n_events = self.n_events + h.hist.sum()
            self.frequencies = self.values / self.n_events


def get_name(full_name):
    return full_name.split('#')[-1].split('__')[0]


def get_aa_type_pairs_histograms(min_value, max_value, n_bins, epsilon):
    types = 'ACDEFGHIKLMNPQRSTVWY'
    histograms = {}
    for i in range(20):
        for j in range(i, 20):
            histograms[types[i] + '_' + types[j]] = Histogram(min_value, max_value, n_bins, epsilon)
    return histograms
