import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def calculate_energy_parameters(rama_energies, types):
    
    types_data = types.types_data     # aa's
    energies = rama_energies.energies # aa's
    offset = 10.5
    for aa in types_data:
        if aa not in ['A', 'G', 'W', 'P']:
            continue
        ramas = len(types_data[aa].ramachandran['phi'])
        grid_dict = create_dict(energies, aa)
        all_xys = list(grid_dict.keys())
        for xy in all_xys:
            x, y = xy[0], xy[1]
            g_xy = 0
            for i in range(ramas):
              pair = [int(types_data[aa].ramachandran['phi'][i]), int(types_data[aa].ramachandran['psi'][i])]
              g_xy_i = distance(x, y, pair, aa, energies)
              g_xy += g_xy_i
            ei = -math.log(g_xy/(1/energies[aa].n_bins ** 2))
            grid_dict[(x,y)].append(ei+offset)
        energies[aa].parameters = dict_to_tensor(grid_dict)
    return 0


def distance(x, y, pair, aa, energies):
    alpha = energies[aa].alpha
    eps = energies[aa].epsilon
    x2, y2 = pair[0], pair[1]
    dist = ((x2 - x)**2) + ((y2 - y)**2)
    power = -alpha*dist
    res = math.exp(power) + eps
    return res
    
def create_dict(energies, aa):
    n = energies[aa].n_bins
    start1, start2 = -180, -180
    dct = {}
    for i in range(n):
        for j in range(n):
            dct[(start1, start2)] = []
            start1 += 10
        start2 += 10
        start1 = -180
    return dct
    
def dict_to_tensor(dct):
    grid = torch.tensor([[0]*36]*36)
    i, j = 0, 0
    for key, vals in dct.items():
        if j==36:
            j = 0
            i += 1
        if vals == []:
            grid[i][j] = 0
        else:
            grid[i][j] = sum(vals)
        j += 1
    return grid
