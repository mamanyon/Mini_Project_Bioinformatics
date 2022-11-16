import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from examples import torsionAngle

def vector_dm(x):
    # x is a vector column 1D tensor
    xtx = x @ x.t()  # column vector times row vector = 3 X2 square matrix
    xtx2 = 2 * xtx  # Scalar multiplication
    xtx_diagonal = x**2  # xtx.diag().unsqueeze(0)  # Unsqueeze adds dimension to the tensor. Here from 2 to 1 X 2
    xtx_diagonal2 = xtx_diagonal + xtx_diagonal.t()
    dm = (xtx_diagonal2 - xtx2).sqrt()
    return dm


def plot_dm(path, index):
    dm, contact_map, sums = get_dm(path, index)
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Distance map')
    plt.imshow(dm.numpy(), cmap=cm.jet)
    contact_map = (dm <= 10)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Contact map')
    plt.imshow(contact_map.numpy(), cmap=cm.jet)
    sums = contact_map.sum(dim=0)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Residue contacts')
    plt.plot(range(len(sums)), sums)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Contact histogram')
    plt.hist(sums.numpy(), 20)
    name = get_id(path, index)
    fig.suptitle(name, fontsize=16)
    plt.show()


def example(path, index):
    dm, contact_map, sums = get_dm(path, index)
    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('Distance map')
    plt.imshow(dm.numpy(), cmap=cm.jet)
    contact_map = (dm <= 10)
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('Contact map')
    plt.imshow(contact_map.numpy(), cmap=cm.jet)
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title('Ramachandran')
    phi, psi = ramachandran(path, index)
    plt.scatter(phi, psi)
    ax3.set_xlim([-180, 180])
    ax3.set_ylim([-180, 180])
    ax3.set_aspect('equal', 'box')
    name = get_id(path, index)
    fig.suptitle(name, fontsize=16)
    plt.show()

def get_dm(path, index):
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    ca_coordinates_i = ca_coordinates[index] * 0.01
    xtx = ca_coordinates_i @ ca_coordinates_i.t()
    x2 = (ca_coordinates_i ** 2).sum(dim=1, keepdim=True)
    x2 = x2 + x2.t()
    n = x2.shape[0]
    dm = torch.relu((x2 - 2 * xtx).sqrt())
    indices = ca_coordinates_i[:, 1] > 999
    mx = torch.max(ca_coordinates_i[ca_coordinates_i < 999])
    dm[indices, :] = mx
    dm[:, indices] = mx
    contact_map = (dm <= 10)
    sums = contact_map.sum(dim=0)
    return dm, contact_map, sums

def ramachandran(path, index):
    n_coordinates = torch.load(path + "/CoordNNative.pt")
    ca_coordinates = torch.load(path + "/CoordCaNative.pt")
    c_coordinates = torch.load(path + "/CoordCNative.pt")

    n_coordinates_i = n_coordinates[index]
    ca_coordinates_i = ca_coordinates[index]
    c_coordinates_i = c_coordinates[index]
    mask_i = n_coordinates_i[:, 0] < 9999

    n_coordinates_i = n_coordinates_i[mask_i, :] / 100
    ca_coordinates_i = ca_coordinates_i[mask_i, :] / 100
    c_coordinates_i = c_coordinates_i[mask_i, :] / 100

    c0_phi = c_coordinates_i[0:-1, :]
    n_phi = n_coordinates_i[1:, :]
    ca_phi = ca_coordinates_i[1:, :]
    c_phi = c_coordinates_i[1:, :]
    phi, cosPhi, sinPhi = torsionAngle(c0_phi.t(), n_phi.t(), ca_phi.t(), c_phi.t())
    n_psi = n_coordinates_i[0:-1, :]
    ca_psi = ca_coordinates_i[0:-1, :]
    c_psi = c_coordinates_i[0:-1, :]
    n1_psi = n_coordinates_i[1:, :]
    psi, cosPsi, sinPsi = torsionAngle(n_psi.t(), ca_psi.t(), c_psi.t(), n1_psi.t())

    return phi, psi



def get_id(path, index):
    ids = torch.load(path + "/ids.pt")
    return ids[index]
