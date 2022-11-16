import torch
import matplotlib.pyplot as plt
import numpy as np


# Simple distance

def a_b_distance(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    out = ((a - b) ** 2).sqrt()
    return out.item()


# Distance map of two variables
def a_b_dm(a, b):
    ab = torch.FloatTensor(2, 1)  # A 2 X 1 tensor, column vector
    ab[0, 0] = a
    ab[1, 0] = b
    ab = ab @ ab.t()  # column vector times row vector = 3 X2 square matrix
    ab2 = 2 * ab  # Scalar multiplication
    ab_diagonal = ab.diag().unsqueeze(0)  # Unsqueeze adds dimension to the tensor. Here from 2 to 1 X 2
    ab_diagonal2 = ab_diagonal + ab_diagonal.t()
    dm = (ab_diagonal2 - ab2).sqrt()
    return dm

def vector_dm(v):
    # v is an n X 1 tensor  (column vector)
    vv = v @ v.t()
    vv2 = 2 * vv
    vv_diagonal = vv.diag().unsqueeze(0)  # Unsqueeze adds dimension to the tensor. Here from n to 1 X n
    vv_diagonal2 = vv_diagonal + vv_diagonal.t()
    dm = (vv_diagonal2 - vv2).sqrt()
    return dm


# Read coordinates of C-alpha
def read_ca_coordinates(path):
    ca_coordinates = torch.load(path+"/CoordCaNative.pt")
    # What is it that we loaded
    print(type(ca_coordinates))
    #  <class 'list'>
    #  What is the length of the list?
    print(len(ca_coordinates))
    #  153
    #  Lets look at the first element of the list
    first = ca_coordinates[0]
    #  What is it?
    print(type(first))
    #  <class 'torch.Tensor'>
    #  Of what dimensions?
    print(first.shape)
    #  torch.Size([215, 3])  That is the protein has 215 C-alpha atoms (rows), each with X, Y, and Z coordinates
    plt.figure()
    ax = plt.axes(projection='3d')
    indices = first[:, 1] < 999
    x = np.array(first[indices, 0])
    y = np.array(first[indices, 1])
    z = np.array(first[indices, 2])
    ax.plot3D(x, y, z, 'gray')
    plt.show()


def a_b_distance_with_grad(a, b):
    a = torch.tensor([a], requires_grad=True)
    b = torch.tensor([b], requires_grad=True)
    out = ((a - b) ** 2).sqrt()
    out.backward()
    return out.item(), a.grad, b.grad


def torsionAngle(V1,V2,V3,V4):
    # V in 3xN
    A = V2 - V1
    B = V3 - V2
    C = V4 - V3

    Bsq = torch.relu(torch.sum(B * B, dim=0, keepdim=True))
    AC  = torch.sum(A * C, dim=0, keepdim=True)
    AB  = torch.sum(A * B, dim=0, keepdim=True)
    BC  = torch.sum(B * C, dim=0, keepdim=True)
    x   = -torch.sum(Bsq*AC, dim=0, keepdim=True) + torch.sum(AB*BC, dim=0, keepdim=True)

    absB = torch.sqrt(Bsq).sum(dim=0, keepdim=True)
    BxC  = torch.cross(B, C)
    y    = torch.sum((absB*A)*BxC, dim=0, keepdim=True)

    cosTheta = x/torch.sqrt(x**2 + y**2 + 1e-3)
    sinTheta = y/torch.sqrt(x**2 + y**2 + 1e-3)
    theta = torch.arccos(cosTheta)
    theta = theta*torch.sign(y)
    return 180 * theta / torch.pi, cosTheta, sinTheta
