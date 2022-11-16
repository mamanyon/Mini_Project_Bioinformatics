from examples import a_b_distance
from examples import a_b_dm
from examples import read_ca_coordinates
from examples import a_b_distance_with_grad
from examples import torsionAngle
import torch

def main():
    print("Example of a_b_distance")
    print(a_b_distance(10, 5))
    print("Example of a_b_dm")
    print(a_b_dm(10, 5))
    print("Example of loading protein Coordinates")
    read_ca_coordinates("C:/Users/keasar/Documents/Work/teaching/mini_project/code/data/sampleData/")
    print("Back to a_b_distance this time with gradient")
    print(a_b_distance_with_grad(10., 5.))
    print("example of torsion angles")
    print(torsionAngle(torch.tensor([[0., 0., 0.]]).t(),
                       torch.tensor([[1., 0., 0.]]).t(),
                       torch.tensor([[1., 1., 0.]]).t(),
                       torch.tensor([[1., 1., 1.]]).t()))
    print(torsionAngle(torch.tensor([[0., 0., 0.]]).t(),
                       torch.tensor([[1., 0., 0.]]).t(),
                       torch.tensor([[1., 1., 0.]]).t(),
                       torch.tensor([[0., 0., 0.]]).t()))
    print(torsionAngle(torch.tensor([[0., 0., 0.]]).t(),
                       torch.tensor([[1., 0., 0.]]).t(),
                       torch.tensor([[1., 1., 0.]]).t(),
                       torch.tensor([[1., 1., -1.]]).t()))



if __name__ == '__main__':
    main()
