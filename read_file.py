""" This file creates atom environment vectors """

import os
import pandas as pd
import natsort
import torch
from constant import DIRECTIONS, FILE_SIZES, CUT_OFF, ATOM_NUMBER
from fre_functions import f_c, exponential_map


class Aev:
    """ processing raw data """
    def __init__(self, path):
        self.filenames = [] 
        for _, __, files in os.walk(path):
            for filename in files:
                if filename.endswith(".csv"):
                    self.filenames.append(path + "/" + filename)
        self.filenames = natsort.natsorted(self.filenames)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage *******************************')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')

    def process_files(self, radial_sample_comb, angular_sample_comb, radial_neighbor_combinations,
                      angular_neighbor_combinations):

        print(str(len(self.filenames)) + " files need to be processed")
        for i in range(FILE_SIZES):
            coordinate_tensor = torch.tensor(pd.read_csv(self.filenames[i], header=None).values, device=self.device)
            print("********************************************")
            print("Data file " + self.filenames[i] + " is being processing:")
            x_coordinate = torch.reshape(coordinate_tensor[:, :1], (1, -1))
            y_coordinate = torch.reshape(coordinate_tensor[:, 1:2], (1, -1))
            z_coordinate = torch.reshape(coordinate_tensor[:, 2:3], (1, -1))
            force = coordinate_tensor[:, 3:]

            x_cat = torch.cat(tuple([x_coordinate for _ in range(x_coordinate.size()[1])]), 0)
            y_cat = torch.cat(tuple([y_coordinate for _ in range(y_coordinate.size()[1])]), 0)
            z_cat = torch.cat(tuple([z_coordinate for _ in range(z_coordinate.size()[1])]), 0)

            # print("### 27 directions need to be handled. ###")
            neighbor_x = [[] for _ in range(ATOM_NUMBER)]
            neighbor_y = [[] for _ in range(ATOM_NUMBER)]
            neighbor_z = [[] for _ in range(ATOM_NUMBER)]
            distance_a = [[] for _ in range(ATOM_NUMBER)]

            neighbor_x, neighbor_y, neighbor_z, distance_a = self.extract_neighbors(x_cat, y_cat, z_cat,
                                                                                    neighbor_x, neighbor_y, neighbor_z,
                                                                                    distance_a)

            self.generate_radial_samples(distance_a, radial_sample_comb, radial_neighbor_combinations)
            self.generate_angular_samples(distance_a, angular_sample_comb, angular_neighbor_combinations)

            # torch.cat([torch.index_select(A, 0, i).unsqueeze(0) for a, i in zip(A, ind)])

            # print(neighbor_y[1])
            # print(neighbor_z[1])
            # print(distance_a[1].size()[0], neighbor_x[1].size()[0])
            print("file " + str(i) + " has been retrieved")

    def extract_neighbors(self, x_cat, y_cat, z_cat, neighbor_x, neighbor_y, neighbor_z, distance_a):
        for i in range(ATOM_NUMBER):
            neighbor_x[i].append(torch.reshape(x_cat[0][i], (1, )))
            neighbor_y[i].append(torch.reshape(y_cat[0][i], (1, )))
            neighbor_z[i].append(torch.reshape(z_cat[0][i], (1, )))
            distance_a[i].append(torch.reshape(torch.tensor(0.0, device=self.device, dtype=torch.float64), (1, )))

        for x_direct, y_direct, z_direct in DIRECTIONS:
            # print("### dealing with direction " + str([x_direct, y_direct, z_direct]))

            x_cat_temp = x_cat.add(x_direct).clone()
            y_cat_temp = y_cat.add(y_direct).clone()
            z_cat_temp = z_cat.add(z_direct).clone()

            distance = (x_cat.t() - x_cat_temp).pow(2) + (y_cat.t() - y_cat_temp).pow(2) + \
                       (z_cat.t() - z_cat_temp).pow(2)

            position = (torch.le(distance, CUT_OFF ** 2) == 1).nonzero()
            if [x_direct, y_direct, z_direct] == [0, 0, 0]:
                position = position[(position[:, 0] != position[:, 1]).nonzero().squeeze(1)]

            # print("-------------------------------------------")

            for i in range(ATOM_NUMBER):
                final_position = position[(position[:, 0] == i).nonzero().squeeze(1)][:, 1]
                if final_position.size()[0] == 0:
                    continue
                neighbor_x[i].append(torch.index_select(x_cat_temp.t(), 0, final_position)[:, 0])
                neighbor_y[i].append(torch.index_select(y_cat_temp.t(), 0, final_position)[:, 0])
                neighbor_z[i].append(torch.index_select(z_cat_temp.t(), 0, final_position)[:, 0])
                temp_distance = torch.reshape(torch.index_select(distance, 0,
                                                                 torch.tensor([i], device=self.device)), (-1, ))
                distance_a[i].append(torch.index_select(temp_distance, 0, final_position).sqrt_())

        for i in range(ATOM_NUMBER):
            neighbor_x[i] = torch.cat(tuple(neighbor_x[i]), dim=0)
            neighbor_y[i] = torch.cat(tuple(neighbor_y[i]), dim=0)
            neighbor_z[i] = torch.cat(tuple(neighbor_z[i]), dim=0)
            distance_a[i] = torch.cat(tuple(distance_a[i]), dim=0)

        return neighbor_x, neighbor_y, neighbor_z, distance_a

    @staticmethod
    def generate_radial_samples(distance_a, radial_sample_comb, radial_neighbor_combinations):
        result = []
        dominant_size = radial_sample_comb.size()[0]
        print("generating " + str(dominant_size) + " radial elements")
        for i in range(len(distance_a)):
            neighbor_size = distance_a[i].size()[0]
            neighbor_pairs = radial_neighbor_combinations[neighbor_size]
            rs_list_init = torch.cat(tuple([torch.index_select(distance_a[i], 0, _).unsqueeze(0)
                                            for __, _ in zip(distance_a[i], neighbor_pairs)]))

            rs_list = rs_list_init[:, 1:]
            manipulate_tensor = torch.reshape(torch.cat(
                tuple([rs_list for _ in range(dominant_size)])), (dominant_size, -1))
            # print(manipulate_tensor)
            # print(f_c(manipulate_tensor))
            result.append(exponential_map(
                manipulate_tensor, radial_sample_comb).mul(f_c(manipulate_tensor)).sum(dim=1))

        return result

    @staticmethod
    def generate_angular_samples(distance_a, angular_sample_comb, angular_neighbor_combinations):
        result = []
        dominant_size = angular_sample_comb.size()[0]
        print(dominant_size)
        print("generating " + str(dominant_size) + " angular elements")
        for i in range(5):
            neighbor_size = distance_a[i].size()[0]
            neighbor_triples = angular_neighbor_combinations[neighbor_size]
            # print(neighbor_triples)

            # part_1: last_fc
            rs_list_init = torch.cat(tuple([torch.index_select(distance_a[i], 0, _).unsqueeze(0)
                                            for __, _ in zip(neighbor_triples, neighbor_triples)]))
            rs_list = rs_list_init[:, 1:]
            mul_list_1 = f_c(rs_list[:, :1]).mul(f_c(rs_list[:, 1:]))
            manipulate_tensor_1 = torch.reshape(torch.cat(
                tuple([mul_list_1 for _ in range(dominant_size)])), (dominant_size, -1))

            print(manipulate_tensor_1.size())

            # part_2: exponential
            mul_list__2 = rs_list[:, :1].add(rs_list[:, 1:])
            print(mul_list__2)






