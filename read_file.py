import os
import pandas as pd
import natsort
import torch
from constant import DIRECTIONS, FILE_SIZES, CUT_OFF, ATOM_NUMBER


class ReadFiles:
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
            print('Memory Usage********************************')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')

    def process_files(self):
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

            print("### 27 directions need to be handled. ###")
            neighbor_x = [[] for _ in range(ATOM_NUMBER)]
            neighbor_y = [[] for _ in range(ATOM_NUMBER)]
            neighbor_z = [[] for _ in range(ATOM_NUMBER)]
            distance_a = [[] for _ in range(ATOM_NUMBER)]

            neighbor_x, neighbor_y, neighbor_z = self.extract_neighbors(x_cat, y_cat, z_cat,
                                                                        neighbor_x, neighbor_y, neighbor_z,
                                                                        distance_a)
            print(neighbor_x)
            print("file " + str(i) + " has been retrieved")

    def extract_neighbors(self, x_cat, y_cat, z_cat, neighbor_x, neighbor_y, neighbor_z, distance_a):
        for i in range(ATOM_NUMBER):
            neighbor_x[i].append(torch.reshape(x_cat[0][i], (1, )))
            neighbor_y[i].append(torch.reshape(y_cat[0][i], (1, )))
            neighbor_z[i].append(torch.reshape(z_cat[0][i], (1, )))

        for x_direct, y_direct, z_direct in DIRECTIONS:
            print("### dealing with direction " + str([x_direct, y_direct, z_direct]))

            x_cat_temp = x_cat.add(x_direct).clone()
            y_cat_temp = y_cat.add(y_direct).clone()
            z_cat_temp = z_cat.add(z_direct).clone()

            distance = (x_cat.t() - x_cat_temp).pow(2) + (y_cat.t() - y_cat_temp).pow(2) + \
                       (z_cat.t() - z_cat_temp).pow(2)

            position = (torch.le(distance, CUT_OFF ** 2) == 1).nonzero()
            position = position[(position[:, 0] != position[:, 1]).nonzero().squeeze(1)]
            print(position)

            print("-------------------------------------------")

            for i in range(ATOM_NUMBER):
                final_position = position[(position[:, 0] == i).nonzero().squeeze(1)][:, 1]
                final_position_2 = position[(position[:, 0] == i).nonzero().squeeze(1)]
                if final_position.size()[0] == 0:
                    continue
                neighbor_x[i].append(torch.index_select(x_cat_temp.t(), 0, final_position)[:, 0])
                neighbor_y[i].append(torch.index_select(y_cat_temp.t(), 0, final_position)[:, 0])
                neighbor_z[i].append(torch.index_select(z_cat_temp.t(), 0, final_position)[:, 0])
                # distance_a.append(torch.index_select(distance, 0, final_position_2))

        for i in range(ATOM_NUMBER):
            neighbor_x[i] = torch.cat(tuple(neighbor_x[i]), dim=0)

        return neighbor_x, neighbor_y, neighbor_y
