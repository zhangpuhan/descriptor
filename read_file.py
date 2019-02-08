import os
import pandas as pd
import natsort
import torch


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
        for i in range(1):
            torch_tensor = torch.tensor(pd.read_csv(self.filenames[i]).values, device=self.device)
            print("********************************************")
            print("Data file " + self.filenames[i] + " is being processing:")
            x_coordinate = torch_tensor[:, :1]
            y_coordinate = torch_tensor[:, 1:2]
            z_coordinate = torch_tensor[:, 2:3]
            force = torch_tensor[:, 3:]
            print(x_coordinate)

            print(y_coordinate)

            print(z_coordinate)
            print(torch_tensor)
            print("********************************************")

