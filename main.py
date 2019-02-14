""" This is a Python code to develop AEV descriptor for molecular dynamics """

import read_file
import util

a = read_file.Aev("data")
b = util.GenerateCombinations()
angular_neighbor_combinations = b.generate_combination_dic(200, 2)
radial_neighbor_combinations = b.generate_combination_dic(200, 1)

a.process_files()
print(angular_neighbor_combinations)
print(radial_neighbor_combinations)
