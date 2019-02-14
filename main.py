""" This is a Python code to develop AEV descriptor for molecular dynamics """

import read_file
import util

a = read_file.ReadFiles("data")
b = util.GenerateCombinations()
neighbor_combinations = b.generate_combination_dic(200, 2)

a.process_files()
print(neighbor_combinations)
