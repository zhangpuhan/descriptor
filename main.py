""" This is a Python code to develop AEV descriptor for molecular dynamics """

import read_file
import util

# compute neighbor parameters:
neighbor_comb = util.GenerateCombinations()
angular_neighbor_combinations = neighbor_comb.generate_combination_dic(200, 2)
radial_neighbor_combinations = neighbor_comb.generate_combination_dic(200, 1)

# computer sample parameters:
sample_comb = util.GenerateSampleGrid()
angular_sample_comb = sample_comb.generate_angular_grid()
radial_sample_comb = sample_comb.generate_radial_grid()

# process snapshots
process_file_init = read_file.Aev("data")
process_file_init.process_files(radial_sample_comb, angular_sample_comb, radial_neighbor_combinations,
                                angular_neighbor_combinations)

