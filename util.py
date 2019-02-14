""" This file contains utility functions """
import torch


class GenerateCombinations:
    """ this function generate position combinations """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def combination_2(self, n_number, n_select):
        nums = [i for i in range(1, n_number)]
        result = []
        self.dfs(nums, 0, [0], result, n_select)
        return torch.tensor(result, device=self.device)

    def dfs(self, nums, index, path, result, n_select):
        if len(path) == n_select + 1:
            result.append(list(path))
            return

        for i in range(index, len(nums)):
            path.append(nums[i])
            self.dfs(nums, i + 1, path, result, n_select)
            path.pop()

    def generate_combination_dic(self, n_number, n_select):
        result = {}
        for i in range(3, n_number + 1):
            result[i] = self.combination_2(i, n_select)

        return result






