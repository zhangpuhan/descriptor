""" This file records constants """

BOX_SIZE = 1.0
K_T = 0.15
CUT_OFF = 1.0
FILE_SIZES = 1001
ATOM_NUMBER = 10

DIRECTIONS = [[-BOX_SIZE, -BOX_SIZE, -BOX_SIZE],
              [0, -BOX_SIZE, -BOX_SIZE],
              [BOX_SIZE, -BOX_SIZE, -BOX_SIZE],
              [-BOX_SIZE, 0, -BOX_SIZE],
              [0, 0, -BOX_SIZE],
              [BOX_SIZE, 0, -BOX_SIZE],
              [-BOX_SIZE, BOX_SIZE, -BOX_SIZE],
              [0, BOX_SIZE, -BOX_SIZE],
              [BOX_SIZE, BOX_SIZE, -BOX_SIZE],
              [-BOX_SIZE, -BOX_SIZE, 0],
              [0, -BOX_SIZE, 0],
              [BOX_SIZE, -BOX_SIZE, 0],
              [-BOX_SIZE, 0, 0],
              [0, 0, 0],
              [BOX_SIZE, 0, 0],
              [-BOX_SIZE, BOX_SIZE, 0],
              [0, BOX_SIZE, 0],
              [BOX_SIZE, BOX_SIZE, 0],
              [-BOX_SIZE, -BOX_SIZE, BOX_SIZE],
              [0, -BOX_SIZE, BOX_SIZE],
              [BOX_SIZE, -BOX_SIZE, BOX_SIZE],
              [-BOX_SIZE, 0, BOX_SIZE],
              [0, 0, BOX_SIZE],
              [BOX_SIZE, 0, BOX_SIZE],
              [-BOX_SIZE, BOX_SIZE, BOX_SIZE],
              [0, BOX_SIZE, BOX_SIZE],
              [BOX_SIZE, BOX_SIZE, BOX_SIZE]]

RADIAL_SAMPLE_RUBRIC = {"Eta": [1.8],
                        "Rs": [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
                               0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375,
                               0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575,
                               0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775,
                               0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]}

ANGULAR_SAMPLE_RUBRIC = {"Eta": [1.0, 2.0, 3.0],
                         "Rs": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                         "Zeta": [2.0, 4.0, 6.0],
                         "Thetas": [0.0, 1.57, 3.14, 4.71]}





