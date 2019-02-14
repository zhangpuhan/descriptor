""" This file records constants """

BOX_SIZE = 6.5
K_T = 0.15
CUT_OFF = 3
FILE_SIZES = 3
ATOM_NUMBER = 50

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

SAMPLE_RUBRIC = {}
