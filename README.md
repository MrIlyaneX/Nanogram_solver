# Nonogram_solver

This repository contains implementations of a nonogram solver using a genetic algorithm. 
Nonograms are puzzles where you fill in cells in a grid based on the numbers given on the sides of the grid.

## Overview

A genetic algorithm is used to solve nonograms, simulating the process of natural selection to evolve a population of potential solutions in the direction of the best one. 
Two variants of the genetic algorithm are implemented here:

List-based representation: Each individual in the population is represented as a list of lists, where each sublist represents a row or column in the grid.
Bit string representation: Each individual in the population is represented as a bit string, where each bit represents a cell in the grid.
