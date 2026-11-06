import random
import time

def construct():
    N = 128
    # Use a modulus m > max_d to ensure all 4-APs have distinct residues
    # max_d = (128 - 1) // 3 = 42
    # So m = 43 ensures distinct residues for any valid 4-AP.
    m = 43
    
    # Color i (1-indexed) with (i-1) % m
    colors = [(i - 1) % m for i in range(1, N + 1)]
    
    return colors
