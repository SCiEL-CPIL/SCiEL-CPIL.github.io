import itertools
import random
import numpy as np

def construct():
    n = 18
    # Spectral approach: Start with a random matrix, but we know Turan graphs are optimal for this metric.
    # We will construct T(18, 9) which has score 4.
    # Parts: 9 parts of size 2.
    parts = []
    for i in range(9):
        parts.append([2*i, 2*i+1])
        
    edges = []
    for i in range(9):
        for j in range(i+1, 9):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
                    
    return edges
