"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

import random

def construct():
    n = 20
    max_colors = 20
    
    def solve_once(seed):
        random.seed(seed)
        adj = [[-1]*n for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        def can_add_edge(u, v, c):
            # Check all 4-cycles containing (u,v)
            # Cycle form: u-v-w-x-u
            # w must be neighbor of v
            # x must be neighbor of w and u
            for w in range(n):
                if w == u or w == v:
                    continue
                c_vw = adj[v][w]
                if c_vw == -1:
                    continue
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    c_wx = adj[w][x]
                    c_xu = adj[u][x]
                    if c_wx == -1 or c_xu == -1:
                        continue
                    
                    # Check cycle u-v-w-x-u
                    # Colors: c, c_vw, c_wx, c_xu
                    if c == c_vw or c == c_wx or c == c_xu or c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                        return False
                    
                    # Check other cycle u-v-x-w-u if edges exist
                    c_vx = adj[v][x]
                    c_wu = adj[w][u]
                    if c_vx != -1 and c_wu != -1:
                        # Colors: c, c_vx, c_wx (same as xw), c_wu
                        if c == c_vx or c == c_wx or c == c_wu or c_vx == c_wx or c_vx == c_wu or c_wx == c_wu:
                            return False
            return True

        for (u, v) in all_edges:
            added = False
            colors_to_try = list(range(max_colors))
            random.shuffle(colors_to_try)
            for c in colors_to_try:
                if can_add_edge(u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    added = True
                    break
            if not added:
                continue
        
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] != -1:
                    result.append([i, j, adj[i][j]])
        return result

    best_edges = []
    # Run multiple restarts to find a better local optimum
    for i in range(100):
        res = solve_once(i)
        if len(res) > len(best_edges):
            best_edges = res
            
    return best_edges
# EVOLVE-BLOCK-END
