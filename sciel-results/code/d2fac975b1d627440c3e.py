import time
import random
import itertools
import sys
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def solve_insertion(n_target, time_limit):
        MAX_N = 20
        adj_np = [np.zeros((MAX_N, MAX_N), dtype=np.int8) for _ in range(4)]
        has_path_len4 = [np.zeros((MAX_N, MAX_N), dtype=bool) for _ in range(4)]
        
        current_n = 0
        
        def update_path_counts(n_curr):
            for c in range(4):
                A = adj_np[c][:n_curr, :n_curr].astype(np.int32)
                A2 = A @ A
                A4 = A2 @ A2
                has_path_len4[c][:n_curr, :n_curr] = (A4 > 0)
        
        for v in range(n_target):
            if time.time() > time_limit:
                return None
                
            if v == 0:
                current_n = 1
                continue
            
            colors_assigned = [-1] * v
            
            def backtrack(idx):
                nonlocal current_n
                if time.time() > time_limit:
                    return False
                
                if idx == v:
                    for i in range(v):
                        c = colors_assigned[i]
                        adj_np[c][v][i] = 1
                        adj_np[c][i][v] = 1
                    current_n = v + 1
                    update_path_counts(current_n)
                    return True
                
                colors_to_try = [0, 1, 2, 3]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    for e in range(idx):
                        if colors_assigned[e] == c:
                            if has_path_len4[c][idx][e]:
                                valid = False
                                break
                    
                    if valid:
                        colors_assigned[idx] = c
                        if backtrack(idx + 1):
                            return True
                        colors_assigned[idx] = -1
                        
                return False
            
            if not backtrack(0):
                return None
        
        colors = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                for c in range(4):
                    if adj_np[c][i][j]:
                        colors.append(c)
                        break
        return colors

    best_n = 0
    best_colors = None
    
    for n_target in [17, 18, 19]:
        if time.time() > deadline - 10:
            break
            
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            res = solve_insertion(n_target, deadline - 10)
            if res:
                best_n = n_target
                best_colors = res
                break
        
    if best_colors:
        return {'n': best_n, 'colors': best_colors}
        
    n = 10
    colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    return {'n': n, 'colors': colors}
