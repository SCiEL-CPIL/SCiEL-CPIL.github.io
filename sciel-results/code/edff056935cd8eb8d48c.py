import numpy as np
import time
import sys
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Target energy is 0 for scaled integers (sum = 8)
    # U, V, W are integers representing 2 * original values
    # Original sum = 1 => (U'/2)*(V'/2)*(W'/2) sum = 1 => sum U'V'W' = 8
    
    def calculate_energy(U, V, W):
        energy = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    diff = s - 8
                    energy += diff * diff
        return energy

    # Try ranks from 20 to 35
    for R in range(20, rank_school):
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        for trial in range(100):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            # Use -1, 0, 1 mostly
            U = [[random.randint(-1, 1) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(R)] for _ in range(m * p)]
            W = [[random.randint(-1, 1) for _ in range(R)] for _ in range(p * n)]
            
            current_energy = calculate_energy(U, V, W)
            if current_energy == 0:
                # Convert back to original scale (divide by 2)
                U_out = [[x / 2.0 for x in row] for row in U]
                V_out = [[x / 2.0 for x in row] for row in V]
                W_out = [[x / 2.0 for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # Simulated Annealing
            T = 10.0
            cooling = 0.99
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                # Pick a random matrix and element
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                elif mat_idx == 1:
                    mat = V
                else:
                    mat = W
                
                row = random.randint(0, len(mat) - 1)
                col = random.randint(0, R - 1)
                
                old_val = mat[row][col]
                # Change by +/- 1
                change = random.choice([-1, 1])
                new_val = old_val + change
                mat[row][col] = new_val
                
                new_energy = calculate_energy(U, V, W)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_energy = new_energy
                else:
                    # Revert
                    mat[row][col] = old_val
                    
                T *= cooling
                
                if current_energy == 0:
                    U_out = [[x / 2.0 for x in row] for row in U]
                    V_out = [[x / 2.0 for x in row] for row in V]
                    W_out = [[x / 2.0 for x in row] for row in W]
                    return {"U": U_out, "V": V_out, "W": W_out}

    return best_result
