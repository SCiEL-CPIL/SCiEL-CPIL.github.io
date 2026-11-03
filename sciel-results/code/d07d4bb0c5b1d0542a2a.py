"""Matrix multiplication 3x3 x 3x5 tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W):
    """Verify that the decomposition is exact for 3x3 * 3x5 multiplication."""
    n, m, p = 3, 3, 5
    rank = len(U[0]) if U else 0
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For standard multiplication: T[i,j,k] = delta_{i,k}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 3, 3, 5
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    # Start with schoolbook (rank 45) as guaranteed valid fallback
    rank_school = nm * mp  # 45
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
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
    
    # Try to find lower-rank decompositions using structured search
    # Strategy: try ranks from 20 to 44, use random initialization + local search
    
    # For matrix multiplication 3x3 * 3x5, try known improved ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Try a range of ranks
    
    candidate_ranks = list(range(20, 45))  # Try ranks 20-44
    random.shuffle(candidate_ranks)
    candidate_ranks.sort()  # Try lower ranks first
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        n_attempts = 3
        for attempt in range(n_attempts):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random values
            U = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.gauss(0, 1) for _ in range(target_rank)] for _ in range(pn)]
            
            # Gradient descent to minimize sum of squared errors
            lr = 0.01
            for iteration in range(2000):
                if time.time() > deadline - 25:
                    break
                
                # Compute errors
                errors = {}
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0 if i == k else 0.0
                            actual = 0.0
                            for c in range(target_rank):
                                actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            err = actual - expected
                            errors[(i, j, k)] = err
                            max_err = max(max_err, abs(err))
                
                if max_err < 1e-8:
                    # Found exact solution
                    # Round to reduce floating point noise
                    for row in U:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx], 6)
                    for row in V:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx], 6)
                    for row in W:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx], 6)
                    
                    if _verify(U, V, W):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U, "V": V, "W": W}
                        return best_result
                    break
                
                # Compute gradients (simplified - only update U as example)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = errors[(i, j, k)]
                            for c in range(target_rank):
                                U[i * m + j][c] -= lr * err * V[j * p + k][c] * W[k * n + i][c]
                                V[j * p + k][c] -= lr * err * U[i * m + j][c] * W[k * n + i][c]
                                W[k * n + i][c] -= lr * err * U[i * m + j][c] * V[j * p + k][c]
                
                if iteration % 100 == 0 and max_err < 0.1:
                    # Try to converge further
                    lr = lr * 0.5
            
            # After gradient descent, check if close to exact
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if i == k else 0.0
                        actual = 0.0
                        for c in range(target_rank):
                            actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        max_err = max(max_err, abs(actual - expected))
            
            if max_err < 1e-6:
                # Round and verify
                for row in U:
                    for idx in range(len(row)):
                        row[idx] = round(row[idx], 8)
                for row in V:
                    for idx in range(len(row)):
                        row[idx] = round(row[idx], 8)
                for row in W:
                    for idx in range(len(row)):
                        row[idx] = round(row[idx], 8)
                
                if _verify(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U, "V": V, "W": W}
                    return best_result
    
    return best_result
