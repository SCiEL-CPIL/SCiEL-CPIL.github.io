"""
Matrix multiplication 3x3 by 3x4 rank minimization.
Combines random search with structured generation and exact integer arithmetic for W.
"""
import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    num_ik = n * p  # 12 pairs of (i, k)
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify(U, V, W, rank):
        """Exact verification using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    def find_sparse_w(m_vec, max_val=2):
        """
        Find a sparse integer vector w such that sum(m_vec[c] * w[c]) = 1.
        m_vec is a list of integers.
        Returns list of integers or None.
        """
        r = len(m_vec)
        # Try 1-term solutions
        for c in range(r):
            if m_vec[c] == 1:
                w = [0] * r
                w[c] = 1
                return w
            if m_vec[c] == -1:
                w = [0] * r
                w[c] = -1
                return w
        
        # Try 2-term solutions
        for c1 in range(r):
            if m_vec[c1] == 0: continue
            for c2 in range(r):
                if c1 == c2: continue
                if m_vec[c2] == 0: continue
                # m1*w1 + m2*w2 = 1
                # We can brute force small w1, w2
                for w1 in range(-max_val, max_val + 1):
                    if m_vec[c1] * w1 == 1:
                        w = [0] * r
                        w[c1] = w1
                        return w
                    if m_vec[c2] != 0:
                        rem = 1 - m_vec[c1] * w1
                        if rem % m_vec[c2] == 0:
                            w2 = rem // m_vec[c2]
                            if abs(w2) <= max_val:
                                w = [0] * r
                                w[c1] = w1
                                w[c2] = w2
                                return w
        return None

    def generate_random_uv(r, density=0.3, val_range=(-1, 1)):
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        for i in range(n * m):
            for c in range(r):
                if random.random() < density:
                    U[i][c] = random.randint(val_range[0], val_range[1])
        for j in range(m * p):
            for c in range(r):
                if random.random() < density:
                    V[j][c] = random.randint(val_range[0], val_range[1])
        return U, V

    def generate_structured_uv(r):
        """
        Generate U, V with more structure.
        Use modular arithmetic to create patterns.
        """
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        for c in range(r):
            # Pattern based on c
            seed = c
            for i in range(n):
                for j in range(m):
                    val = (i * 3 + j * 7 + seed * 5) % 7
                    if val < 3:
                        U[i * m + j][c] = 1
                    elif val < 5:
                        U[i * m + j][c] = -1
                    else:
                        U[i * m + j][c] = 0
            
            for k in range(p):
                for j in range(m):
                    val = (k * 5 + j * 11 + seed * 3) % 7
                    if val < 3:
                        V[j * p + k][c] = 1
                    elif val < 5:
                        V[j * p + k][c] = -1
                    else:
                        V[j * p + k][c] = 0
        return U, V

    # Search ranks from low to high
    # Theoretical lower bound is around 23-25. Schoolbook is 36.
    # We try a range of ranks.
    for target_rank in range(24, 36):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        attempts = 0
        max_attempts = 500 if target_rank < 30 else 200
        
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            
            # Alternate between random and structured
            if attempts % 3 == 0:
                U, V = generate_structured_uv(target_rank)
            else:
                # Vary density and value range
                density = 0.2 + 0.4 * (attempts % 5) / 5
                U, V = generate_random_uv(target_rank, density=density)
            
            # Compute M[i, k, c] = sum_j U[i, j, c] * V[j, k, c]
            # M is a tensor of size (n, p, r)
            M = [[[0] * target_rank for _ in range(p)] for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j][c] * V[j * p + k][c]
                        M[i][k][c] = s
            
            # Try to find W
            W_candidate = [[0] * target_rank for _ in range(p * n)]
            valid = True
            for i in range(n):
                for k in range(p):
                    m_vec = M[i][k]
                    # Filter out zeros to make search faster? No, find_sparse_w handles it.
                    w_sol = find_sparse_w(m_vec, max_val=3)
                    if w_sol is None:
                        valid = False
                        break
                    W_candidate[k * n + i] = w_sol
                if not valid:
                    break
            
            if valid:
                # Final verification
                if verify(U, V, W_candidate, target_rank):
                    best_result = {"U": U, "V": V, "W": W_candidate}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
