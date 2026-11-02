import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute valid indices for the full problem
    valid_indices_full = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices_full.add((a, b, c))

    # Build the full tensor as a float array for verification if needed
    T_full = np.zeros((n*m, m*p, p*n), dtype=float)
    for (a, b, c) in valid_indices_full:
        T_full[a, b, c] = 1.0

    def verify_float(U, V, W, rank, valid_indices, T_target):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Vectorized contraction
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check valid
        for (a, b, c) in valid_indices:
            if abs(T_recon[a, b, c] - T_target[a, b, c]) > 1e-6:
                return False
        
        # Check invalid
        mask = np.zeros(T_recon.shape, dtype=bool)
        for (a, b, c) in valid_indices:
            mask[a, b, c] = True
        invalid_vals = T_recon[~mask]
        if np.any(np.abs(invalid_vals) > 1e-6):
            return False
        return True

    def verify_exact(U, V, W, rank, valid_indices):
        Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(len(U))]
        Vf = [[Fraction(V[i][r]) for r in range(rank)] for i in range(len(V))]
        Wf = [[Fraction(W[i][r]) for r in range(rank)] for i in range(len(W))]
        
        for (a, b, c) in valid_indices:
            s = Fraction(0)
            for r in range(rank):
                s += Uf[a][r] * Vf[b][r] * Wf[c][r]
            if s != 1:
                return False
        
        # Check invalids
        # We need to iterate over all a,b,c not in valid_indices
        # To speed up, we can just check the ones that are "close" or all?
        # For exactness, we must check all.
        # But for small sub-problems, it's fine.
        for a in range(len(U)):
            for b in range(len(V)):
                for c in range(len(W)):
                    if (a, b, c) in valid_indices:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if s != 0:
                        return False
        return True

    def search_decomp(n_sub, m_sub, p_sub, target_rank, entries, time_limit, seed=42):
        """
        Search for a decomposition of M_{n_sub, m_sub, p_sub} with the given rank.
        Returns (U, V, W) or None.
        """
        random.seed(seed)
        start = time.time()
        
        valid_indices_sub = set()
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    valid_indices_sub.add((a, b, c))
        
        T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub), dtype=float)
        for (a, b, c) in valid_indices_sub:
            T_sub[a, b, c] = 1.0
            
        num_iters = 50000 # High number, will be limited by time
        for _ in range(num_iters):
            if time.time() > start + time_limit:
                break
                
            U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n_sub*m_sub)]
            V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m_sub*p_sub)]
            W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p_sub*n_sub)]
            
            if verify_float(U, V, W, target_rank, valid_indices_sub, T_sub):
                if verify_exact(U, V, W, target_rank, valid_indices_sub):
                    return U, V, W
                    
        return None

    # Strategy: Block Decomposition
    # Split B (3x7) into B1 (3x3) and B2 (3x4)
    # M_{2,3,3} + M_{2,3,4}
    
    best_rank = rank_school
    best_U = [[0]*rank_school for _ in range(n*m)]
    best_V = [[0]*rank_school for _ in range(m*p)]
    best_W = [[0]*rank_school for _ in range(p*n)]
    
    # Initialize with Schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                best_U[a][col] = 1
                best_V[b][col] = 1
                best_W[c][col] = 1
                col += 1

    # Try to find better decomposition for M_{2,3,3}
    # Known rank is 13. Try 13, 14, 15.
    U_sub = None
    V_sub = None
    W_sub = None
    rank_sub = None
    
    # Time budget for sub-search
    time_left = deadline - time.time()
    time_per_rank = max(5.0, time_left / 10.0)
    
    for r_sub in range(13, 16):
        if time.time() > deadline - 30:
            break
        res = search_decomp(2, 3, 3, r_sub, [-1, 0, 1], time_per_rank, seed=100+r_sub)
        if res:
            U_sub, V_sub, W_sub = res
            rank_sub = r_sub
            break
            
    # Try to find better decomposition for M_{2,3,4}
    # Known rank is likely 16 or 17. Try 16, 17, 18.
    U_sub2 = None
    V_sub2 = None
    W_sub2 = None
    rank_sub2 = None
    
    time_left = deadline - time.time()
    time_per_rank2 = max(5.0, time_left / 10.0)
    
    for r_sub2 in range(16, 19):
        if time.time() > deadline - 10:
            break
        res = search_decomp(2, 3, 4, r_sub2, [-1, 0, 1], time_per_rank2, seed=200+r_sub2)
        if res:
            U_sub2, V_sub2, W_sub2 = res
            rank_sub2 = r_sub2
            break

    # Combine
    if U_sub is not None:
        total_rank = rank_sub + (rank_sub2 if U_sub2 is not None else 24)
        if total_rank < best_rank:
            # Build the combined matrices
            U_new = [[0]*total_rank for _ in range(n*m)]
            V_new = [[0]*total_rank for _ in range(m*p)]
            W_new = [[0]*total_rank for _ in range(p*n)]
            
            # Part 1: M_{2,3,3} (cols 0-2 of B)
            for r in range(rank_sub):
                for i in range(2):
                    for j in range(3):
                        a = i*3 + j
                        U_new[a][r] = U_sub[a][r]
                        
                        # V mapping: b_full = j*7 + k. Here k is 0,1,2.
                        # b_sub = j*3 + k.
                        for k in range(3):
                            b_full = j*7 + k
                            b_sub = j*3 + k
                            V_new[b_full][r] = V_sub[b_sub][r]
                            
                            # W mapping: c_full = k*2 + i. Here k is 0,1,2.
                            # c_sub = k*2 + i.
                            c_full = k*2 + i
                            c_sub = k*2 + i
                            W_new[c_full][r] = W_sub[c_sub][r]
                            
            # Part 2: M_{2,3,4} (cols 3-6 of B)
            offset = rank_sub
            if U_sub2 is not None:
                for r in range(rank_sub2):
                    for i in range(2):
                        for j in range(3):
                            a = i*3 + j
                            U_new[a][offset + r] = U_sub2[a][r]
                            
                            # V mapping: b_full = j*7 + k. Here k is 3,4,5,6.
                            # In sub-problem, k' is 0,1,2,3.
                            # b_sub2 = j*4 + k'.
                            for k_prime in range(4):
                                k_full = k_prime + 3
                                b_full = j*7 + k_full
                                b_sub2 = j*4 + k_prime
                                V_new[b_full][offset + r] = V_sub2[b_sub2][r]
                                
                                # W mapping: c_full = k_full*2 + i.
                                # c_sub2 = k_prime*2 + i.
                                c_full = k_full*2 + i
                                c_sub2 = k_prime*2 + i
                                W_new[c_full][offset + r] = W_sub2[c_sub2][r]
            else:
                # Use schoolbook for the second part
                col = 0
                for i in range(2):
                    for j in range(3):
                        for k in range(3, 7):
                            a = i*3 + j
                            b = j*7 + k
                            c = k*2 + i
                            U_new[a][offset + col] = 1
                            V_new[b][offset + col] = 1
                            W_new[c][offset + col] = 1
                            col += 1
                            
            # Verify the combined solution
            if verify_float(U_new, V_new, W_new, total_rank, valid_indices_full, T_full):
                if verify_exact(U_new, V_new, W_new, total_rank, valid_indices_full):
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    best_rank = total_rank

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
