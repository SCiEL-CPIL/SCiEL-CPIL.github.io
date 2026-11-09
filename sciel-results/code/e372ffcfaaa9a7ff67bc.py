"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S = [
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),
        ({(2,1),(3,1)}, {(0,1)}, {(1,1),(3,-1)}),
        ({(0,1)}, {(1,1),(3,-1)}, {(2,1),(3,1)}),
        ({(3,1)}, {(2,1),(0,-1)}, {(0,1),(1,1)}),
        ({(0,1),(1,1)}, {(3,1)}, {(0,-1),(2,1)}),
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,1)}),
    ]
    
    j_offsets = [0, 2]
    k_offsets = [0, 2]
    
    col = 0
    for j_off in j_offsets:
        for k_off in k_offsets:
            for u_terms, v_terms, w_terms in S:
                for (li, coeff) in u_terms:
                    i, j = li // 2, li % 2
                    gi, gj = i, j + j_off
                    U[gi * m + gj][col] += coeff
                
                for (li, coeff) in v_terms:
                    j, k = li // 2, li % 2
                    gj, gk = j + j_off, k + k_off
                    V[gj * p + gk][col] += coeff
                
                for (li, coeff) in w_terms:
                    k, i = li // 2, li % 2
                    gk, gi = k + k_off, i
                    W[gk * n + gi][col] += coeff
                
                col += 1
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_rank12_sub():
    """
    Try to find a rank 12 decomposition for M_{2,4,2} with half-integer entries.
    This corresponds to integer entries U', V', W' satisfying sum U'V'W' = 8.
    Uses structured patterns based on Strassen-like constructions.
    """
    n_sub, m_sub, p_sub = 2, 4, 2
    target_rank = 12
    
    # Try structured patterns: use pairs of columns that mirror each other
    # Pattern: for each of 6 "Strassen-like" terms, have 2 columns (one for each i)
    # Try various structured U, V patterns
    
    def try_pattern(U_prime, V_prime):
        """Given U', V', solve for W' and check if integer solution exists."""
        A = np.zeros((n_sub * p_sub, n_sub * p_sub * target_rank))
        b = np.full(n_sub * p_sub, 8.0)
        
        for i in range(n_sub):
            for k in range(p_sub):
                eq_idx = i * p_sub + k
                for r in range(target_rank):
                    coeff = 0
                    for j in range(m_sub):
                        coeff += U_prime[i * m_sub + j, r] * V_prime[j * p_sub + k, r]
                    if coeff != 0:
                        w_idx = (k * n_sub + i) * target_rank + r
                        A[eq_idx, w_idx] = coeff
        
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        W_prime_sol = W_flat.reshape((n_sub * p_sub, target_rank))
        
        if not np.all(np.abs(W_prime_sol - np.round(W_prime_sol)) < 1e-6):
            return None
        
        W_prime_int = np.round(W_prime_sol).astype(int)
        
        # Verify
        for i in range(n_sub):
            for k in range(p_sub):
                s_val = 0
                for r in range(target_rank):
                    s_val += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime_int[k*n_sub+i, r]
                    # This is wrong, let me fix
                # Actually need to check all (i,j,k) triples
        # Full verification
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s_val = 0
                    for r in range(target_rank):
                        s_val += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime_int[k*n_sub+i, r]
                    if s_val != 8:
                        return None
        
        return W_prime_int
    
    # Structured approach: try to build from known small matrix multiplication decompositions
    # M_{2,4,2} can be seen as 2 copies of M_{1,4,2} but that's not right
    # Actually M_{2,4,2}: multiply 2x4 by 4x2 to get 2x2
    
    # Try: use the fact that this is equivalent to computing 4 dot products of length 4
    # Each output element (i,k) = sum_j A[i,j] B[j,k]
    
    # Strategy: try rank 12 with specific structures
    # Pattern 1: symmetric pairs
    # For 6 base terms, each term contributes to both i=0 and i=1 outputs
    
    for attempt in range(5000):
        U_prime = np.zeros((n_sub * m_sub, target_rank), dtype=int)
        V_prime = np.zeros((m_sub * p_sub, target_rank), dtype=int)
        
        if attempt < 1000:
            # Very structured: 6 pairs of columns
            for t in range(6):
                r1 = 2*t
                r2 = 2*t+1
                # Term t: U has entries for i=0, i=1; V has entries for j,k
                # Try: U[0,j,r1] and U[1,j,r2] are related
                for j in range(m_sub):
                    U_prime[0*m_sub+j, r1] = random.choice([-2,-1,1,2])
                    U_prime[1*m_sub+j, r2] = random.choice([-2,-1,1,2])
                for j in range(m_sub):
                    for k in range(p_sub):
                        V_prime[j*p_sub+k, r1] = random.choice([-2,-1,1,2])
                        V_prime[j*p_sub+k, r2] = random.choice([-2,-1,1,2])
        elif attempt < 3000:
            # Random sparse
            for r in range(target_rank):
                nz_u = random.sample(range(n_sub * m_sub), random.randint(1,3))
                for idx in nz_u:
                    U_prime[idx, r] = random.choice([-2, -1, 1, 2])
                nz_v = random.sample(range(m_sub * p_sub), random.randint(1,3))
                for idx in nz_v:
                    V_prime[idx, r] = random.choice([-2, -1, 1, 2])
        else:
            # Mixed: some structured, some random
            for r in range(target_rank):
                if r % 2 == 0:
                    nz_u = random.sample(range(0, m_sub), 2)
                    for idx in nz_u:
                        U_prime[idx, r] = random.choice([-2, -1, 1, 2])
                else:
                    nz_u = random.sample(range(m_sub, n_sub*m_sub), 2)
                    for idx in nz_u:
                        U_prime[idx, r] = random.choice([-2, -1, 1, 2])
                nz_v = random.sample(range(m_sub * p_sub), random.randint(2,4))
                for idx in nz_v:
                    V_prime[idx, r] = random.choice([-2, -1, 1, 2])
        
        W_prime_int = try_pattern(U_prime, V_prime)
        if W_prime_int is not None:
            U_sol = (U_prime / 2.0).tolist()
            V_sol = (V_prime / 2.0).tolist()
            W_sol = (W_prime_int / 2.0).tolist()
            return U_sol, V_sol, W_sol
            
    return None

def _try_find_rank14_sub():
    """Try rank 14 for M_{2,4,2} as a fallback."""
    n_sub, m_sub, p_sub = 2, 4, 2
    target_rank = 14
    
    for _ in range(1000):
        U_prime = np.zeros((n_sub * m_sub, target_rank), dtype=int)
        V_prime = np.zeros((m_sub * p_sub, target_rank), dtype=int)
        
        for r in range(target_rank):
            nz_u = random.sample(range(n_sub * m_sub), random.randint(1,3))
            for idx in nz_u:
                U_prime[idx, r] = random.choice([-2, -1, 1, 2])
            nz_v = random.sample(range(m_sub * p_sub), random.randint(1,3))
            for idx in nz_v:
                V_prime[idx, r] = random.choice([-2, -1, 1, 2])
        
        A = np.zeros((n_sub * p_sub, n_sub * p_sub * target_rank))
        b = np.full(n_sub * p_sub, 8.0)
        
        for i in range(n_sub):
            for k in range(p_sub):
                eq_idx = i * p_sub + k
                for r in range(target_rank):
                    coeff = 0
                    for j in range(m_sub):
                        coeff += U_prime[i * m_sub + j, r] * V_prime[j * p_sub + k, r]
                    if coeff != 0:
                        w_idx = (k * n_sub + i) * target_rank + r
                        A[eq_idx, w_idx] = coeff
        
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        W_prime_sol = W_flat.reshape((n_sub * p_sub, target_rank))
        
        if not np.all(np.abs(W_prime_sol - np.round(W_prime_sol)) < 1e-6):
            continue
        
        W_prime_int = np.round(W_prime_sol).astype(int)
        
        valid = True
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s_val = 0
                    for r in range(target_rank):
                        s_val += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime_int[k*n_sub+i, r]
                    if s_val != 8:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            U_sol = (U_prime / 2.0).tolist()
            V_sol = (V_prime / 2.0).tolist()
            W_sol = (W_prime_int / 2.0).tolist()
            return U_sol, V_sol, W_sol
            
    return None

def construct():
    start = time.time()
    deadline = start + 145.0
    
    result = _build_rank28()
    best_rank = len(result["U"][0])
    
    # Try to find a rank-12 sub-solution for rank-24 full solution
    if time.time() < deadline - 25:
        sub_sol = _try_find_rank12_sub()
        if sub_sol:
            U_sub, V_sub, W_sub = sub_sol
            # Construct full 2x4x4 solution with rank 24
            U_full = [[0] * 24 for _ in range(8)]
            V_full = [[0] * 24 for _ in range(16)]
            W_full = [[0] * 24 for _ in range(8)]
            
            # k=0,1 block (columns 0-11)
            for r in range(12):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+k][r] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[k*2+i][r] = W_sub[k*2+i][r]
                        
            # k=2,3 block (columns 12-23)
            for r in range(12):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r+12] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+2+k][r+12] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(2+k)*2+i][r+12] = W_sub[k*2+i][r]
            
            if _verify_decomposition(U_full, V_full, W_full):
                result = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 24
    
    # Try rank-14 sub-solution for rank-28 full solution (same as current, but different structure)
    # This won't improve rank, but might find a valid solution if rank-28 build fails
    if best_rank > 28 and time.time() < deadline - 10:
        sub_sol14 = _try_find_rank14_sub()
        if sub_sol14:
            U_sub, V_sub, W_sub = sub_sol14
            U_full = [[0] * 28 for _ in range(8)]
            V_full = [[0] * 28 for _ in range(16)]
            W_full = [[0] * 28 for _ in range(8)]
            
            for r in range(14):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+k][r] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[k*2+i][r] = W_sub[k*2+i][r]
                        
            for r in range(14):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r+14] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+2+k][r+14] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(2+k)*2+i][r+14] = W_sub[k*2+i][r]
            
            if _verify_decomposition(U_full, V_full, W_full):
                result = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 28
    
    return result
# EVOLVE-BLOCK-END
