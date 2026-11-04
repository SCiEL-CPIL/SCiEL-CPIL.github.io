import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(candidate):
        """Exact rational validation."""
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True
    
    def build_strassen_2x3_8():
        """
        Build a rank-44 decomposition using Strassen for 2x2 sub-blocks.
        
        Strategy:
        - A is 2x3, B is 3x8
        - Split A into A1 (2x2) and A2 (2x1)
        - Split B into B1 (2x8) and B2 (1x8)
        - C = A1*B1 + A2*B2
        - For A1*B1: use Strassen on 2x2 * 2x2 blocks, 4 blocks of 2 columns each
        - For A2*B2: schoolbook, 2*8 = 16 terms
        - Total: 4*7 + 16 = 44 terms
        """
        # Strassen 2x2x2 matrices
        # U indices: (i, a) where i in {0,1}, a in {0,1} -> flat index i*2+a
        # V indices: (a, k) where a in {0,1}, k in {0,1} -> flat index a*2+k
        # W indices: (k, i) where k in {0,1}, i in {0,1} -> flat index k*2+i
        
        S_U = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, 1],   # M2
            [1, 0, 0, 0],   # M3
            [0, 0, 0, 1],   # M4
            [1, 1, 0, 0],   # M5
            [-1, 0, 1, 0],  # M6
            [0, 1, 0, -1]   # M7
        ]
        S_V = [
            [1, 0, 0, 1],   # M1
            [1, 0, 0, 0],   # M2
            [0, 1, 0, -1],  # M3
            [-1, 0, 1, 0],  # M4
            [0, 0, 0, 1],   # M5
            [1, 1, 0, 0],   # M6
            [0, 0, 1, 1]    # M7
        ]
        S_W = [
            [1, 0, 0, 1],   # M1
            [0, 1, 0, -1],  # M2
            [0, 0, 1, 1],   # M3
            [1, 1, 0, 0],   # M4
            [-1, 0, 1, 0],  # M5
            [0, 0, 0, 1],   # M6
            [1, 0, 0, 0]    # M7
        ]
        
        # Verify Strassen decomposition
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return None
        
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: A1 (2x2) * B1 (2x8)
        # B1 has rows 0,1. Cols 0..7.
        # Split B1 cols into 4 blocks of 2: (0,1), (2,3), (4,5), (6,7)
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                # U indices: i in {0,1}; a in {0,1} (global a = local a)
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li*2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val
                
                # V indices: a in {0,1} (global a = local a); k in {k0, k1}
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la*2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val
                
                # W indices: k in {k0, k1}; i in {0,1}
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk*2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val
                
                col += 1
                if time.time() > deadline:
                    return None
        
        # Part 2: A2 (2x1) * B2 (1x8)
        # A2 uses col 2 of A (global a=2).
        # B2 uses row 2 of B (global a=2).
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                if time.time() > deadline:
                    return None
        
        candidate = {"U": U, "V": V, "W": W}
        
        if validate(candidate):
            return candidate
        return None
    
    def try_reduce_rank(base_candidate, max_reduction=4):
        """
        Try to reduce rank by finding linear dependencies.
        This is a heuristic search - may not always succeed.
        """
        if time.time() > deadline:
            return None
        
        base_rank = len(base_candidate["U"][0])
        U = np.array(base_candidate["U"], dtype=float)
        V = np.array(base_candidate["V"], dtype=float)
        W = np.array(base_candidate["W"], dtype=float)
        
        # Build the tensor T from the decomposition
        # T[i, a, k] = sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]
        T = np.zeros((n, m, p))
        for c in range(base_rank):
            for i in range(n):
                for a in range(m):
                    for k in range(p):
                        T[i, a, k] += U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]
        
        # For each subset of columns, check if we can express the tensor
        # with fewer terms by finding a lower-rank approximation
        # This is NP-hard in general, so we use a greedy approach
        
        best_result = None
        for target_rank in range(base_rank - max_reduction, base_rank):
            if time.time() > deadline:
                break
            
            # Try random subsets of columns
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                # Select a random subset of columns
                cols = np.random.choice(base_rank, target_rank, replace=False)
                U_sub = U[:, cols]
                V_sub = V[:, cols]
                W_sub = W[:, cols]
                
                # Check if this subset still spans the tensor
                # We need to find coefficients such that:
                # T = sum_c alpha_c * (u_c outer v_c outer w_c)
                # This is a system of linear equations
                
                # Flatten the tensor
                T_flat = T.flatten()
                
                # Build the design matrix
                A = np.zeros((n*m*p, target_rank))
                for c in range(target_rank):
                    for i in range(n):
                        for a in range(m):
                            for k in range(p):
                                idx = i*m*p + a*p + k
                                A[idx, c] = U_sub[i*m+a, c] * V_sub[a*p+k, c] * W_sub[k*n+i, c]
                
                # Solve for coefficients
                try:
                    coeffs, residuals, rank_a, sv = np.linalg.lstsq(A, T_flat, rcond=None)
                except:
                    continue
                
                # Check if the solution is valid
                if residuals.size > 0 and residuals[0] < 1e-6:
                    # Verify exactness
                    T_calc = np.zeros((n, m, p))
                    for c in range(target_rank):
                        for i in range(n):
                            for a in range(m):
                                for k in range(p):
                                    T_calc[i, a, k] += coeffs[c] * U_sub[i*m+a, c] * V_sub[a*p+k, c] * W_sub[k*n+i, c]
                    
                    if np.allclose(T_calc, T, atol=1e-9):
                        # Round to half-integers
                        U_r = np.round(U_sub * 2) / 2
                        V_r = np.round(V_sub * 2) / 2
                        W_r = np.round(W_sub * 2) / 2
                        
                        # Re-verify with rounded values
                        T_calc2 = np.zeros((n, m, p))
                        for c in range(target_rank):
                            for i in range(n):
                                for a in range(m):
                                    for k in range(p):
                                        T_calc2[i, a, k] += U_r[i*m+a, c] * V_r[a*p+k, c] * W_r[k*n+i, c]
                        
                        if np.allclose(T_calc2, T, atol=1e-9):
                            result = {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                            if validate(result):
                                best_result = result
                                return best_result
        
        return best_result
    
    # First, build the rank-44 Strassen-based decomposition
    base = build_strassen_2x3_8()
    
    if base is not None:
        # Try to reduce the rank further
        reduced = try_reduce_rank(base, max_reduction=4)
        if reduced is not None:
            return reduced
        return base
    
    # Fallback to schoolbook
    return get_schoolbook()
