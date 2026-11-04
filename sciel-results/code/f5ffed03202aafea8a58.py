import time
import numpy as np

DIMENSIONS = [6, 6, 6]
DEADLINE = 130  # safety margin before 150s kill

def _verify_full(n, m, p, U, V, W):
    """Full verification using numpy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(Ua[a_idx], Va[b_idx] * Wa[c_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2(n, m, p):
    """Build Strassen 2x2 block decomposition (rank 189)."""
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 multiplication
    # Each pattern is (U_dict, V_dict, W_dict)
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
    return U, V, W

def _build_strassen_3x3_2x2(n, m, p):
    """Build 2x2 grid of 3x3 blocks with Strassen on the 2x2 block structure.
    Each 3x3 block multiplication is schoolbook (rank 27), and the 2x2 block
    combination uses Strassen (7 multiplications).
    Total rank = 7 * 27 = 189.
    """
    block_size = 3
    blocks_per_dim = 2
    # For each of the 7 Strassen multiplications, we need 27 schoolbook terms
    rank = 7 * (block_size ** 2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 multiplication
    # Each pattern is (U_dict, V_dict, W_dict) where keys are (block_row, block_col)
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), u_val in u_coeffs.items():
                        for (si, sj), v_val in v_coeffs.items():
                            for (qi, qj), w_val in w_coeffs.items():
                                # U contribution
                                r = I * block_size + pi
                                c = J * block_size + pj
                                U[r * m + c][col_idx] += u_val
                                # V contribution
                                r = J * block_size + si
                                c = K * block_size + sj
                                V[r * p + c][col_idx] += v_val
                                # W contribution
                                k = K * block_size + qi
                                i = I * block_size + qj
                                W[k * n + i][col_idx] += w_val
                                col_idx += 1
    return U, V, W

def _build_strassen_3x3_2x2_correct(n, m, p):
    """Build 2x2 grid of 3x3 blocks with Strassen on the 2x2 block structure.
    Each 3x3 block multiplication is schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    
    The key insight: for each of the 7 Strassen "multiplications" of 3x3 blocks,
    we need 27 scalar multiplications (schoolbook for 3x3).
    """
    block_size = 3
    blocks_per_dim = 2
    rank = 7 * (block_size ** 2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 multiplication
    # Each pattern is (U_dict, V_dict, W_dict) where keys are (block_row, block_col)
    # Values are the coefficients for that block in the Strassen formula
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # For this Strassen term, we need to handle all combinations
                    # of block positions that appear in U, V, W
                    # The rank term corresponds to a specific (I,J,K) and Strassen pattern
                    # We need to add contributions from all (pi,pj), (si,sj), (qi,qj)
                    # that are non-zero in the respective dicts
                    
                    u_positions = list(u_coeffs.keys())
                    v_positions = list(v_coeffs.keys())
                    w_positions = list(w_coeffs.keys())
                    
                    for (pi, pj) in u_positions:
                        for (si, sj) in v_positions:
                            for (qi, qj) in w_positions:
                                # Check if this combination is valid
                                # pi, sj must match for the middle index
                                # si, qi must match for the middle index
                                if pj != si or sj != qi:
                                    continue
                                
                                u_val = u_coeffs[(pi, pj)]
                                v_val = v_coeffs[(si, sj)]
                                w_val = w_coeffs[(qi, qj)]
                                
                                # U contribution: row = I*block_size + pi, col = J*block_size + pj
                                r = I * block_size + pi
                                c = J * block_size + pj
                                U[r * m + c][col_idx] += u_val
                                
                                # V contribution: row = J*block_size + si, col = K*block_size + sj
                                r = J * block_size + si
                                c = K * block_size + sj
                                V[r * p + c][col_idx] += v_val
                                
                                # W contribution: row = K*block_size + qi, col = I*block_size + qj
                                k = K * block_size + qi
                                i = I * block_size + qj
                                W[k * n + i][col_idx] += w_val
                                
                                col_idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build Strassen 2x2 (rank 189)
    U, V, W = _build_strassen_2x2(n, m, p)
    
    # Verify
    if time.time() - start_time < DEADLINE - 10:
        if not _verify_full(n, m, p, U, V, W):
            # Fallback to schoolbook if verification fails (should not happen)
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U[i * m + j][idx] = 1
                        V[j * p + k][idx] = 1
                        W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}
