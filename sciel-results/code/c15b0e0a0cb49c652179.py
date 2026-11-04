"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]
DEADLINE = 140.0

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen terms for 2x2 * 2x2 -> 2x2
    # Each term: (U_factors, V_factors, W_factors)
    # U_factors: list of (row, col, coeff) for the 2x2 P matrix
    # V_factors: list of (row, col, coeff) for the 2x2 Q matrix
    # W_factors: list of (row, col, coeff) for the 2x2 R matrix
    
    strassen_terms = [
        # M1 = (P00+P11)*(Q00+Q11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (P10+P11)*Q00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = P00*(Q01-Q11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = P11*(Q10-Q00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (P00+P01)*Q11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (P10-P00)*(Q00+Q01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (P01-P11)*(Q10+Q11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col_idx = 0
    for source in range(2):  # 0 = left block, 1 = right block of A
        for c in range(3):  # column block of B
            for t in range(7):  # Strassen term
                u_facs, v_facs, w_facs = strassen_terms[t]
                
                # Map P (2x2) to U (8x?)
                # P[pi,pj] corresponds to A[pi, pj + 2*source]
                # U index: pi * 4 + (pj + 2*source)
                for pi, pj, coeff in u_facs:
                    U[pi * 4 + (pj + 2 * source), col_idx] += coeff
                
                # Map Q (2x2) to V (24x?)
                # Q[qi,qj] corresponds to B[qi + 2*source, 2*c + qj]
                # V index: (qi + 2*source) * 6 + (2*c + qj)
                for qi, qj, coeff in v_facs:
                    V[(qi + 2 * source) * 6 + (2 * c + qj), col_idx] += coeff
                
                # Map R (2x2) to W (12x?)
                # R[ri,rj] contributes to C[ri, 2*c + rj]
                # W index: (2*c + rj) * 2 + ri
                for ri, rj, coeff in w_facs:
                    W[(2 * c + rj) * 2 + ri, col_idx] += coeff
                
                col_idx += 1
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen, but safety guard
        pass
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
