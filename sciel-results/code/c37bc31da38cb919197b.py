"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_decomposition():
    """
    Build the full 4x4 * 4x6 -> 4x6 tensor decomposition.
    
    Uses 2x2 block structure:
    - A: 2x2 blocks of 2x2 (indices ai, bj in {0,1})
    - B: 2x3 blocks of 2x2 (indices bj in {0,1}, ck in {0,1,2})
    - C: 2x3 blocks of 2x2 (indices ai in {0,1}, ck in {0,1,2})
    
    For cols 0,1: use Strassen at block level (7 terms)
    For col 2: use 4 schoolbook terms
    
    Total rank: 11 * 7 = 77
    """
    sU, sV, sW = _strassen_2x2()
    
    # Define outer terms as (U_dict, V_dict, W_dict)
    # Each dict maps (block_row, block_col) -> coefficient
    outer_terms = []
    
    # 7 Strassen terms for cols 0,1
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for a in range(2):
            for b in range(2):
                coeff = sU[a*2+b, t]
                if coeff != 0:
                    U_dict[(a,b)] = U_dict.get((a,b), 0) + coeff
        for b in range(2):
            for c in range(2):
                coeff = sV[b*2+c, t]
                if coeff != 0:
                    V_dict[(b,c)] = V_dict.get((b,c), 0) + coeff
        for a in range(2):
            for c in range(2):
                coeff = sW[c*2+a, t]
                if coeff != 0:
                    W_dict[(a,c)] = W_dict.get((a,c), 0) + coeff
        outer_terms.append((U_dict, V_dict, W_dict))
    
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            U_dict = {(a,b): 1}
            V_dict = {(b,2): 1}
            W_dict = {(a,2): 1}
            outer_terms.append((U_dict, V_dict, W_dict))
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, (U_dict, V_dict, W_dict) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            for (ai, bj), coeff_outer in U_dict.items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            for (bj, ck), coeff_outer in V_dict.items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            for (ai, ck), coeff_outer in W_dict.items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    return U, V, W


def construct():
    U, V, W = _build_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
