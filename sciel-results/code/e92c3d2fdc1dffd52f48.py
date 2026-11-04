"""Tensor decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def _to_halfint_list(mat):
    """Convert numpy array to list of lists of ints or 0.5/2.5 etc."""
    # Multiply by 2 to check if all are integers
    doubled = mat * 2
    if np.allclose(doubled, np.round(doubled)):
        # Check if original has .5 values
        rounded = np.round(doubled)
        result = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = doubled[i, j]
                if abs(v - round(v)) < 1e-10:
                    result[i, j] = round(v) / 2.0
                    # Store as int if integer, else float
                    if abs(v - round(v)) < 1e-10 and v % 1 == 0:
                        result[i, j] = int(round(v / 2))
                    else:
                        result[i, j] = round(v) / 2.0
        # Convert to python list
        return result.tolist()
    else:
        # Fallback: just return as is (shouldn't happen for exact decomps)
        return mat.tolist()

def _strassen_222():
    """
    Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    A is 2x2, B is 2x2, C is 2x2.
    U indices: (i, j) -> i*2 + j
    V indices: (j, k) -> j*2 + k
    W indices: (k, i) -> k*2 + i  (trace convention W[k*n+i, r] with n=2)
    """
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    # Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # U: coefficients for A (2x2 = 4 entries)
    # U[(i,j), c]
    # c0: M1
    U[0, 0] = 1; U[3, 0] = 1   # A11, A22
    # c1: M2
    U[2, 1] = 1; U[3, 1] = 1   # A21, A22
    # c2: M3
    U[0, 2] = 1                # A11
    # c3: M4
    U[3, 3] = 1                # A22
    # c4: M5
    U[0, 4] = 1; U[1, 4] = 1   # A11, A12
    # c5: M6
    U[2, 5] = 1; U[0, 5] = -1  # A21, -A11
    # c6: M7
    U[1, 6] = 1; U[3, 6] = -1  # A12, -A22
    
    # V: coefficients for B (2x2 = 4 entries)
    # V[(j,k), c]
    # c0: M1
    V[0, 0] = 1; V[3, 0] = 1   # B11, B22
    # c1: M2
    V[0, 1] = 1                # B11
    # c2: M3
    V[2, 2] = 1; V[3, 2] = -1  # B21, -B22
    # c3: M4
    V[1, 3] = 1; V[0, 3] = -1  # B12, -B11
    # c4: M5
    V[3, 4] = 1                # B22
    # c5: M6
    V[0, 5] = 1; V[1, 5] = 1   # B11, B12
    # c6: M7
    V[2, 6] = 1; V[3, 6] = 1   # B21, B22
    
    # W: coefficients for C (2x2 = 4 entries), W[k*2 + i, c]
    # C11 = M1 + M4 - M5 + M7  -> W[0*2+0]=W[0], C12 = M3 + M5 -> W[0*2+1]=W[1]
    # C21 = M2 + M4            -> W[1*2+0]=W[2], C22 = M1 - M2 + M3 + M6 -> W[1*2+1]=W[3]
    
    # c0: M1 contributes to C11 and C22
    W[0, 0] = 1; W[3, 0] = 1
    # c1: M2 contributes to C21 and C22 (with -1)
    W[2, 1] = 1; W[3, 1] = -1
    # c2: M3 contributes to C12 and C22
    W[1, 2] = 1; W[3, 2] = 1
    # c3: M4 contributes to C11 and C21
    W[0, 3] = 1; W[2, 3] = 1
    # c4: M5 contributes to C11 (with -1) and C12
    W[0, 4] = -1; W[1, 4] = 1
    # c5: M6 contributes to C22
    W[3, 5] = 1
    # c6: M7 contributes to C11
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
    Uses block decomposition: split dimensions into 2s and 1s, use Strassen for 2x2x2 blocks,
    schoolbook for 1x1x1 blocks.
    """
    deadline = time.time() + 115.0  # Safety margin
    
    n, m, p = 3, 5, 6
    
    # Split n=3 into [2, 1], m=5 into [2, 2, 1], p=6 into [2, 2, 2]
    # We'll iterate over all combinations of blocks
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2]
    
    # Build the full decomposition by combining block decompositions
    # Total rank = sum over all block combinations of (rank of that block)
    # For a block of size (ni, mi, pi):
    #   if ni==2 and mi==2 and pi==2: rank 7
    #   else: rank ni*mi*pi (schoolbook)
    
    # Let's compute the total rank first
    total_rank = 0
    block_specs = []
    for ni in n_blocks:
        for mi in m_blocks:
            for pi in p_blocks:
                if ni == 2 and mi == 2 and pi == 2:
                    block_rank = 7
                else:
                    block_rank = ni * mi * pi
                total_rank += block_rank
                block_specs.append((ni, mi, pi, block_rank))
    
    # Allocate full matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    i_offset = 0
    for ni in n_blocks:
        j_offset = 0
        for mi in m_blocks:
            k_offset = 0
            for pi in p_blocks:
                # Get the decomposition for this block
                if ni == 2 and mi == 2 and pi == 2:
                    Ub, Vb, Wb = _strassen_222()
                else:
                    Ub, Vb, Wb = _schoolbook(ni, mi, pi)
                
                # Place into full matrices with correct offsets
                # U indices: (i, j) where i is in [i_offset, i_offset+ni), j is in [j_offset, j_offset+mi)
                for r in range(ni * mi):
                    i_local, j_local = divmod(r, mi)
                    i_global = i_offset + i_local
                    j_global = j_offset + j_local
                    for c in range(Ub.shape[1]):
                        U[i_global * m + j_global, col + c] = Ub[r, c]
                
                # V indices: (j, k) where j is in [j_offset, j_offset+mi), k is in [k_offset, k_offset+pi)
                for r in range(mi * pi):
                    j_local, k_local = divmod(r, pi)
                    j_global = j_offset + j_local
                    k_global = k_offset + k_local
                    for c in range(Vb.shape[1]):
                        V[j_global * p + k_global, col + c] = Vb[r, c]
                
                # W indices: (k, i) where k is in [k_offset, k_offset+pi), i is in [i_offset, i_offset+ni)
                # W[k*n + i, c]
                for r in range(pi * ni):
                    k_local, i_local = divmod(r, ni)
                    k_global = k_offset + k_local
                    i_global = i_offset + i_local
                    for c in range(Wb.shape[1]):
                        W[k_global * n + i_global, col + c] = Wb[r, c]
                
                col += Ub.shape[1]
                k_offset += pi
            j_offset += mi
        i_offset += ni
    
    # Convert to half-integer lists
    U_list = _to_halfint_list(U)
    V_list = _to_halfint_list(V)
    W_list = _to_halfint_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
