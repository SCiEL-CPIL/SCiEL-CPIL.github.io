"""
Matrix multiplication 4x4x7 tensor decomposition.
Uses Strassen-8 (rank 49) for 4x4x4 block and schoolbook (rank 48) for 4x4x3 block.
Total rank: 97.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_4x4x4_decomposition():
    """
    Return the rank-49 Strassen-8 decomposition for 4x4x4 matrix multiplication.
    U: 16 x 49, V: 16 x 49, W: 16 x 49
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    For 4x4x4: n=m=p=4, so U is 16x49, V is 16x49, W is 16x49.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen-8 algorithm for 4x4x4
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # We use the 8 multiplications of Strassen applied to 2x2 blocks
    # where each "element" is actually a 2x2 matrix.
    
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2
    # Let B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A*B, C = [[C11, C12], [C21, C22]]
    
    # The 8 Strassen multiplications for 2x2 (of 2x2 blocks):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # M8 = A11 * A11  (or some other; actually standard Strassen uses 7, but for 4x4x4 we need 8)
    
    # Actually, the known rank-49 algorithm for 4x4x4 is based on 7 multiplications
    # of 2x2 matrices (each being a rank-7 decomposition), giving 7*7=49.
    # Let me implement this properly.
    
    # The 2x2x2 Strassen decomposition has rank 7.
    # For 4x4x4, we view it as 2x2 block multiplication where each block is 2x2.
    # The tensor for 4x4x4 is the "matrix multiplication tensor" for 2x2 matrices
    # of 2x2 blocks. The rank is 7*7 = 49.
    
    # The 2x2x2 Strassen coefficients (for A, B, C each 2x2):
    # M1: A coeffs [1,0,0,1], B coeffs [1,0,0,1], C coeffs [1,0,0,1]
    # M2: A coeffs [0,0,1,1], B coeffs [1,0,0,0], C coeffs [0,0,1,-1]
    # M3: A coeffs [1,0,0,0], B coeffs [0,1,0,-1], C coeffs [0,1,0,1]
    # M4: A coeffs [0,0,0,1], B coeffs [0,0,1,-1], C coeffs [1,0,1,0]
    # M5: A coeffs [1,1,0,0], B coeffs [0,0,0,1], C coeffs [-1,1,0,0]
    # M6: A coeffs [1,0,-1,0], B coeffs [1,1,0,0], C coeffs [0,0,0,1]
    # M7: A coeffs [0,1,0,-1], B coeffs [0,0,1,1], C coeffs [1,0,0,0]
    
    # In 1D indexing for 2x2: A00=0, A01=1, A10=2, A11=3
    # Similarly for B and C.
    
    # For the 4x4x4 case, we need to "inflate" each 2x2 coefficient into a 2x2 block.
    # The 4x4 matrix A has indices (i,j) for i,j in 0..3.
    # Block A11 = A[0:2, 0:2], indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Block A12 = A[0:2, 2:4], indices: (0,2)=4, (0,3)=5, (1,2)=6, (1,3)=7
    # Block A21 = A[2:4, 0:2], indices: (2,0)=8, (2,1)=9, (3,0)=10, (3,1)=11
    # Block A22 = A[2:4, 2:4], indices: (2,2)=12, (2,3)=13, (3,2)=14, (3,3)=15
    
    # In 1D: U index = i*4+j
    # A11: 0,1,2,3
    # A12: 4,5,6,7
    # A21: 8,9,10,11
    # A22: 12,13,14,15
    
    # Similarly for B (same indexing) and C (same indexing).
    
    # The 7 Strassen multiplications for the 2x2 "meta-matrix":
    # Each meta-multiplication involves two 2x2 "elements" (which are 2x2 matrices).
    # For each pair of 2x2 matrices being multiplied, we use the rank-7 Strassen.
    
    # So the total decomposition has 7 (meta-level) * 7 (inner-level) = 49 terms.
    
    # Let me define the 2x2x2 Strassen coefficients as a list of 7 tuples
    # (U_coeffs, V_coeffs, W_coeffs) where each is a dict from 1D index to coeff.
    
    # Strassen 2x2x2 coefficients:
    # For each of the 7 multiplications, we have:
    # U: dict from A_index to coefficient
    # V: dict from B_index to coefficient  
    # W: dict from C_index to coefficient
    
    strassen_2x2 = [
        # M1: (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2: (A21+A22)*B11 -> C21 += 1, C22 += -1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # M3: A11*(B12-B22) -> C12 += 1, C22 += 1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # M4: A22*(B21-B11) -> C11 += 1, C21 += 1
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # M5: (A11+A12)*B22 -> C11 += -1, C12 += 1
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6: (A21-A11)*(B11+B12) -> C22 += 1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7: (A12-A22)*(B21+B22) -> C11 += 1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # Now, for the 4x4x4 decomposition, each "element" in the meta-level
    # is a 2x2 matrix. When we multiply two such elements (using Strassen 2x2x2),
    # we need to expand each element into its 2x2 sub-elements.
    
    # The meta-level has 4 elements for A: A11, A12, A21, A22
    # In the 4x4 matrix, these correspond to blocks of size 2x2.
    # Block A11: rows 0-1, cols 0-1 -> 1D indices 0,1,2,3
    # Block A12: rows 0-1, cols 2-3 -> 1D indices 4,5,6,7
    # Block A21: rows 2-3, cols 0-1 -> 1D indices 8,9,10,11
    # Block A22: rows 2-3, cols 2-3 -> 1D indices 12,13,14,15
    
    # Similarly for B and C.
    
    # The meta-level indices:
    # meta_A: 0=A11, 1=A12, 2=A21, 3=A22
    # meta_B: 0=B11, 1=B12, 2=B21, 3=B22
    # meta_C: 0=C11, 1=C12, 2=C21, 3=C22
    
    # Block-to-1D index mapping:
    def meta_to_indices(block_idx, size=4):
        """Convert meta index (0-3) to the 4 1D indices in the 4x4 matrix."""
        if block_idx == 0:  # top-left
            return [0, 1, 2, 3]
        elif block_idx == 1:  # top-right
            return [4, 5, 6, 7]
        elif block_idx == 2:  # bottom-left
            return [8, 9, 10, 11]
        elif block_idx == 3:  # bottom-right
            return [12, 13, 14, 15]
    
    # Now construct the 49-term decomposition:
    # For each meta-level multiplication r_meta (0..6), and each inner-level
    # Strassen multiplication r_inner (0..6), we have a term.
    
    term = 0
    for r_meta in range(7):
        U_meta, V_meta, W_meta = strassen_2x2[r_meta]
        for r_inner in range(7):
            U_inner, V_inner, W_inner = strassen_2x2[r_inner]
            
            # For this term, the U coefficients come from:
            # For each meta_A block a_meta in U_meta:
            #   For each 1D index a_idx in meta_to_indices(a_meta):
            #     U[a_idx, term] += U_meta[a_meta] * U_inner[local_idx]
            # where local_idx is the position within the 2x2 block.
            
            # Similarly for V and W.
            
            for a_meta, u_coeff in U_meta.items():
                a_indices = meta_to_indices(a_meta)
                for local_idx, a_idx in enumerate(a_indices):
                    if local_idx in U_inner:
                        U[a_idx, term] += u_coeff * U_inner[local_idx]
            
            for b_meta, v_coeff in V_meta.items():
                b_indices = meta_to_indices(b_meta)
                for local_idx, b_idx in enumerate(b_indices):
                    if local_idx in V_inner:
                        V[b_idx, term] += v_coeff * V_inner[local_idx]
            
            for c_meta, w_coeff in W_meta.items():
                c_indices = meta_to_indices(c_meta)
                for local_idx, c_idx in enumerate(c_indices):
                    if local_idx in W_inner:
                        W[c_idx, term] += w_coeff * W_inner[local_idx]
            
            term += 1
    
    return U, V, W, 49

def _schoolbook_4x4x3_decomposition():
    """
    Return the schoolbook decomposition for 4x4x3 matrix multiplication.
    U: 16 x 48, V: 12 x 48, W: 12 x 48
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    For n=4, m=4, p=3: U is 16x48, V is 12x48, W is 12x48.
    """
    n = m = p = 3  # Wait, p=3 for the sub-problem
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W, rank

def _construct_4x4x7_decomposition():
    """
    Construct a rank-97 decomposition for 4x4x7 matrix multiplication.
    Split B into B1 (4x4) and B2 (4x3).
    A*B = A*B1 + A*B2 (concatenated)
    Use Strassen rank-49 for 4x4x4 and schoolbook rank-48 for 4x4x3.
    Total rank: 49 + 48 = 97.
    
    Global indices:
    U: 16 x 97 (i*4+j for i,j in 0..3)
    V: 28 x 97 (j*7+k for j in 0..3, k in 0..6)
    W: 28 x 97 (k*4+i for k in 0..6, i in 0..3)
    
    For the 4x4x4 part:
    U_44: 16 x 49
    V_44: 16 x 49 (rows j*4+k for j in 0..3, k in 0..3 -> global V rows j*7+k)
    W_44: 16 x 49 (rows k*4+i for k in 0..3, i in 0..3 -> global W rows k*4+i)
    
    For the 4x4x3 part:
    U_43: 16 x 48
    V_43: 12 x 48 (rows j*3+k for j in 0..3, k in 0..2 -> global V rows j*7+(k+4))
    W_43: 12 x 48 (rows k*4+i for k in 0..2, i in 0..3 -> global W rows (k+4)*4+i)
    """
    n, m, p = 4, 4, 7
    nm, mp, pn = n*m, m*p, p*n  # 16, 28, 28
    
    # Get 4x4x4 Strassen decomposition
    U_44, V_44, W_44, r1 = _strassen_4x4x4_decomposition()
    # U_44: 16x49, V_44: 16x49, W_44: 16x49
    
    # Get 4x4x3 schoolbook decomposition
    U_43, V_43, W_43, r2 = _schoolbook_4x4x3_decomposition()
    # U_43: 16x48, V_43: 12x48, W_43: 12x48
    
    total_rank = r1 + r2  # 97
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Fill in the 4x4x4 part (columns 0..48)
    U[:, 0:r1] = U_44
    # V_44 rows: j*4+k for j in 0..3, k in 0..3
    # Global V row: j*7+k
    for j in range(4):
        for k in range(4):
            local_idx = j*4 + k
            global_idx = j*7 + k
            V[global_idx, 0:r1] = V_44[local_idx, :]
    # W_44 rows: k*4+i for k in 0..3, i in 0..3
    # Global W row: k*4+i (same indexing since n=4)
    for k in range(4):
        for i in range(4):
            local_idx = k*4 + i
            global_idx = k*4 + i
            W[global_idx, 0:r1] = W_44[local_idx, :]
    
    # Fill in the 4x4x3 part (columns 49..96)
    U[:, r1:total_rank] = U_43
    # V_43 rows: j*3+k for j in 0..3, k in 0..2
    # Global V row: j*7+(k+4)
    for j in range(4):
        for k in range(3):
            local_idx = j*3 + k
            global_idx = j*7 + (k+4)
            V[global_idx, r1:total_rank] = V_43[local_idx, :]
    # W_43 rows: k*4+i for k in 0..2, i in 0..3
    # Global W row: (k+4)*4+i
    for k in range(3):
        for i in range(4):
            local_idx = k*4 + i
            global_idx = (k+4)*4 + i
            W[global_idx, r1:total_rank] = W_43[local_idx, :]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify that the factors decompose the tensor."""
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0
    
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    return np.linalg.norm(T_recon - T_target) < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    result = _construct_4x4x7_decomposition()
    
    # Convert to lists as required
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = U.shape[1]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
