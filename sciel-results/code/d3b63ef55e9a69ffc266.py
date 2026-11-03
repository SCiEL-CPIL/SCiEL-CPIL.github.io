"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # A00 is U[0], A11 is U[3]
    # B00 is V[0], B11 is V[3]
    # C00 is W[0], C11 is W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # A10 is U[2], A11 is U[3]
    # B00 is V[0]
    # C10 is W[1], C11 is W[3]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # A00 is U[0]
    # B01 is V[1], B11 is V[3]
    # C01 is W[2], C11 is W[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # A11 is U[3]
    # B10 is V[2], B00 is V[0]
    # C00 is W[0], C10 is W[1]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # A00 is U[0], A01 is U[1]
    # B11 is V[3]
    # C00 is W[0], C01 is W[2]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # A10 is U[2], A00 is U[0]
    # B00 is V[0], B01 is V[1]
    # C11 is W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    # A01 is U[1], A11 is U[3]
    # B10 is V[2], B11 is V[3]
    # C00 is W[0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta_{j,k} is WRONG.
    Correct Identity for Matrix Mult (A n x m, B m x p -> C n x p):
    C[i, k] = sum_j A[i, j] B[j, k]
    Tensor T[i, j, k] = delta_{j, j'} is not right.
    
    The tensor for matrix multiplication M_{n,m,p} has entries T[i, j, k] = 1 if j is the index connecting A and B.
    Wait, the standard tensor definition for matrix multiplication:
    T is a tensor of order 3 with dimensions (n, m, p).
    T[i, j, k] = 1 if j is the "middle" index? No.
    
    Let's look at the constraint in the prompt:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    This implies:
    For fixed i, k, the sum over c of W_{ik} U_{ij} V_{j'k} equals delta_{j,j'}.
    This is exactly the condition that the matrices U, V, W define a decomposition of the identity tensor in the middle dimension?
    No, this is the condition for the *multiplication* tensor.
    Let A be n x m, B be m x p. C = A B.
    C_{ik} = sum_j A_{ij} B_{jk}.
    If we write A_{ij} U_{ij,c} ... no.
    
    The decomposition is:
    C_{ik} = sum_c W_{ik,c} (sum_j U_{ij,c} A_{ij}) (sum_j' V_{j'k,c} B_{j'k}) ? No.
    
    The standard bilinear complexity formulation:
    C_{ik} = sum_c M_c
    M_c = (sum_j U_{ij,c} A_{ij}) * (sum_j V_{j'k,c} B_{j'k})
    
    Expanding M_c:
    M_c = sum_j sum_j' U_{ij,c} V_{j'k,c} A_{ij} B_{j'k}
    
    So C_{ik} = sum_c sum_j sum_j' U_{ij,c} V_{j'k,c} A_{ij} B_{j'k}
    
    We require C_{ik} = sum_j A_{ij} B_{jk}.
    This means for all i, j, j', k:
    sum_c U_{ij,c} V_{j'k,c} W_{ik,c} = delta_{j, j'}
    
    Yes, this matches the prompt's constraint:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    # Optimization: use numpy broadcasting or loops?
    # n=2, m=4, p=7. Small enough for loops.
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, 40s buffer)
    
    # Total Rank Calculation
    # m=4 split into 2 blocks of 2.
    # p=7 split into 2, 2, 2, 1.
    # For each m-block, we have 3 blocks of 2x2x2 (rank 7) and 1 block of 2x2x1 (rank 4).
    # Rank per m-block = 3*7 + 4 = 25.
    # Total Rank = 2 * 25 = 50.
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # We iterate over m-blocks
    # m_blocks: [0, 2) and [2, 4)
    m_blocks = [(0, 2), (2, 4)]
    
    # p_blocks: [0, 2), [2, 4), [4, 6), [6, 7)
    # Types: 2x2x2, 2x2x2, 2x2x2, 2x2x1
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    # Pre-fetch sub-tensors to avoid repeated creation if needed, 
    # but they are small so creating is fine.
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        # j_sub is 0 to 1 within the block
        # i is 0 to 1 (n=2)
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
                # p_sub = 2
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
                # p_sub = 1
            
            # Place U
            # U_sub indices: i_sub (0..1), j_sub (0..1)
            # Global U indices: i (0..1), j (j_start..j_end-1)
            # U[i*m + j, col]
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            # Place V
            # V_sub indices: j_sub (0..1), k_sub (0..p_sub-1)
            # Global V indices: j (j_start..j_end-1), k (k_start..k_end-1)
            # V[j*p + k, col]
            p_sub = k_end - k_start
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            # Place W
            # W_sub indices: k_sub (0..p_sub-1), i (0..1)
            # Global W indices: k (k_start..k_end-1), i (0..1)
            # W[k*n + i, col]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    # Verify
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails (should not happen)
            r = n * m * p
            U_fb = np.zeros((n * m, r))
            V_fb = np.zeros((m * p, r))
            W_fb = np.zeros((p * n, r))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, column] = 1
                        V_fb[j * p + k, column] = 1
                        W_fb[k * n + i, column] = 1
                        column += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    else:
        # Time out, return best effort (which is the constructed one)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
