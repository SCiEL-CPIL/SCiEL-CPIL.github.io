"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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


def _round_to_half_int(arr, tol=1e-6):
    """Round each entry to the nearest half-integer; return rounded array and success flag."""
    rounded = np.round(arr * 2) / 2
    diff = np.abs(arr - rounded)
    return rounded, np.all(diff <= tol)


def _verify_fast(U, V, W, tol=1e-8):
    """Vectorized verification using einsum."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    For C = A * B where A, B are 2x2 matrices:
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    Strassen products:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b01)
    M7 = (a01 - a11) * (b10 + b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Returns U, V, W arrays of shape (4, 7), (4, 7), (4, 7) for the 2x2x2 case.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    
    # M1 = (a00 + a11) * (b00 + b11)
    # U: a00 -> U[0,0]=1, a11 -> U[3,0]=1
    # V: b00 -> V[0,0]=1, b11 -> V[3,0]=1
    # W: contributes to C00 (k=0,i=0 -> W[0,0]), C11 (k=1,i=1 -> W[3,0])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (a10 + a11) * b00
    # U: a10 -> U[2,1]=1, a11 -> U[3,1]=1
    # V: b00 -> V[0,1]=1
    # W: contributes to C10 (k=0,i=1 -> W[2,1])
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10 += M2
    
    # M3 = a00 * (b01 - b11)
    # U: a00 -> U[0,2]=1
    # V: b01 -> V[1,2]=1, b11 -> V[3,2]=-1
    # W: contributes to C01 (k=1,i=0 -> W[2,2]), C11 (k=1,i=1 -> W[3,2])
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = a11 * (b10 - b00)
    # U: a11 -> U[3,3]=1
    # V: b10 -> V[2,3]=1, b00 -> V[0,3]=-1
    # W: contributes to C00 (k=0,i=0 -> W[0,3]), C10 (k=0,i=1 -> W[2,3])
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (a00 + a01) * b11
    # U: a00 -> U[0,4]=1, a01 -> U[1,4]=1
    # V: b11 -> V[3,4]=1
    # W: contributes to C00 (k=0,i=0 -> W[0,4]), C01 (k=1,i=0 -> W[2,4])
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (a10 - a00) * (b00 + b01)
    # U: a10 -> U[2,5]=1, a00 -> U[0,5]=-1
    # V: b00 -> V[0,5]=1, b01 -> V[1,5]=1
    # W: contributes to C11 (k=1,i=1 -> W[3,5])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (a01 - a11) * (b10 + b11)
    # U: a01 -> U[1,6]=1, a11 -> U[3,6]=-1
    # V: b10 -> V[2,6]=1, b11 -> V[3,6]=1
    # W: contributes to C00 (k=0,i=0 -> W[0,6]), C11 (k=1,i=1 -> W[3,6])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1  # C00 += M7, C11 += M7
    
    return U, V, W


def _build_block_decomposition(deadline=None):
    """
    Build a decomposition for 2x3x6 using block structure.
    Split B (3x6) into three 3x2 blocks.
    For each 2x2 output block, we need to compute A(2x3) * B_block(3x2).
    
    We can write A = [A_left(2x2) | A_right(2x1)]
    And B_block = [B_top(2x2); B_bottom(1x2)]
    
    Then A * B_block = A_left * B_top + A_right * B_bottom
    
    This is a sum of two 2x2x2 multiplications. We can use Strassen for each,
    but that gives rank 14 per block, total 42 which is worse than schoolbook.
    
    Better approach: Use the structure more cleverly.
    Let's try to find a direct low-rank decomposition by solving a system.
    """
    n, m, p = DIMENSIONS
    if deadline is not None and time.time() > deadline:
        return None
    
    # Try to construct a decomposition with rank R by exploiting block structure
    # We'll try R from 21 down to 15
    
    for R in range(21, 14, -1):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try multiple random initializations with structured sparsity
        for attempt in range(50):
            if deadline is not None and time.time() > deadline:
                break
            
            # Initialize U and V with sparse structure
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            # Each row of U gets 1-2 nonzeros
            for row in range(n * m):
                nnz = random.randint(1, 2)
                cols = random.sample(range(R), nnz)
                for c in cols:
                    U[row, c] = random.choice([-1, 1])
            
            # Each row of V gets 1-2 nonzeros
            for row in range(m * p):
                nnz = random.randint(1, 2)
                cols = random.sample(range(R), nnz)
                for c in cols:
                    V[row, c] = random.choice([-1, 1])
            
            # Build linear system for W
            num_eqs = n * m * p
            num_vars = p * n * R
            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            var_idx = (k * n + i) * R + r
                            A_mat[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_, singular = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W_rounded, is_half_int = _round_to_half_int(W_flat)
            if not is_half_int:
                continue
            
            W = W_rounded.reshape(p * n, R)
            
            if _verify_fast(U, V, W):
                return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_s, V_s, W_s = _schoolbook()
    
    # Try block-structured approach
    result = _build_block_decomposition(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook if no better found
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
