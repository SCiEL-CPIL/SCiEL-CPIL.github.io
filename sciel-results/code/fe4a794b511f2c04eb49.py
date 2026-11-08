import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110  # Safety margin
    
    # Strategy: Decompose 6x6 matrix multiplication into 3x3 blocks of 2x2 matrices.
    # Each 2x2 block multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 27 * 7 = 189.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    
    # Initialize U, V, W as lists of lists (as per seed interface expectations)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    # Each pattern is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # coeffs are lists of (row, col, val)
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # M1 = (A10+A11)B00
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # M2 = A00(B01-B11)
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        # M3 = A11(B10-B00)
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        # M4 = (A00+A01)B11
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        # M5 = (A10-A00)(B00+B01)
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        # M6 = (A01-A11)(B10+B11)
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U
                    for pi, pj, val in u_coeffs:
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    # Update V
                    for pi, pj, val in v_coeffs:
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    # Update W
                    for pi, pj, val in w_coeffs:
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
                    if time.time() > deadline:
                        # Fallback to naive if timeout (unlikely)
                        return _naive_construct(n, m, p)
        
    # Self-test to ensure correctness
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to naive if verification fails
        return _naive_construct(n, m, p)

def _naive_construct(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][idx] = 1
                V[k * p + j][idx] = 1
                W[j * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    # Convert to numpy for fast verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random test matrices
    np.random.seed(42)
    A = np.random.randint(0, 10, (n, m))
    B = np.random.randint(0, 10, (m, p))
    C_true = A @ B
    
    # Compute C_decomp
    # C_ij = sum_r U_{ik,r} V_{kj,r} W_{ij,r}
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    # Note: U is (n*m, rank). Index is (i*m + k). So reshape to (n, m, rank).
    U_reshaped = U_np.reshape(n, m, -1)
    V_reshaped = V_np.reshape(m, p, -1)
    W_reshaped = W_np.reshape(p, n, -1) # Index is (j*n + i). Wait.
    
    # Check W index: W[k * n + i]. k is row (0..p-1), i is col (0..n-1).
    # So W is (p, n, rank).
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            # Sum over k and r
            # C_ij = sum_k sum_r U[i, k, r] * V[k, j, r] * W[j, i, r]
            # Vectorize the sum over k and r
            # U_slice: (m, rank)
            # V_slice: (m, rank)
            # W_slice: (rank)
            
            # This loop is O(n*p*m*rank). 6*6*6*189 ~ 40k ops. Very fast.
            sum_val = 0.0
            for k in range(m):
                u_vec = U_reshaped[i, k, :]
                v_vec = V_reshaped[k, j, :]
                w_val = W_reshaped[j, i, :] # Wait, W is (p, n, rank). W[j, i] is a vector of rank?
                # No, W is (p*n, rank). Index j*n + i.
                # So W_reshaped is (p, n, rank).
                # W_reshaped[j, i] is a vector of length rank.
                
                # The formula is sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
                # This is dot product of (U[i,k,:] * V[k,j,:]) with W[j,i,:]
                prod_vec = u_vec * v_vec
                sum_val += np.dot(prod_vec, w_val)
            C_decomp[i, j] = sum_val
            
    return np.allclose(C_true, C_decomp, atol=1e-6)

# EVOLVE-BLOCK-END
