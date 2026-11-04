import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# Strassen 2x2 decomposition factors
# U_t, V_t, W_t are 2x2 matrices for the t-th term (1..7)
_STRASSEN_U = [
    [[1, 0], [0, 1]],
    [[0, 0], [1, 1]],
    [[1, 0], [0, 0]],
    [[0, 0], [0, 1]],
    [[1, 1], [0, 0]],
    [[-1, 0], [1, 0]],
    [[0, 1], [0, -1]]
]

_STRASSEN_V = [
    [[1, 0], [0, 1]],
    [[1, 0], [0, 0]],
    [[0, 1], [0, -1]],
    [[-1, 0], [1, 0]],
    [[0, 0], [0, 1]],
    [[1, 1], [0, 0]],
    [[0, 0], [1, 1]]
]

_STRASSEN_W = [
    [[1, 0], [0, 1]],
    [[0, 0], [1, -1]],
    [[0, 1], [0, 1]],
    [[1, 0], [1, 0]],
    [[-1, 1], [0, 0]],
    [[0, 0], [0, 1]],
    [[1, 0], [0, 1]]
]

def _build_6x6_strassen_block():
    """
    Constructs the 6x6x6 decomposition using 3x3 blocks of 2x2 matrices.
    Each 2x2 block multiplication is done via Strassen (rank 7).
    Total rank = 3 (block rows) * 3 (block cols) * 3 (block sum index) * 7 (strassen terms) = 189.
    """
    n = m = p = 6
    block_size = 2
    num_blocks = 3
    rank = num_blocks**3 * 7
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for I in range(num_blocks):      # Block row of A and C
        for K in range(num_blocks):  # Block col of A and Block row of B
            for J in range(num_blocks): # Block col of B and C
                for t in range(7):
                    U_t = _STRASSEN_U[t]
                    V_t = _STRASSEN_V[t]
                    W_t = _STRASSEN_W[t]
                    
                    # Fill U
                    for u in range(block_size):
                        for v in range(block_size):
                            val = U_t[u][v]
                            if val != 0:
                                # Global row i = I*2 + u
                                # Global col j = K*2 + v
                                i_global = I * block_size + u
                                j_global = K * block_size + v
                                U[i_global * m + j_global, col] = val
                                
                    # Fill V
                    for v in range(block_size):
                        for w in range(block_size):
                            val = V_t[v][w]
                            if val != 0:
                                # Global row j = K*2 + v
                                # Global col k = J*2 + w
                                j_global = K * block_size + v
                                k_global = J * block_size + w
                                V[j_global * p + k_global, col] = val
                                
                    # Fill W
                    for w in range(block_size):
                        for u in range(block_size):
                            val = W_t[w][u]
                            if val != 0:
                                # Global row k = J*2 + w
                                # Global col i = I*2 + u
                                # Interface says W[k*n+i, r]
                                k_global = J * block_size + w
                                i_global = I * block_size + u
                                W[k_global * n + i_global, col] = val
                                
                    col += 1
                    
    return U, V, W

def _verify(U, V, W, n=6, m=6, p=6, tol=1e-8):
    """
    Verify the decomposition by checking if the reconstructed tensor matches the identity tensor.
    """
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # M[a, b] = sum_r U[a, r] * V[b, r] * W[c(a,b), r]
    # c(a,b) = (b % p) * n + (a // m)
    
    M = np.zeros((a_dim, b_dim))
    for r in range(U.shape[1]):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        # Compute c indices for all b
        b_idx = np.arange(b_dim)
        a_idx = np.arange(a_dim)
        c_idx = (b_idx[None, :] % p) * n + (a_idx[:, None] // m)
        
        # M[a,b] += U[a,r] * V[b,r] * W[c_idx[a,b], r]
        # Note: W is indexed by c_idx.
        # w_r is a vector of length c_dim.
        # We need to gather w_r[c_idx]
        w_gathered = w_r[c_idx]
        
        M += np.outer(u_r, v_r * w_gathered)
        
    # True tensor: M[a, b] = 1 if a//m == b%p else 0
    a_idx = np.arange(a_dim)
    b_idx = np.arange(b_dim)
    M_true = ((a_idx[:, None] // m) == (b_idx[None, :] % p)).astype(float)
    
    return np.allclose(M, M_true, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120
    
    U, V, W = _build_6x6_strassen_block()
    rank = U.shape[1]
    
    # Optional: Verify if time permits
    if time.time() < deadline - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U[i * m + j, r] = 1
                        V[j * p + k, r] = 1
                        W[k * n + i, r] = 1
            rank = U.shape[1]
            
    # Convert to list of lists of integers
    Uo = [[int(round(x)) for x in row] for row in U]
    Vo = [[int(round(x)) for x in row] for row in V]
    Wo = [[int(round(x)) for x in row] for row in W]
    
    return {"U": Uo, "V": Vo, "W": Wo}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
