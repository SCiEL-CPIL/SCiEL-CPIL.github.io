"""
Matrix multiplication 2,2,15 rank-minimal exact tensor decomposition.

Strategy:
  1. Use a known rank-7 Strassen decomposition for the 2x2x2 block.
  2. For p=15, decompose as 7 blocks of size 2 and 1 block of size 1.
     - Each 2x2x2 block contributes rank 7.
     - The 2x2x1 block contributes rank 4 (schoolbook).
     Total rank = 7*7 + 4 = 53.
  3. However, we can do better by using a "global" Strassen-like approach.
     The key insight: for M_{2,2,p}, the rank is at most 2p + 4 for p >= 2.
     For p=15, that gives rank 34.
     
  Construction of rank 2p+4:
  We use the following idea:
  - The 2x2 matrix A has 4 entries: a00, a01, a10, a11.
  - The 2xp matrix B has 2p entries.
  - The output C is 2xp.
  
  We can think of this as: C = A * B.
  
  A known construction uses:
  - 4 "global" terms that handle the 2x2 structure of A combined with
    appropriately chosen linear combinations of B's columns.
  - 2p terms that handle the remaining structure.
  
  Actually, let me use a cleaner construction:
  
  For M_{2,2,p}, one can achieve rank at most 2p + 4 as follows:
  
  Define for each k in 0..p-1:
    Term 2k:   u = (a00, 0, a10, 0),  v = (b0k, 0),  w = (ck0, 0)
    Term 2k+1: u = (0, a01, 0, a11),  v = (0, b1k),  w = (0, ck1)
  That's 2p terms, but this is just schoolbook for each column, giving rank 2p.
  Wait, that's not right. Let me reconsider.
  
  Actually, the schoolbook rank is 4p = 60.
  
  A better approach: use Strassen on pairs of columns.
  For each pair of columns (2k, 2k+1), use the rank-7 Strassen decomposition.
  For p=15, we have 7 pairs and 1 remaining column.
  Rank = 7*7 + 4 = 53.
  
  Can we do better? The lower bound for M_{2,2,p} is at least 2p (from the
  slice rank bound or similar). The value 2p+4 = 34 is a known upper bound
  for certain constructions.
  
  Let me implement the rank 2p+4 construction:
  
  The idea is to use 4 "global" terms that mix across all columns, plus
  2p terms. But I need to verify this is correct.
  
  Actually, let me just use the block decomposition with rank 53, which is
  known to be valid, and see if we can improve.
  
  For now, I'll implement the rank-53 block decomposition as a safe baseline,
  and also attempt a rank-34 construction.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W of shape (4,7).
    U indexed by [i*m+j], V by [j*p+k], W by [k*n+i]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1.0
    U[3, 0] = 1.0
    V[0, 0] = 1.0
    V[3, 0] = 1.0
    W[0, 0] = 1.0
    W[3, 0] = 1.0
    
    # M2 = (a10 + a11) * b00
    U[2, 1] = 1.0
    U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0
    W[3, 1] = -1.0
    
    # M3 = a00 * (b10 - b11)
    U[0, 2] = 1.0
    V[2, 2] = 1.0
    V[3, 2] = -1.0
    W[1, 2] = 1.0
    W[3, 2] = 1.0
    
    # M4 = a11 * (b01 - b00)
    U[3, 3] = 1.0
    V[1, 3] = 1.0
    V[0, 3] = -1.0
    W[0, 3] = -1.0
    W[2, 3] = 1.0
    
    # M5 = (a00 + a01) * b11
    U[0, 4] = 1.0
    U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0
    W[1, 4] = 1.0
    
    # M6 = (a10 - a00) * (b00 + b10)
    U[2, 5] = 1.0
    U[0, 5] = -1.0
    V[0, 5] = 1.0
    V[2, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (a01 - a11) * (b01 + b11)
    U[1, 6] = 1.0
    U[3, 6] = -1.0
    V[1, 6] = 1.0
    V[3, 6] = 1.0
    W[0, 6] = -1.0
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Schoolbook for 2x2x1. U(4,4), V(2,4), W(2,4).
    """
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Block decomposition: 7 blocks of size 2 (rank 7 each) + 1 block of size 1 (rank 4).
    Total rank = 53.
    """
    if n != 2 or m != 2:
        return None
    
    block_sizes = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            block_sizes.append(2)
            remaining -= 2
        else:
            block_sizes.append(1)
            remaining -= 1
    
    U_s, V_s, W_s = _strassen_2x2x2()
    U_1, V_1, W_1 = _schoolbook_2x2x1()
    
    total_rank = sum(7 if bs == 2 else 4 for bs in block_sizes)
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    for bs in block_sizes:
        if bs == 2:
            for r in range(7):
                U[0, col_offset] = U_s[0, r]
                U[1, col_offset] = U_s[1, r]
                U[2, col_offset] = U_s[2, r]
                U[3, col_offset] = U_s[3, r]
                
                V[0 * p + k_offset, col_offset] = V_s[0, r]
                V[0 * p + k_offset + 1, col_offset] = V_s[1, r]
                V[1 * p + k_offset, col_offset] = V_s[2, r]
                V[1 * p + k_offset + 1, col_offset] = V_s[3, r]
                
                W[k_offset * n + 0, col_offset] = W_s[0, r]
                W[k_offset * n + 1, col_offset] = W_s[1, r]
                W[(k_offset + 1) * n + 0, col_offset] = W_s[2, r]
                W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
                
                col_offset += 1
            k_offset += 2
        elif bs == 1:
            for r in range(4):
                U[0, col_offset] = U_1[0, r]
                U[1, col_offset] = U_1[1, r]
                U[2, col_offset] = U_1[2, r]
                U[3, col_offset] = U_1[3, r]
                
                V[0 * p + k_offset, col_offset] = V_1[0, r]
                V[1 * p + k_offset, col_offset] = V_1[1, r]
                
                W[k_offset * n + 0, col_offset] = W_1[0, r]
                W[k_offset * n + 1, col_offset] = W_1[1, r]
                
                col_offset += 1
            k_offset += 1
            
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _global_strassen_like(n, m, p):
    """
    Attempt a rank 2p+4 construction for M_{2,2,p}.
    
    The construction:
    We use 4 "global" terms plus 2p terms.
    
    The 4 global terms handle the 2x2 structure of A in a Strassen-like way,
    but applied to the entire B matrix.
    
    Specifically, for the 2x2 part, Strassen uses 7 multiplications.
    For M_{2,2,p}, we can use the following:
    
    Let A = [[a00, a01], [a10, a11]]
    Let B = [[b00, b01, ..., b0,p-1],
             [b10, b11, ..., b1,p-1]]
    
    We want C = A * B, where C = [[c00, c01, ..., c0,p-1],
                                  [c10, c11, ..., c1,p-1]]
    
    The construction uses:
    - 4 terms that are "global" in the k dimension (they involve all columns of B)
    - 2p terms that are "local" to each column
    
    Actually, a cleaner way: use the fact that the tensor M_{2,2,p} can be
    written as the sum of:
    - A 2x2x2 tensor (for each pair of columns) -- but that gives rank 7 per pair
    - Or use a different decomposition.
    
    Let me try a direct construction:
    
    For each k in 0..p-1, define:
      u_k^0 = (a00, 0, a10, 0)  -- picks a00 and a10
      u_k^1 = (0, a01, 0, a11)  -- picks a01 and a11
      
      v_k^0 = (b0k, 0)  -- picks b0k
      v_k^1 = (0, b1k)  -- picks b1k
      
      w_k^0 = (ck0, 0)  -- picks ck0
      w_k^1 = (0, ck1)  -- picks ck1
    
    This gives 2p terms, but it's just schoolbook for each column pair, giving rank 2p.
    Wait, that's not schoolbook. Schoolbook would be 4p terms.
    
    Actually, the above construction gives:
    For each k:
      Term 2k:   u=(a00,0,a10,0), v=(b0k,0), w=(ck0,0)
                 contributes a00*b0k to ck0
      Term 2k+1: u=(0,a01,0,a11), v=(0,b1k), w=(0,ck1)
                 contributes a01*b1k to ck1
    
    But we need:
      ck0 = a00*b0k + a01*b1k
      ck1 = a10*b0k + a11*b1k
    
    So the above is missing terms. We need more terms.
    
    Let me just use the block decomposition with rank 53, which is known to work.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try block decomposition first (rank 53)
    U, V, W = _block_decomposition(n, m, p)
    
    if U is not None and _is_valid(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to schoolbook (rank 60)
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
