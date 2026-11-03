"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Winograd-based decomposition using 5-point DFT and Strassen
    # The idea: use the 5-point DFT to transform the 5×5 block structure
    # Then apply Strassen (rank 7) to 2×2×2 sub-blocks
    
    # 5-point DFT matrix (using primitive 5th root of unity)
    # ω = exp(2πi/5)
    # For exact arithmetic, we use the minimal polynomial approach
    # But for this problem, we can use a structured decomposition
    
    # Actually, let's use a different approach:
    # Split the 5×5 matrix into blocks and use Strassen where possible
    
    # For 3×5×6:
    # - The 5×5 middle dimension can be split as 2+2+1
    # - We can use Strassen for the 2×2×2 blocks
    # - Use schoolbook for the remaining 1×1 and cross terms
    
    def winograd_strassen_decomposition():
        """
        Use Winograd-style decomposition with Strassen for 2×2×2 blocks.
        
        Split dimensions:
        - n=3: keep as is (or split as 2+1)
        - m=5: split as 2+2+1  
        - p=6: split as 2+2+2
        
        Apply Strassen to each 2×2×2 block, schoolbook to rest.
        """
        if time.time() > deadline:
            return None
        
        # Strassen's rank-7 decomposition for 2×2×2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # For our tensor: T[i,j,k] = 1 for all valid (i,j,k)
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's construct the decomposition explicitly
        # Split m into m1=2, m2=2, m3=1
        # Split p into p1=2, p2=2, p3=2
        # n stays as 3
        
        # For each (i, j_block, k_block) where j_block, k_block in {1,2,3}:
        # If both j_block and k_block are size 2, use Strassen (7 products)
        # Otherwise, use schoolbook (size_j * size_k products)
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Define the splits
        m_splits = [(0, 2), (2, 4), (4, 5)]  # j ranges: [0,2), [2,4), [4,5)
        p_splits = [(0, 2), (2, 4), (4, 6)]  # k ranges: [0,2), [2,4), [4,6)
        
        for i in range(n):
            for mi, (j_start, j_end) in enumerate(m_splits):
                for pi, (k_start, k_end) in enumerate(p_splits):
                    j_size = j_end - j_start
                    k_size = k_end - k_start
                    
                    # Determine if we can use Strassen
                    if j_size == 2 and k_size == 2:
                        # Use Strassen: 7 products
                        # A block is 1×2 (row i, columns j_start to j_end)
                        # B block is 2×2 (rows j_start to j_end, columns k_start to k_end)
                        # C block is 1×2 (row i, columns k_start to k_end)
                        
                        # Strassen for 1×2 * 2×2 -> 1×2
                        # This is a bit different from standard 2×2
                        # Let's use schoolbook for simplicity here, or adapt Strassen
                        
                        # Actually, for 1×2 * 2×2, we need:
                        # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
                        # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
                        # This requires 4 products (schoolbook)
                        # Strassen doesn't help for rectangular 1×2 * 2×2
                        pass
                    
                    # For now, use schoolbook for all blocks
                    for j in range(j_start, j_end):
                        for k in range(k_start, k_end):
                            U_col = np.zeros(n * m)
                            V_col = np.zeros(m * p)
                            W_col = np.zeros(p * n)
                            
                            U_col[i * m + j] = 1
                            V_col[j * p + k] = 1
                            W_col[k * n + i] = 1
                            
                            U_cols.append(U_col)
                            V_cols.append(V_col)
                            W_cols.append(W_col)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        rank = U.shape[1]
        
        if verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist(), 'rank': rank}
        
        return None
    
    # Try the Winograd-Strassen approach
    result = winograd_strassen_decomposition()
    if result is not None:
        rank = result.pop('rank')
        if rank < best_rank:
            best_rank = rank
            best_result = result
    
    # Try to find even better decompositions
    # For now, return the best we have
    return best_result
# EVOLVE-BLOCK-END
