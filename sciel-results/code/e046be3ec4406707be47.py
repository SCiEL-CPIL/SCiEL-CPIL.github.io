"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _time_left(deadline):
    return deadline - time.time()

def _verify(U, V, W, n, m, p):
    """Check if the tensor decomposition is exact (within tolerance)."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = len(U[0]) if U else 0
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can check this by building the matrix product and comparing
    # For each c, the contribution to the product C = A*B is:
    # C += U[:,c] reshaped as (n,m) outer with V[:,c] reshaped as (m,p)
    # Actually, the standard tensor decomposition for matrix mult:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_c sum_j (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] ... no that's not right.
    
    # Let me reconsider. The matrix multiplication tensor T is a 3-tensor with indices (i,j,k)
    # where T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    # A rank-r decomposition means T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention says W[k*n+i, r]. So W has indices (k,i) flattened as k*n+i.
    
    # So: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    # To verify: for random A, B, check that A*B = sum_c (A * U_c) * (V_c * B) ... 
    # Actually, the decomposition means:
    # C = A*B = sum_c (A @ U_c_col) * (V_c_row @ B) ... no.
    
    # Let's think differently. U has shape (n*m, r). Reshape each column c to (n, m): U_c.
    # V has shape (m*p, r). Reshape each column c to (m, p): V_c.
    # W has shape (p*n, r). Reshape each column c to (p, n): W_c.
    
    # The decomposition says: for all A (n x m) and B (m x p),
    # A @ B = sum_c (A @ U_c) * (W_c @ B) ... but (A @ U_c) is (n x p) if U_c is (n x p)? No.
    
    # Let me re-derive. U[i*m+j, c] is a scalar. If we fix c, we get a matrix U_c of shape (n, m).
    # Similarly V_c of shape (m, p), W_c of shape (p, n).
    
    # The claim is: (A @ B)[i,k] = sum_c sum_j A[i,j] * U_c[i,j] * V_c[j,k] * W_c[k,i]
    # That doesn't look right dimensionally for a simple product.
    
    # Actually, the standard form is:
    # (A @ B)[i,k] = sum_c (sum_j A[i,j] * U_c[i,j]) * (sum_j' V_c[j',k] * B[j',k]) ... no.
    
    # Let's just verify by checking the tensor directly.
    # T[i,j,k] should be 1 for all i in range(n), j in range(m), k in range(p).
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a trilinear form. We can check it on random inputs efficiently.
    # For random A (n x m) and B (m x p), the product C = A @ B has C[i,k] = sum_j A[i,j]*B[j,k].
    # The decomposition gives: C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_k' ... ) 
    # Hmm, let me just compute the tensor sum directly for verification.
    
    # Efficient check: pick random A and B, compute A@B, and compare with the decomposition-based product.
    # The decomposition-based product: for each c, define:
    #   A_c = A @ U_c  where U_c is (n, m) -> A_c is (n, p)? No, A is (n,m), U_c is (n,m), so A @ U_c doesn't work.
    
    # I think the correct interpretation is:
    # C[i,k] = sum_c (A @ U_c)[i, ?] ... 
    
    # Let's just do the direct tensor check with a few random vectors.
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 for all i,j,k.
    
    # Efficient method: for random a (n x m) and b (m x p), 
    # sum_{i,j,k} a[i,j]*b[j,k]*T[i,j,k] should equal sum_{i,j,k} a[i,j]*b[j,k] = sum_i,k (a@b)[i,k]
    
    # And sum_{i,j,k} a[i,j]*b[j,k]*T[i,j,k] = sum_c (sum_{i,j} a[i,j]*U[i*m+j,c]) * (sum_{j,k} b[j,k]*V[j*p+k,c]) * (sum_{k,i} 1*W[k*n+i,c])
    # No, that's not right either because the sums are coupled.
    
    # Actually: sum_{i,j,k} a[i,j]*b[j,k]*sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_c (sum_{i,j,k} a[i,j]*b[j,k]*U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])
    
    # This is a trilinear form. Let me just verify with random A, B by computing:
    # LHS = sum_{i,k} (A@B)[i,k] = sum_{i,j,k} A[i,j]*B[j,k]
    # RHS = sum_c (A : U_c) * (B : V_c) * (1 : W_c) ... no.
    
    # I'll just do a direct numerical check by evaluating the trilinear form.
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        # Compute via decomposition:
        # C[i,k] = sum_c sum_j A[i,j] * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is expensive to compute directly. Instead:
        # For each c, define M_c = U_c (n x m), N_c = V_c (m x p), P_c = W_c (p x n)
        # Then contribution to C[i,k] from term c is: sum_j A[i,j]*M_c[i,j]*N_c[j,k]*P_c[k,i]
        # This is not a simple matrix product.
        
        # Let me just compute it directly for small dimensions.
        C_decomp = np.zeros((n, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for c in range(rank):
                        s += A[i,j] * B[j,k] * U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                C_decomp[i,k] = s
        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2(n, m, p):
    """Try to use Strassen-like block decomposition.
    For 3x3 * 3x8, we can split the 8 columns into blocks.
    Split B into B1 (3x4) and B2 (3x4). Then A*B = [A*B1, A*B2].
    Each A*B1 is 3x3 * 3x4. We can further split.
    
    Alternatively, split A into rows and B into columns.
    Let's try: A is 3x3, B is 3x8.
    Split B into two 3x4 blocks: B = [B_left, B_right].
    C = [A*B_left, A*B_right], each is 3x4.
    
    For 3x3 * 3x4: split A into top 2 rows and bottom 1 row? Not clean.
    
    Let's try a different approach: use the fact that 3x3 * 3x3 can be done with rank 23 (Strassen-like) 
    and then extend. But 3x3 matrix multiplication has minimum rank 23 (known result).
    
    Actually, for 3x3 * 3x8, we can view it as 8 separate 3x3 * 3x1 products plus cross terms.
    
    Let me try a block approach:
    Split the 8 columns of B into groups. Say 4+4.
    C = [C1, C2] where C1 = A*B1 (3x4), C2 = A*B2 (3x4).
    
    For each 3x3 * 3x4 product, split B_i into two 3x2 blocks.
    C_i = [C_i1, C_i2] where C_i1 = A*B_i1 (3x2), C_i2 = A*B_i2 (3x2).
    
    For 3x3 * 3x2: split A into top 2 rows and bottom 1 row? 
    A = [A_top; A_bottom] where A_top is 2x3, A_bottom is 1x3.
    B_i1 is 3x2.
    C_i1 = [A_top @ B_i1; A_bottom @ B_i1] = [2x2 result; 1x2 result].
    
    The 2x2 * 3x2 part: split B_i1 into two 3x1 columns.
    A_top @ B_i1 = [A_top @ b1, A_top @ b2] where b1, b2 are 3x1.
    A_top is 2x3, b1 is 3x1. This is 2x3 * 3x1.
    
    This is getting complex. Let me just try to find a known low-rank decomposition.
    
    Known: 2x2 * 2x2 has rank 7 (Strassen).
    3x3 * 3x3 has rank 23 (best known exact).
    
    For 3x3 * 3x8, one approach: use 8 copies of a 3x3 * 3x1 decomposition.
    3x3 * 3x1: C[i] = sum_j A[i,j] * B[j]. This is just a linear combination, rank 9 (schoolbook) but can be done with rank 9.
    Actually, 3x3 * 3x1 is just matrix-vector multiplication, which has rank 9 (each output is a linear combination).
    
    Hmm, let me think about this differently. The total rank for 3x3 * 3x8 schoolbook is 3*3*8 = 72.
    
    Can we do better? Let's try a block Strassen approach.
    
    Split B (3x8) into B1 (3x4) and B2 (3x4).
    C = [A*B1, A*B2].
    
    For A*B1 (3x3 * 3x4): split B1 into B1a (3x2) and B1b (3x2).
    A*B1 = [A*B1a, A*B1b].
    
    For A*B1a (3x3 * 3x2): 
    Split A into A1 (2x3) and A2 (1x3).
    A*B1a = [A1*B1a; A2*B1a] where A1*B1a is 2x2, A2*B1a is 1x2.
    
    A1*B1a is 2x2 * 3x2. Split B1a into two 3x1 vectors b1, b2.
    A1*b1 is 2x3 * 3x1. Rank 6 schoolbook.
    
    This isn't leading to obvious savings. Let me just return the schoolbook for now and add a search.
    """
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition using random search / structured methods
    # For now, return schoolbook. The search would need to be more sophisticated.
    
    # Simple improvement attempt: check if we can reduce rank by exploiting structure
    # For 3x3 * 3x8, let's try a few structured decompositions
    
    # Try: split 8 into 4+4, use separate decompositions for each half
    # Each 3x3 * 3x4 has schoolbook rank 36. Total 72. No improvement.
    
    # Try: use the fact that some entries might be shared
    # This requires actual tensor rank computation which is hard.
    
    # Return schoolbook
    return best
# EVOLVE-BLOCK-END
