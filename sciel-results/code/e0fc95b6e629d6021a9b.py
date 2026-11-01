"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The tensor element for (i,j,k) should be sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if the "multiplication" contributes, i.e., for the standard MM tensor.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _try_rank_11():
    """
    Attempt to construct a rank-11 decomposition.
    
    Strategy: The 2x2x2 MM tensor has rank 7 (Strassen). For 2x2x3, we can try to
    use the 7 Strassen terms for the first two columns of B, plus 4 additional terms
    for the third column, but with some sharing.
    
    A cleaner approach: Use the known fact that R(2,2,3) = 11.
    We can construct this by taking the rank-7 Strassen decomposition for columns 1,2
    of B, and adding 4 terms for column 3. But we need to be careful about the tensor structure.
    
    Actually, let's use a direct constructive approach:
    The multiplication tensor for 2x2 * 2x3 has 12 nonzero entries (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1,2}.
    We need to cover all 12 with rank-1 tensors (trilinear forms).
    
    Let's try: 7 Strassen terms for (i,j) in {0,1}^2 and k in {0,1}, plus 4 terms for k=2.
    But the k=2 terms can share structure with k=0 or k=1 terms.
    
    A known construction: R(2,2,3) = R(2,2,2) + 4 = 11, since the third column adds 4 new
    input-output pairs (i=0,1; k=2) that can't be covered by existing terms without adding new rank-1 terms.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen's 7 terms for 2x2x2
    # Standard Strassen decomposition for C = A*B where A, B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M7
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M3 + M6
    
    # In our indexing: A[i,j], B[j,k], C[i,k]
    # For k=0,1 (first two columns of B), we use Strassen-like terms.
    # Let's define the 7 Strassen terms for the 2x2 part (k=0,1).
    
    # Term 0: (A00+A11)(B00+B11) -> U[0]=1,U[3]=1; V[0]=1,V[3]=1; W[0]=1,W[3]=1
    # But wait, W[k*n+i]: for k=0, W[0*2+0]=W[0], W[0*2+1]=W[1]; for k=1, W[1*2+0]=W[2], W[1*2+1]=W[3]
    # The Strassen combination happens in W, not in U/V. So each rank-1 term contributes to specific (i,j,k).
    
    # Let me think differently. Each rank-1 term c has:
    # U_c: a vector in R^4 (entries for i*m+j)
    # V_c: a vector in R^6 (entries for j*p+k)
    # W_c: a vector in R^6 (entries for k*n+i)
    # The contribution to tensor entry (i,j,k) is U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # For a rank-1 term to contribute to entry (i,j,k), we need all three to be nonzero.
    # The Strassen decomposition uses 7 terms where each term is a product of linear forms.
    
    # Let's use a known explicit rank-11 decomposition for 2x2x3.
    # One construction: take the rank-7 decomposition of 2x2x2 and add 4 terms for the third column.
    
    # Strassen terms for 2x2x2 (k=0,1):
    # We'll set up 7 terms that handle k=0 and k=1.
    
    # Term 0: U: [1,0,0,1], V: [1,0,1,0,0,0], W: [1,0,1,0,0,0]
    # This contributes to (0,0,0): 1*1*1=1, (0,0,1): 1*1*0=0... hmm, this isn't right.
    
    # Let me use a more systematic approach. The tensor T[i,j,k] = 1 for all i,j,k in the 2x2x3 case.
    # We need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k.
    
    # This is a tensor completion / decomposition problem. Let's try to find it via search.
    
    # Given time constraints, let's use a known explicit decomposition.
    # From literature: R(2,2,3) = 11.
    
    # I'll use a constructive approach with 11 terms.
    # Terms 0-6: handle the 2x2x2 part (k=0,1) using Strassen.
    # Terms 7-10: handle k=2.
    
    # For Strassen on 2x2x2, the 7 terms are:
    # c=0: U=[1,0,0,1], V=[1,0,1,0,0,0], W=[1,0,0,1,0,0]
    #   Contributes to (0,0,0): 1*1*1=1, (1,1,1): 1*1*1=1, others 0
    #   But we need (0,0,0)=1, (0,1,0)=1, (1,0,0)=1, (1,1,0)=1, etc.
    
    # This is getting complex. Let me just verify if a rank-11 solution exists by trying a simple construction.
    
    # Simple construction: 4 terms for each column k=0,1,2 would be 12 terms.
    # To get 11, we need to share one term between two columns.
    
    # Let's try: for k=0 and k=1, use a shared structure.
    # Actually, the minimal construction likely uses the fact that the 2x2x3 tensor
    # can be seen as 2x2x2 + 2x2x1, and R(2,2,1)=4, R(2,2,2)=7, but R(2,2,3) < 11?
    # No, R(2,2,3) = 11 is known.
    
    # Let me try a direct search with a time limit.
    
    return None  # Fallback: no explicit rank-11 found yet

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-11 construction
    # Use a systematic search for rank-11 decomposition
    
    # For now, return the schoolbook rank-12 decomposition as fallback
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
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find rank-11 decomposition
    if time.time() < deadline:
        # Use a known explicit rank-11 decomposition
        # Construction: 7 Strassen terms for k=0,1 + 4 terms for k=2
        # But we need to verify.
        
        # Let's try a specific construction:
        # The 2x2x3 MM tensor can be decomposed as follows:
        # For each (i,j) pair, we need to handle k=0,1,2.
        
        # Known rank-11 decomposition (from tensor decomposition literature):
        # This is a 4x6x6 tensor (U:4x11, V:6x11, W:6x11)
        
        # I'll construct it explicitly:
        U11 = [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # i=0,j=0
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # i=0,j=1
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # i=1,j=0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # i=1,j=1 - placeholder
        ]
        
        # This is getting too speculative. Let me just return the schoolbook for now
        # and rely on the verification to ensure correctness.
        
        pass
    
    return best
# EVOLVE-BLOCK-END
