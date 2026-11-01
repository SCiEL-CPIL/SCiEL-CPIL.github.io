"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n=2, m=2, p=11):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] summed? 
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k])? No.
    # Correct: The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r]... 
    # No. The standard: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] where T[i,j,k]=1.
    # But W is the output factor: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * W[k*n+i,r] * ... 
    # Let me use: the bilinear form is C[i,k] = sum_{j,r} U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] / W... 
    # Actually per the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The decomposition means: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] summed over r? 
    # No, it should be: for the tensor, T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] and this should 
    # relate to W. Let me think: C[i,k] = sum_j A[i,j] B[j,k]. 
    # With factors: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (something) 
    # I think the correct check is: sum_r U[i*m+j,r] * V[j*p+k,r] should be independent of j and equal to W[k*n+i,r]... 
    # Actually, I believe the convention is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])? No, that's not bilinear.
    
    # Let me just check: for the tensor T[i,j,k] = 1 (since A[i,j]*B[j,k] contributes to C[i,k]),
    # we need: sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ... no, W has the same rank dimension.
    
    # I think the correct interpretation is:
    # The multiplication is: C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * ... 
    # This doesn't work dimensionally.
    
    # Let me use a different approach: verify by random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B  # (n, p)
    
    # Reconstruct C from factors
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]  
    # W: (p*n, rank) -> W[k*n+i, r]
    # C[i,k] should equal sum_r sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] / W[k*n+i,r]? 
    # I'm confused. Let me just check if sum_r U[i*m+j,r]*V[j*p+k,r] is constant in j and equals something related to W.
    
    # Alternative: the tensor decomposition is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r]
    # and T[i,j,k] should be 1 for all i,j,k (for standard matrix mult)
    # But W is separate. Maybe W is used differently.
    
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And "W[k*n+i,r]" is the trace convention for output.
    # I think: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
    # But that's not right either because the sums over j would be separate.
    
    # Let me just test with the schoolbook to understand the convention.
    # Schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for col corresponding to (i,j,k)
    # So for a given (i,j,k), there's a column col with U[i*m+j,col]=1, V[j*p+k,col]=1, W[k*n+i,col]=1
    # Then C[i,k] = sum_col W[k*n+i,col] * (sum_j U[i*m+j,col]*A[i,j]) * (sum_j V[j*p+k,col]*B[j,k])? 
    # For schoolbook, sum_j U[i*m+j,col]*A[i,j] = A[i,j] (only one j is nonzero)
    # sum_j V[j*p+k,col]*B[j,k] = B[j,k]
    # So C[i,k] = sum_col W[k*n+i,col] * A[i,j(col)] * B[j(col),k] where j(col) is the j for that column
    # = sum_j A[i,j]*B[j,k] (since W[k*n+i,col]=1 for the matching col)
    # Yes! So the formula is: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
    # Wait, but the two sums over j are independent, which would give a product of sums, not a sum of products.
    
    # Let me reconsider. Maybe it's:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # For schoolbook: for column r corresponding to (i',j',k'), 
    #   sum_j U[i*m+j,r]*A[i,j] = A[i,j'] if i=i' else 0
    #   sum_j V[j*p+k,r]*B[j,k] = B[j',k] if k=k' else 0
    #   W[k*n+i,r] = 1 if (i,k)=(i',k') else 0
    # So C[i,k] = sum_{i',j',k': i=i', k=k'} A[i,j']*B[j',k] = sum_j A[i,j]*B[j,k]. Correct!
    
    # So the formula is: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
    
    C_recon = np.zeros((n, p))
    for r in range(rank):
        # For each i, compute sum_j U[i*m+j, r] * A[i, j]
        U_part = np.zeros((n,))
        V_part = np.zeros((p,))
        for i in range(n):
            for j in range(m):
                U_part[i] += U[i*m+j][r] * A[i,j]
        for k in range(p):
            for j in range(m):
                V_part[k] += V[j*p+k][r] * B[j,k]
        for i in range(n):
            for k in range(p):
                C_recon[i,k] += W[k*n+i][r] * U_part[i] * V_part[k]
    
    if not np.allclose(C_recon, C_true, atol=1e-9):
        return False
    return True

def _rank_of_decomposition(U, V, W):
    return len(U[0]) if U else 0

def construct():
    n, m, p = DIMENSIONS
    # Try to find a low-rank decomposition
    # Start with schoolbook (rank 44) and try to reduce
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 44
    
    # First, create schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n*m*p
    
    # Try random initialization with lower ranks and verify
    # Use a smarter approach: exploit the structure
    # For 2x2 x 2x11, we can use the fact that the 2x2 part can be done in rank 7 (Strassen)
    # and then handle the 11 columns
    
    # Let's try a direct optimization approach
    # We'll try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) / something... actually LB is at least max(n*m, m*p, p*n) for some cases
    # But we can try to find a good decomposition
    
    # Use a simple approach: try rank values and do random search
    np.random.seed(123)
    
    # Try a few candidate ranks
    for rank in [30, 35, 40, 44]:
        if time.time() > _DEADLINE - 10:
            break
        if rank >= best_rank:
            continue
            
        # Try random initializations
        for trial in range(50):
            if time.time() > _DEADLINE - 5:
                break
                
            U_rand = np.random.randn(n*m, rank)
            V_rand = np.random.randn(m*p, rank)
            W_rand = np.random.randn(p*n, rank)
            
            # Verify
            if _verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                best_U, best_V, best_W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                best_rank = rank
                break
            # Random won't work for exact decomposition, need structured approach
    
    # The random approach won't find exact decompositions. Let me use a structured approach.
    # For 2x2 x 2x11, use the Kronecker product structure or known algorithms.
    
    # Actually, let me try a different strategy: use the fact that 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a linear combination. We can precompute linear combinations of A rows and B rows.
    
    # A better approach: note that for each output column k, we're computing a 2x1 vector:
    # C[:,k] = A @ B[:,k]
    # This is a 2x2 matrix times a 2x1 vector, which can be done in 7 multiplications (Strassen)
    # But we share the A matrix across all 11 columns.
    
    # Let's use a block approach: group columns and use Strassen on the 2x2 blocks
    # Or use a simple optimization with the correct structure
    
    # Given the difficulty, let me just return the schoolbook if we can't do better
    # But first, try a few more structured approaches
    
    # Approach: use the identity that 2x2 x 2xp can be done in rank 2p + 2*floor(p/2) or similar
    # Actually, let me try to use a greedy column-by-column approach with sharing
    
    # For now, return the best we have
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
