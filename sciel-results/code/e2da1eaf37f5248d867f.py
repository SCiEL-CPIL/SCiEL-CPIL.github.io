import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p  # 72

    def make_schoolbook():
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek rank-r decomposition: T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Lower bound for rank is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Let's try to find a decomposition with rank < 72
    
    # Strategy: Use random search with structured initialization
    # For each candidate rank r, try to find U, V, W such that the tensor equation holds
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try Strassen-style approach:
    # Partition A (3x4) into blocks and B (4x6) into blocks
    # A = [A11 A12] where A11 is 3x2, A12 is 3x2
    # B = [B11; B21] where B11 is 2x6, B21 is 2x6
    # C = A11*B11 + A12*B21
    # Each 3x2 * 2x6 multiplication has rank at most 3*2*6 = 36 schoolbook
    # But we can use Strassen on 2x2 subblocks...
    
    # Actually, let's try a more systematic approach:
    # The tensor for 3x4*4x6 has dimension 3*4*6 = 72
    # Lower bound is 24 (from max of slices)
    
    # Let's try random search for ranks from 24 upward
    best_result = make_schoolbook()
    
    # For small ranks, try structured random search
    for target_rank in range(24, 72):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            np.random.seed(42 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Check if this works (it won't, but this is a placeholder)
            # We need a better approach
            
            # Instead, let's use the fact that the tensor is all ones
            # T[i,j,k] = 1 for all i,j,k
            # This means we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # One simple approach: use rank = n*m*p with schoolbook
            # Another: try to exploit structure
            
            # For now, skip this and just return schoolbook
            pass
    
    # Actually, let's try a different approach:
    # The tensor T[i,j,k] = 1 for all i,j,k is a rank-1 tensor in the tensor sense!
    # Wait, no. T[i,j,k] = 1 means it's the all-ones tensor.
    # The all-ones tensor of size n*m*p has rank 1!
    # Because T[i,j,k] = 1 = 1*1*1, so U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # would give sum_r ... = 1 for all i,j,k.
    
    # Wait, that's not right either. Let me re-examine.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T satisfies: C[i,k] = sum_{i',j,k'} T[i',j,k'] * A[i',j] * B[j,k'] * delta_{i',i} * delta_{k',k}
    # Hmm, I'm confusing myself.
    
    # Let me think again. The standard formulation:
    # We have matrices A (n x m), B (m x p), C (n x p)
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The rank-r decomposition uses r "scalar multiplications"
    # Each scalar multiplication s_r is computed from linear combinations of entries of A and B
    # Then C[i,k] is a linear combination of the s_r's
    
    # In the tensor formulation:
    # T is a 3-tensor with indices (i, j, k) where:
    # - i indexes rows of A / rows of C
    # - j indexes columns of A / rows of B  
    # - k indexes columns of B / columns of C
    # 
    # T[i,j,k] = 1 if this term contributes to C, 0 otherwise
    # Actually T[i,j,k] = 1 for all valid i,j,k since each A[i,j]*B[j,k] contributes to C[i,k]
    
    # The rank-r decomposition means:
    # T[i,j,k] = sum_{r=0}^{R-1} U_r[i,j] * V_r[j,k] * W_r[k,i]
    # where U_r, V_r, W_r are the r-th factors
    
    # In the matrix form used here:
    # U has shape (n*m, R), so U[i*m+j, r] is the (i,j) entry of factor r
    # V has shape (m*p, R), so V[j*p+k, r] is the (j,k) entry of factor r  
    # W has shape (p*n, R), so W[k*n+i, r] is the (k,i) entry of factor r
    
    # So we need: for all i,j,k:
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This IS the all-ones tensor equation!
    # So T[i,j,k] = 1 for all i,j,k
    
    # The all-ones tensor of size n*m*p has rank 1!
    # Because we can set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_r ... = 1*1*1 = 1 for all i,j,k
    
    # Wait, but that can't be right for matrix multiplication.
    # Let me verify with a simple case: 1x1 * 1x1
    # A = [[a]], B = [[b]], C = [[a*b]]
    # C[0,0] = A[0,0] * B[0,0]
    # T[0,0,0] = 1
    # Rank-1: U[0,0]=1, V[0,0]=1, W[0,0]=1 gives 1*1*1=1. Correct!
    
    # For 2x2 * 2x2:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # If rank-1 works: U[i*2+j, 0]=1, V[j*2+k, 0]=1, W[k*2+i, 0]=1
    # Then sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1*1*1 = 1 for all i,j,k
    # So the tensor equation is satisfied!
    
    # But wait, this means the rank of the matrix multiplication tensor is 1?
    # That contradicts Strassen's result that rank(M_{2,2,2}) = 7.
    
    # I must be misunderstanding the tensor formulation.
    
    # Let me look at this more carefully. The issue is that the tensor T[i,j,k]
    # doesn't just encode "does this term exist" but also how the terms combine.
    
    # Actually, I think the correct formulation is:
    # The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to the (i,k) entry of C.
    # But the way they combine matters!
    
    # In the bilinear complexity framework:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is already a sum of m scalar multiplications for each (i,k) pair.
    # Total: n*m*p scalar multiplications.
    
    # The rank of the tensor is the minimum number of scalar multiplications needed.
    # The tensor T is defined such that its "slice" along the j-axis gives the multiplication pattern.
    
    # I think the correct interpretation is:
    # T is a 3-tensor where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k
    # BUT the rank decomposition must respect the bilinear structure!
    
    # The rank-r decomposition means:
    # There exist matrices U_r (n x m), V_r (m x p), W_r (n x p) such that:
    # C = sum_r (U_r * A) ⊙ (V_r * B) * W_r ... no, that's not right either.
    
    # Let me just go with the schoolbook decomposition which is known to be correct.
    # The rank is n*m*p = 72.
    
    return best_result
# EVOLVE-BLOCK-END
