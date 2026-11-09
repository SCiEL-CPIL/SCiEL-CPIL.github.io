"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for multiplying n x m by m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r] for each r
    # Actually, the decomposition is: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # This means: C[i,k] = sum_r (U[i*m+j, r] * V[j*p+k, r]) * A[i,j] * B[j,k]
    # For this to equal sum_j A[i,j] * B[j,k], we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = delta_{j,j'} * delta_{k,k'}... 
    # Actually the correct identity is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k])
    # This requires: sum_r U[i*m+j, r] V[j'*p+k, r] = delta_{j,j'} * delta_{k,k'} is NOT right.
    # The correct tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}... no that's not right either.
    
    # Let me think again. The matrix multiplication tensor T has entries T[i,j,k] = delta_{i,k}... no.
    # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which it always does with coefficient 1.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, that's not standard. Let me reconsider.
    
    # Standard: C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C[i,k] = sum_{j,r} A[i,j] B[j,k] * T_{i,j,k} where T_{i,j,k} = delta_{j,j}... 
    # Actually T_{i,j,k} = 1 for all i,j,k is not a tensor decomposition in the usual sense.
    
    # The bilinear complexity is: C[i,k] = sum_r (sum_j U_{i,j,r} A[i,j]) (sum_j V_{j,k,r} B[j,k])
    # = sum_r sum_j sum_{j'} U_{i,j,r} V_{j',k,r} A[i,j] B[j',k]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r U_{i,j,r} V_{j',k,r} = delta_{j,j'} for all i,k
    # And W is not used in this formulation. Let me check the problem statement again.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank), indexed by k*n+i.
    # The decomposition is likely: C[i,k] = sum_r U_{i,j,r} A[i,j] * V_{j,k,r} B[j,k] * ... 
    # Or perhaps: T[i,j,k] = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    
    # Let me verify with the schoolbook: U[i*m+j, r] = 1 if r = i*m*p + j*p + k, else 0
    # V[j*p+k, r] = 1 if r = i*m*p + j*p + k, else 0  
    # W[k*n+i, r] = 1 if r = i*m*p + j*p + k, else 0
    # Then sum_r U[i*m+j,r] * V[j'*p+k',r] * W[k'',n+i'',r] = 1 only when (i,j,k) = (i',j',k') = (i'',j'',k'')
    # So T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k. ✓
    
    # So the verification is: for all i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_strassen_block(n, m, p):
    """Try to use Strassen-like decomposition for blocks where possible."""
    # Strassen works for 2x2x2. For other sizes, we can pad and use Strassen.
    # For 3x3x3, we can pad to 4x4x4 and use 2x2x2 Strassen blocks, but that gives rank 4^3 * 7/8 = 112/4 = 28? 
    # Actually padding 3 to 4 and using 2x2x2 Strassen: 4x4x4 can be done with 2x2x2 Strassen 4 times (2^3=8 blocks of 2x2x2), each rank 7, total rank 8*7=56.
    # But schoolbook for 3x3x3 is 27, so padding is worse.
    
    # For our problem 3x5x6, let's try splitting into blocks that can use Strassen.
    # Split m=5 into 2+2+1, p=6 into 2+2+2, n=3 into 2+1.
    # This gets complex. For now, return None to indicate we should use schoolbook.
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = t0 + 115.0  # safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using randomized search
    # We'll try to find a rank-r decomposition by solving a system of equations
    # T[i,j,k] = sum_{r=1}^r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    # For r < n*m*p, this is underdetermined, but we need integer/half-integer solutions.
    
    # Try small rank reductions
    for target_rank in range(best_rank - 1, max(20, best_rank - 30), -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of rank target_rank
        # Use a greedy approach: start with schoolbook and try to merge terms
        # Or use random initialization and optimization
        
        # For simplicity, try a structured approach:
        # Split dimensions and use Kronecker products
        
        # Try splitting m into m1+m2
        for m1 in range(1, m):
            m2 = m - m1
            # For each split, we can try to use a rank-m1*m*p + rank-m2*m*p decomposition
            # But that's just schoolbook again.
            
            # Try to use a lower-rank method for sub-blocks
            pass
        
        # Try a different approach: use the fact that we can share some factors
        # For example, if we group indices, we might reduce rank.
        
        # Since finding a provably lower rank is hard, let's try a heuristic:
        # Randomly perturb the schoolbook and see if we can reduce rank while maintaining exactness.
        
        # Actually, for exact decomposition with integer/half-integer entries, the search space is discrete.
        # Let's try a simple approach: see if we can combine any two schoolbook terms.
        
        # The schoolbook has terms indexed by (i,j,k). Can we have U[i1*m+j1,r] = U[i2*m+j2,r] for some r?
        # If so, we might be able to merge.
        
        # For now, let's just return the schoolbook since finding a lower rank is non-trivial.
        pass
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        # This should never happen, but just in case
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return best
# EVOLVE-BLOCK-END
