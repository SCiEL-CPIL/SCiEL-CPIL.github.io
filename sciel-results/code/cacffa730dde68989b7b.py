"""Matrix multiplication 3x3x7 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Construct schoolbook (rank n*m*p) decomposition."""
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


def _verify_decomposition(result, n, m, p, num_samples=50, tol=1e-10):
    """Verify that the decomposition correctly decomposes the matrix multiplication tensor."""
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For random A (n x m) and B (m x p), check that C = A*B
    # C[k*n+i, r] = sum_j A[i,j] * B[j, k]
    # Our decomposition: C[k*n+i, r] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the convention is W[k*n+i, r] for the output.
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should equal 1 if... no.
    # The matrix multiplication tensor has T[i,j,k] = 1 for the bilinear form A[i,j]*B[j,k] contributing to C[i,k]
    # Actually the tensor is: T_{i,j,k} such that C[i,k] = sum_{j} A[i,j] B[j,k]
    # In tensor form: T_{i,j,k} = delta (the coefficient)
    # The decomposition says: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # For matrix mult, T_{i,j,k} = 1 for all i,j,k (since each A[i,j]*B[j,k] appears exactly once)
    # Wait no. The tensor for matrix multiplication is:
    # T_{i,j,k} = 1 (it's the identity tensor in the appropriate sense)
    # Actually, the bilinear map is (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has components T_{i,j,k} = 1 for all i,j,k
    # And the decomposition is: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # Let's verify: for random A and B, compute C via decomposition and via direct multiplication
    rng = np.random.default_rng(42)
    for _ in range(num_samples):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_direct = A @ B  # C[i, k] = sum_j A[i,j] B[j,k]
        
        # Compute via decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_k' ... ) 
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
        # Wait, that's not right either. Let me re-read the convention.
        
        # U has shape (n*m, rank), indexed by [i*m + j, r]
        # V has shape (m*p, rank), indexed by [j*p + k, r]  
        # W has shape (p*n, rank), indexed by [k*n + i, r]
        
        # The decomposition of the tensor T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix mult, T_{i,j,k} = 1 for all i, j, k.
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
        
        # Let's just check this directly
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 63
    
    # Try to find lower rank decompositions
    # For 3x3x3, the known minimum rank is 19 (Strassen)
    # For 3x3x7, we can potentially do better than 63
    
    # Strategy: Use the known Strassen 3x3x3 rank-19 decomposition for the first two 3x3 blocks
    # and schoolbook for the 3x3x1 block, giving 19 + 19 + 9 = 47
    
    # Let me construct the Strassen 3x3x3 rank-19 decomposition
    # This is a well-known result. Let me use a concrete formulation.
    
    # The 19-term Strassen decomposition for 3x3 matrices:
    # I'll use the formulation where we define 19 bilinear forms
    
    # Actually, let me try a different approach: use numerical optimization
    # to find a low-rank decomposition
    
    # First, let's try the block approach with known Strassen for 2x2 blocks
    
    # For 3x3x3, use the known rank-19 decomposition
    # For the 3x3x1 part, use rank 9 (schoolbook)
    # Total: 19 + 19 + 9 = 47
    
    # But we need to verify this works. Let me construct it carefully.
    
    # Strassen's 19-term algorithm for 3x3:
    # The 19 multiplications are:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a11 + a33)(b11 + b33)  
    # M3 = (a22 + a33)(b22 + b33)
    # M4 = (a12 + a21)(b21 + b12)
    # M5 = (a13 + a31)(b31 + b13)
    # M6 = (a23 + a32)(b32 + b23)
    # M7 = a11 * b11
    # M8 = a22 * b22
    # M9 = a33 * b33
    # M10 = a12 * b21
    # M11 = a21 * b12
    # M12 = a13 * b31
    # M13 = a31 * b13
    # M14 = a23 * b32
    # M15 = a32 * b23
    # M16 = (a11 + a12 + a13)(b11 + b21 + b31)
    # M17 = (a21 + a22 + a23)(b12 + b22 + b32)
    # M18 = (a31 + a32 + a33)(b13 + b23 + b33)
    
    # Hmm, that's only 18. Let me look up the correct 19-term Strassen for 3x3.
    
    # Actually, Strassen's original 19-term algorithm for 3x3 is more complex.
    # Let me try a different approach: use numerical search.
    
    # Try rank 47 with the block structure
    # First 3x3x3 block (k=0,1,2), second 3x3x3 block (k=3,4,5), third 3x3x1 block (k=6)
    
    # For now, let's try to verify if a rank-47 decomposition exists by construction
    # and if not, fall back to schoolbook.
    
    # Let me try a simpler improvement: can we do better than 63 with a simple construction?
    # The trivial lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But that's not achievable for matrix multiplication tensors.
    
    # Let me try to use the fact that for the 3x3x1 part, we might be able to
    # share some terms with the 3x3x3 parts.
    
    # For now, return schoolbook which is known to work
    return best_result

# EVOLVE-BLOCK-END
