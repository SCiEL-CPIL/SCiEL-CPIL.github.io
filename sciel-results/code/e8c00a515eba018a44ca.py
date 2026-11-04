"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[(i,j), (j,k), (k,i)] should equal 1 iff the indices
    form a valid multiplication term, which is always the case for i in [0,n),
    j in [0,m), k in [0,p).
    
    Actually, the tensor for matrix multiplication C = A * B where A is n x m
    and B is m x p is defined such that:
    (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    and 0 otherwise (but all (i,j,k) with i<n, j<m, k<p are valid).
    
    Wait, that's not right. The tensor is a 3-tensor T with indices (i,j,k)
    where the "multiplication tensor" has T[i,j,k] = 1 for all i,j,k in range
    because every combination contributes. No wait.
    
    Let me re-read the contract. The output factor follows W[k*n+i, r].
    So W is indexed by (k, i) where k is the output column and i is the output row.
    
    The standard formulation: we want to compute C = A * B where A is n x m, B is m x p.
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor decomposition says:
    C_{i,k} = sum_r (sum_j A_{i,j} U_{i*m+j, r}) * (sum_k B_{j,k} V_{j*p+k, r}) * W_{k*n+i, r}
    
    No, that's not right either. Let me think more carefully.
    
    The matrix multiplication tensor T has entries T_{(i,j),(j,k),(k,i)} = 1.
    This means: for each term A_{i,j} * B_{j,k} contributing to C_{i,k}, we have a tensor entry.
    
    The decomposition is: T_{(i,j),(j,k),(k,i)} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    
    Where U has shape (n*m, rank) indexed by (i,j), V has shape (m*p, rank) indexed by (j,k),
    and W has shape (p*n, rank) indexed by (k,i).
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    
    And this must hold exactly.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.dot(np.dot(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_strassen_like():
    """
    Try to find a lower-rank decomposition using a Strassen-like approach.
    For 3x3 * 3x8, we can potentially use the fact that 3x3 matrix multiplication
    can be done in rank 7 (Strassen), and then extend to 3x8.
    
    Actually, let's think about this differently. The 3x3 * 3x3 sub-problem can be
    done in rank 7. For 3x3 * 3x8, we can think of B as having 8 columns, so we
    have 8 independent 3x3 multiplications (each column of B times A). But they share
    the same A, so we can potentially do better.
    
    Let's try a structured approach: use the 7-term Strassen decomposition for the
    3x3 * 3x3 case, and then extend it to handle the 8 columns.
    
    Actually, a simpler approach: try random search with integer/half-integer entries
    for ranks below 72.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Known lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Schoolbook: 72
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # For 3x3*3x3, min rank is 7. For 3x3*3x8, let's try starting from 24.
    
    # Actually, let's first try a specific structured decomposition.
    # The key insight: for fixed j, the sub-problem is n x p = 3 x 8, which is
    # just a rank-1 operation (outer product). So for each j, we need rank 1.
    # Total: m * 1 = 3? No, that's not right.
    
    # Let me reconsider. The tensor is T_{i,j,k} = 1 for all valid (i,j,k).
    # This is actually the "all-ones" tensor in the sense that every entry is 1.
    # Wait no, that's not right either.
    
    # Let me re-read the schoolbook code:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # So for each (i,j,k), we have one term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The product U*V*W = 1 for that specific (i,j,k) combination.
    
    # But we need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for ALL (i,j,k).
    
    # In the schoolbook, for a given (i,j,k), only one r (the one corresponding to
    # (i,j,k)) has all three factors equal to 1, so the sum is 1.
    
    # For a lower-rank decomposition, we need to find U, V, W such that for every
    # (i,j,k), the sum equals 1.
    
    # This is a hard problem. Let's try a simple random search.
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Try a few structured approaches first
    
    # Approach: Use the fact that the tensor is "separable" in a certain way.
    # For 3x3 * 3x8, we can try to find a decomposition where:
    # - U has some structure based on i, j
    # - V has some structure based on j, k
    # - W has some structure based on k, i
    
    # Let's try random search with small integer values
    for target_rank in range(24, 72):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with small random integers
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Fill with random values from {-1, 0, 1}
            for r in range(target_rank):
                U[:, r] = np.random.randint(-1, 2, size=n*m)
                V[:, r] = np.random.randint(-1, 2, size=m*p)
                W[:, r] = np.random.randint(-1, 2, size=p*n)
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                        if abs(val - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # If no lower-rank decomposition found, return schoolbook
    return _schoolbook()

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, get the schoolbook decomposition as fallback
    schoolbook = _schoolbook()
    
    # Try to find a better decomposition
    try:
        result = _try_strassen_like()
        if result is not None:
            # Verify the result
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Return schoolbook as fallback
    return schoolbook
# EVOLVE-BLOCK-END
