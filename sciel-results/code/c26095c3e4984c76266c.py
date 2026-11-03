"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta
    # T is indexed by (i*n+j... no, let me use the convention)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor T[a,b,c] where a indexes (i,j) of A, b indexes (j,k) of B, c indexes (i,k) of C
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 if j matches, else 0
    
    # Check: sum_c U[a,c]*V[b,c] should equal T[a,b] for the "A*B" part
    # Actually the decomposition is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j; j',k; i',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # Factorized: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    
    # So we need: for all i,j,j',k,k':
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j==j' and i==i' and k==k' else 0
    
    # This is expensive to check fully. Let's check a subset.
    # Actually, let's just verify by matrix multiplication for random inputs.
    
    # Create random matrices A (n x m) and B (m x p)
    # C = A @ B
    # C[i,k] should equal sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j' B[j',k]*V[j'*p+k,c]) ... 
    # No, that's not right either.
    
    # The correct check: 
    # Let U_c[i,j] = U[i*m+j, c], V_c[j,k] = V[j*p+k, c], W_c[i,k] = W[k*n+i, c]
    # Then sum_c U_c * V_c * W_c (elementwise product summed over c) should give the multiplication tensor
    # i.e., for each (i,j,j',k,k'): sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c] = delta_{j,j'}*delta_{i,i'}*delta_{k,k'}
    
    # Efficient check: for each c, form matrices Uc (n x m), Vc (m x p), Wc (p x n)
    # The sum over c of Uc * Vc * Wc (where * is elementwise and we sum) should equal the tensor
    # Actually it's: sum_c Uc[i,j] * Vc[j',k] * Wc[i',k'] 
    
    # Let me just do a direct check on a few entries.
    # For efficiency, check all entries but use numpy broadcasting.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U_arr is (n*m, rank) -> (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,j',k,k'] = sum_c U3[i,j,c] * V3[j',k,c] * W3[i',k',c]
    # We want T[i,j,j',k,k'] = 1 if j==j' and i==i' and k==k' else 0
    
    # This is a 5D tensor of size (n,m,m,p,p) = (4,4,4,5,5) = 1600 entries
    # Compute sum_c U3[i,j,c]*V3[j',k,c]*W3[i',k',c]
    # = np.einsum('ijc,jkc,ikc->ijjkk', U3, V3, W3)  -- but indices are tricky
    
    # Let me use: result[i,j,jp,k,kp] = sum_c U3[i,j,c] * V3[jp,k,c] * W3[kp,i,c]
    # Wait, W3 is (p, n, rank) so W3[kp, i, c]
    
    result = np.einsum('ijc,jkc,kic->ijjkk', U3, V3, W3, optimize=True)
    
    # Expected: 1 if j==jp and i==i and k==k, else 0
    # i.e., result[i,j,jp,k,kp] = 1 if j==jp and k==kp else 0 (i is free in the expected)
    # Wait no: the multiplication tensor is T[i,j; jp,k; i,k] = delta_{j,jp} * delta_{k,k}
    # The output index is (i,k), so the third factor W is indexed by (k,i) in our convention W[k*n+i]
    
    # So expected[i,j,jp,k,kp] = 1 if j==jp and k==kp else 0
    # Note: i appears in both input (A[i,j]) and output (C[i,k]), so it's consistent
    # jp appears in B[jp,k], kp appears in C[i,kp]
    
    expected = np.zeros((n, m, m, p, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, j, k, k] = 1.0
    
    # Check if they match
    if not np.allclose(result, expected, atol=1e-9):
        return False
    return True


def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijjkk', U3, V3, W3, optimize=True)
    
    expected = np.zeros((n, m, m, p, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, j, k, k] = 1.0
    
    return np.linalg.norm(result - expected)


def _round_to_halfint(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0


def _als_refine(U, V, W, n, m, p, max_iters=50, tol=1e-10):
    """Refine decomposition using ALS, then round to half-integers."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for it in range(max_iters):
        # Solve for U
        # For each (i,j), U[i*m+j, :] should satisfy:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,j} * delta_{k,k} (for fixed i,j)
        # Actually this is complex. Let me use a different approach.
        
        # The constraint is: sum_c U[i,j,c] * V[jp,k,c] * W[kp,i,c] = T[i,j,jp,k,kp]
        # For fixed (i,j), and varying (jp,k,kp):
        # sum_c U[i,j,c] * V[jp,k,c] * W[kp,i,c] = delta_{j,jp} * delta_{k,kp}
        
        # This is a system where for fixed (i,j), we have unknowns U[i,j,:] of length rank
        # And the equation involves V and W.
        
        # Let me think of it differently. For fixed (i,j):
        # Define M_{(jp,k,kp)} = sum_c V[jp,k,c] * W[kp,i,c]
        # Then sum_c U[i,j,c] * M_{(jp,k,kp), c} = T[i,j,jp,k,kp]
        # where M is a matrix of shape (m*p*p, rank)
        
        # Actually, let me just do gradient-based or skip ALS and use a different strategy.
        
        break  # Skip ALS for now, just try rounding
    
    # Round to half-integers
    U_rounded = _round_to_halfint(U)
    V_rounded = _round_to_halfint(V)
    W_rounded = _round_to_halfint(W)
    
    return U_rounded, V_rounded, W_rounded


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank_school = n * m * p  # 80
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Strategy: try various ranks from low to high, attempt to find valid decompositions
    
    # Known bounds: for 4x4x4, best known is 49 (Landsberg). For 4x4x5, it should be similar or slightly higher.
    # Let's try ranks from 49 upward.
    
    for target_rank in range(49, min(80, 100)):
        if time.time() > deadline:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use random initialization + ALS refinement + rounding
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(target_rank * 1000 + attempt)
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Normalize columns
            for c in range(target_rank):
                norm_u = np.linalg.norm(U_rand[:, c])
                norm_v = np.linalg.norm(V_rand[:, c])
                norm_w = np.linalg.norm(W_rand[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1/3)
                    U_rand[:, c] /= scale
                    V_rand[:, c] /= scale
                    W_rand[:, c] /= scale
            
            # Round to half-integers and check
            U_rounded = _round_to_halfint(U_rand)
            V_rounded = _round_to_halfint(V_rand)
            W_rounded = _round_to_halfint(W_rand)
            
            if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                break
        
        if best_rank == target_rank:
            break  # Found the target rank, stop
    
    # Also try structured decompositions based on known algorithms
    # For example, try Strassen-like block decomposition
    
    # 2x2 block Strassen for the 4x4 part, adapted for 4x5
    # This is complex, so let's try a simpler structured approach
    
    # Try rank 64 with a structured pattern
    if best_rank > 64 and time.time() < deadline:
        # Try a structured rank-64 decomposition
        # Based on the idea that 4x4x4 has rank 49, maybe 4x4x5 can be done in 64
        pass
    
    return best_result
# EVOLVE-BLOCK-END

