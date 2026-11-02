"""Tensor decomposition search for 4x4 * 4x8 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _make_schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, schoolbook rank n*m*p."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    rank = len(U[0])
    Um = np.array(U, dtype=float)  # (n*m, rank)
    Vm = np.array(V, dtype=float)  # (m*p, rank)
    Wm = np.array(W, dtype=float)  # (p*n, rank)
    # Build left-hand side: for each c, U[:,c] (n*m) * V[:,c] (m*p) -> outer -> reshape
    # LHS[i,j,k,c] = U[i*m+j,c]*V[j*p+k,c]
    # We need: for each (i,j,k), the vector over c equals W[k*n+i, :]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = Um.reshape(n, m, rank)
    Vr = Vm.reshape(m, p, rank)
    Wr = Wm.reshape(p, n, rank)
    # LHS[i,j,k,c] = Ur[i,j,c]*Vr[j,k,c]
    # Sum is not needed; it's elementwise product per c.
    # We want Ur[i,j,:] * Vr[j,k,:] == Wr[k,i,:]  (elementwise over c)
    # Broadcast: Ur (n,m,rank) * Vr (m,p,rank) -> need to align j
    # Ur[i,j,c] * Vr[j,k,c]: result shape (n, m, p, rank)
    lhs = Ur[:, :, None, :] * Vr[None, :, :, :]  # (n, m, p, rank)
    rhs = Wr[None, None, :, :]  # (1, 1, p, n, rank) -> broadcast to (n,m,p,rank)
    rhs = Wr.transpose(2, 0, 1)  # (p, n, rank)
    rhs = rhs[None, None, :, :, :]  # (1,1,p,n,rank)
    # Compare lhs (n,m,p,rank) with rhs (1,1,p,n,rank): need lhs[i,j,k,c] == rhs[k,i,c]
    # Reshape lhs to (n, m, p, rank), rhs to (n, m, p, rank) by broadcasting
    # rhs[k, i, c] -> we need to index: for each (i,j,k,c): rhs[k, i, c]
    # Create rhs_full: (n, m, p, rank) where rhs_full[i,j,k,c] = Wr[k, i, c]
    rhs_full = Wr[None, None, :, :, :]  # (1, 1, p, n, rank)
    # We need to select: for each i, j, k, c: rhs_full[i, j, k, i, c] -- but j is not in rhs
    # Actually rhs doesn't depend on j, so broadcast over j.
    # rhs_full shape (1,1,p,n,rank). We want (n,m,p,rank).
    # rhs_full[k,i,c] for all i,j,k,c -> use indexing
    # Let's do it differently:
    lhs_np = Ur[:, :, None, :] * Vr[None, :, :, :]  # (n, m, p, rank)
    # rhs: Wr[k, i, c] -> shape (p, n, rank) -> need (n, m, p, rank)
    # rhs[i, j, k, c] = Wr[k, i, c]
    rhs_np = Wr[:, :, :]  # (p, n, rank)
    # Expand: rhs_np[k, i, c] -> we want to broadcast to (n, m, p, rank)
    # rhs_np_trans = rhs_np.transpose(1, 0, 2)  # (n, p, rank)
    # Then rhs_full[i, j, k, c] = rhs_np_trans[i, k, c]
    rhs_np_trans = rhs_np.transpose(1, 0, 2)  # (n, p, rank)
    rhs_full = rhs_np_trans[:, None, :, :]  # (n, 1, p, rank) -> broadcast to (n, m, p, rank)
    diff = np.abs(lhs_np - rhs_full)
    return bool(np.all(diff < tol))

def _try_kronecker_decomposition(n, m, p):
    """
    Try to find a Kronecker-structured decomposition.
    For matrix multiplication, if we can write the indices in a factored way,
    we might reduce rank.
    
    For 4x4 * 4x8: n=4, m=4, p=8.
    Note p = 2*n. Try to see if we can exploit this.
    
    One idea: if m divides p or n divides p, we might use a block structure.
    """
    # Schoolbook as baseline
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Try: can we factor the multiplication tensor using the structure?
    # The tensor T[i,j,k] = delta (it's the identity in the sense that (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the multiplication tensor has entries T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Wait no. The tensor for matrix multiplication: T is a 3-tensor where
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k] = 1
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a 3-dimensional identity tensor (all ones).
    # The rank of the all-ones tensor of size n*m x m*p x p*n... 
    # Actually it's not a standard identity, it's the "all ones" 3-tensor.
    # The tensor rank of the all-ones 3-tensor of dimensions a x b x c is max(a,b,c) if we can use arbitrary entries.
    # But here the dimensions are coupled: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    # And the constraint is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let me re-examine. The constraint couples i,j,k in specific ways through the index mappings.
    # U depends on (i,j), V on (j,k), W on (k,i).
    
    # This is exactly the matrix multiplication tensor. Its rank is what we're trying to minimize.
    
    # Known results: for 4x4 * 4x8, the rank is at most 4*4*8 = 128 (schoolbook).
    # Can we do better? 
    
    # Let's try a simple structured approach:
    # If we set U[i,j,c] = a[i,c]*b[j,c], V[j,k,c] = b[j,c]*c[k,c], W[k,i,c] = c[k,c]*a[i,c]
    # Then product = a[i,c]*b[j,c]*b[j,c]*c[k,c]*c[k,c]*a[i,c] = a[i,c]^2 * b[j,c]^2 * c[k,c]^2
    # Sum over c: sum_c a[i,c]^2 * b[j,c]^2 * c[k,c]^2 = 1 for all i,j,k
    # This would require a[i,c]^2 * b[j,c]^2 * c[k,c]^2 to factor as a product of separate functions.
    # If we set a[i,c] = alpha[i,c], b[j,c] = beta[j,c], c[k,c] = gamma[k,c]
    # Then sum_c alpha[i,c]^2 * beta[j,c]^2 * gamma[k,c]^2 = 1
    # This factors if alpha[i,c]^2 * beta[j,c]^2 * gamma[k,c]^2 = f[i]*g[j]*h[k] for each c,
    # which requires alpha[i,c]^2 = f[i]*d[c], beta[j,c]^2 = g[j]*e[c], gamma[k,c]^2 = h[k]*d[c]... 
    # This is getting complicated. Let me just do a search.
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Verify schoolbook
    assert _verify(U, V, W, n, m, p), "Schoolbook verification failed"
    
    # Randomized search: try to find lower rank decompositions
    # Strategy: for a given rank r, try random rational entries and check if they satisfy the constraints.
    # The constraints are: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 128 equations in r*(n*m + m*p + p*n) = r*(16+32+32) = 80r unknowns.
    # For r < 128, we have more unknowns than equations potentially.
    
    # Better approach: use the structure. 
    # Let's try: U[i,j,c] = u_i[c] * v_j[c], V[j,k,c] = v_j[c] * w_k[c], W[k,i,c] = w_k[c] * u_i[c]
    # (cyclic pattern)
    # Then: sum_c u_i[c]^2 * v_j[c]^2 * w_k[c]^2 = 1 for all i,j,k
    # This means: sum_c (u_i[c]*v_j[c]*w_k[c])^2 = 1
    # For this to hold for all i,j,k, we need the vectors to be orthogonal in a specific way.
    
    # Alternative: try to find a decomposition where the rank equals max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # or some other structured value.
    
    # Let's try a more systematic approach:
    # The multiplication tensor T[i,j,k] = 1 (all ones) with the specific index coupling.
    # We want: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Try rank = 32 (which is m*p = p*n):
    # Set W[k*n+i, c] = 1 for c = k (if k < 32) ... this is getting messy.
    
    # Let me try a different structured approach:
    # For the all-ones tensor with coupled indices, one known decomposition uses
    # the fact that if we can write 1 = sum_c f_c(i) g_c(j) h_c(k) where
    # f_c depends on i, g_c on j, h_c on k, and the index mappings work out.
    
    # Actually, let me just do a randomized search with small rational entries.
    
    random.seed(42)
    
    for target_rank in range(32, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random integers
            U_try = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V_try = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W_try = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U_try, V_try, W_try, n, m, p):
                best_rank = target_rank
                best_result = (U_try, V_try, W_try)
                break
        
        if best_rank <= target_rank:
            break
    
    # Also try a more structured search: Kronecker-like
    # For 4x4 * 4x8, try to see if rank 64 works (half of schoolbook)
    # by exploiting the p=2n structure.
    
    U, V, W = best_result
    
    # Final verification
    assert _verify(U, V, W, n, m, p), "Final verification failed"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

