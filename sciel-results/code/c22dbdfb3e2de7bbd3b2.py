import numpy as np
import time
import itertools
from copy import deepcopy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20

    def build_strassen_rank35():
        rank = 35
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        strassen = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
            ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
            ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        col = 0
        for block in range(5):
            for a, b, c in strassen:
                U[0, col] = a[0]
                U[1, col] = a[1]
                U[2, col] = a[2]
                U[3, col] = a[3]
                
                V[0*p + 2*block, col] = b[0]
                V[0*p + 2*block+1, col] = b[1]
                V[1*p + 2*block, col] = b[2]
                V[1*p + 2*block+1, col] = b[3]
                
                W[(2*block)*2 + 0, col] = c[0]
                W[(2*block)*2 + 1, col] = c[1]
                W[(2*block+1)*2 + 0, col] = c[2]
                W[(2*block+1)*2 + 1, col] = c[3]
                
                col += 1
        
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k*n+i,r] == delta_{j,k}"""
        rank = U.shape[1]
        # Build the full tensor check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    total = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    expected = 1.0 if j == k else 0.0
                    if abs(total - expected) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, target_rank, time_budget=10.0):
        """
        Try to reduce rank by finding linear dependencies among rank-1 terms.
        Each term r defines a rank-1 tensor T_r = u_r ⊗ v_r ⊗ w_r.
        If some subset of terms sums to zero (or can be eliminated), we can reduce rank.
        
        Strategy: Try to find a linear combination of columns that gives zero tensor.
        """
        rank = U.shape[1]
        if rank <= target_rank:
            return None
        
        # Check if any single term is zero
        for r in range(rank):
            if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
                new_U = np.delete(U, r, axis=1)
                new_V = np.delete(V, r, axis=1)
                new_W = np.delete(W, r, axis=1)
                if verify(new_U, new_V, new_W):
                    return (new_U, new_V, new_W)
        
        # Try pairs of terms that might cancel
        start = time.time()
        for r1 in range(rank):
            if time.time() - start > time_budget:
                break
            for r2 in range(r1 + 1, rank):
                # Check if term r1 and r2 are negatives of each other
                if np.allclose(U[:, r1], -U[:, r2], atol=1e-9) and \
                   np.allclose(V[:, r1], -V[:, r2], atol=1e-9) and \
                   np.allclose(W[:, r1], -W[:, r2], atol=1e-9):
                    # They cancel, remove both
                    keep = [i for i in range(rank) if i != r1 and i != r2]
                    new_U = U[:, keep]
                    new_V = V[:, keep]
                    new_W = W[:, keep]
                    if verify(new_U, new_V, new_W):
                        return (new_U, new_V, new_W)
        
        return None

    def structured_search(rank_target, time_budget=20.0):
        """
        Structured search for lower-rank decomposition.
        Uses the fact that the tensor has special structure.
        """
        start = time.time()
        
        # Key insight: The tensor T[i,j,k] = delta_{j,k} for i in {0,1}, j in {0,1}, k in {0,...,9}
        # This means for each (j,k) with j=k, we need T[0,j,j] = 1 and T[1,j,j] = 1
        # For j != k, T[i,j,k] = 0 for all i
        
        # A rank-1 term u ⊗ v ⊗ w contributes to T[i,j,k] the value u[i,j]*v[j,k]*w[k*n+i]
        # For this to be nonzero, we need u[i,j] != 0, v[j,k] != 0, w[k*n+i] != 0
        
        # Try a different structure: use terms where U depends only on i
        # If U[i,j,r] = alpha[i,r] (independent of j), then:
        # T[i,j,k] = sum_r alpha[i,r] * v[j,k,r] * w[k*n+i,r]
        # For j=k: T[i,j,j] = sum_r alpha[i,r] * v[j,j,r] * w[j*n+i,r] = 1
        # For j!=k: T[i,j,k] = sum_r alpha[i,r] * v[j,k,r] * w[k*n+i,r] = 0
        
        # This is a different constraint structure. Let's try small alpha patterns.
        
        # Try: alpha[0,r] and alpha[1,r] from small set {-1, 0, 1}
        # And v[j,k,r], w[k*n+i,r] from {-1, 0, 1}
        
        # Actually, let's try a more systematic approach:
        # The minimal rank for 2x2x2 is 7 (Strassen), so for 2x2x10 we expect at least 7*5=35?
        # No! The blocks are not independent because U is shared.
        
        # Let me think differently. The tensor is:
        # T[i,j,k] = delta_{j,k}
        # This is actually a "diagonal" tensor in the (j,k) indices.
        
        # For fixed i, the tensor T[i,:,:] is a 2x10 matrix that's identity-like (only diagonal entries = 1).
        # But it's the same for i=0 and i=1!
        
        # So T[0,:,:] = T[1,:,:] = diag(1,1,0,0,0,0,0,0,0,0) [in j,k space]
        # Wait, no. T[i,j,k] = delta_{j,k}, so for fixed i, it's the 2x10 matrix with 1s at (0,0) and (1,1).
        
        # This means the tensor has rank at most 2! Because:
        # T[i,j,k] = delta_{j,k} = sum over s of delta_{j,s} * delta_{s,k}
        # But that's not a rank-1 decomposition in the tensor sense.
        
        # Actually, the tensor T[i,j,k] = delta_{j,k} can be written as:
        # T[i,j,k] = sum_{s=0}^{1} delta_{j,s} * delta_{s,k} * 1_i
        # where 1_i = 1 for all i.
        
        # So: u_s[i,j] = delta_{j,s}, v_s[j,k] = delta_{s,k}, w_s[k*n+i] = 1
        # That gives rank 2!
        
        # Let me verify: for s=0: u_0[i,0]=1, u_0[i,1]=0; v_0[0,k]=delta_{0,k}; w_0[k*n+i]=1
        # Contribution to T[i,j,k]: u_0[i,j] * v_0[j,k] * w_0[k*n+i]
        # = delta_{j,0} * delta_{0,k} * 1 = delta_{j,0} * delta_{k,0}
        
        # For s=1: contribution = delta_{j,1} * delta_{k,1}
        
        # Total: delta_{j,0}*delta_{k,0} + delta_{j,1}*delta_{k,1} = delta_{j,k} for j,k in {0,1}
        # But for k >= 2, delta_{j,k} = 0, and our sum gives 0. Good.
        
        # So rank 2 suffices?! Let me double-check the tensor definition.
        
        # The matrix multiplication tensor for C = A*B where A is 2x2, B is 2x10:
        # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
        # The tensor T[i,j,k] satisfies C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k], else 0.
        # Indeed, T[i,j,k] = 1 for all i,j,k where the multiplication is defined, i.e., T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,9}.
        
        # Wait, I was wrong! T[i,j,k] is NOT delta_{j,k}. 
        # T[i,j,k] = 1 for ALL i, j, k in the valid range!
        # Because C[i,k] = sum_j A[i,j] B[j,k], so every triple (i,j,k) contributes.
        
        # So the tensor is the all-ones tensor of shape (2, 2, 10)!
        # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,9}
        
        # The rank of the all-ones tensor of shape (n,m,p) is 1!
        # u[i,j] = 1, v[j,k] = 1, w[k*n+i] = 1
        # Then sum_r u[i,j,r]*v[j,k,r]*w[k*n+i,r] = 1*1*1 = 1 = T[i,j,k]
        
        # So the answer is RANK 1?!
        
        # Let me re-read the problem statement carefully.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 10 matrix."
        
        # The matrix multiplication tensor T[i,j,k] for C = A*B is defined such that:
        # C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k.
        
        # Therefore, the rank is 1, with:
        # U[i*m+j, 0] = 1 for all i,j
        # V[j*p+k, 0] = 1 for all j,k  
        # W[k*n+i, 0] = 1 for all k,i
        
        # Let me verify this is correct.
        
        return None  # Placeholder, we'll handle rank 1 separately

    # First, let's test the rank-1 hypothesis
    U1 = np.ones((nm, 1), dtype=float)
    V1 = np.ones((mp, 1), dtype=float)
    W1 = np.ones((pn, 1), dtype=float)
    
    if verify(U1, V1, W1):
        # Rank 1 works!
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    
    # If rank 1 doesn't work (maybe I misunderstood the tensor definition),
    # fall back to Strassen rank 35
    U, V, W = build_strassen_rank35()
    
    # Try to reduce rank
    best_U, best_V, best_W = U, V, W
    best_rank = 35
    
    # Try greedy reduction
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        reduced = try_reduce_rank(best_U, best_V, best_W, best_rank - 1, time_budget=2.0)
        if reduced is None:
            break
        best_U, best_V, best_W = reduced
        best_rank = best_U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
