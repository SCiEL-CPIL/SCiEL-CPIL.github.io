import numpy as np
import time
from itertools import combinations

def construct():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Partitions into 8 independent 2x2 * 2x2 sub-problems using Strassen (rank 7 each).
    Output convention: W[k*n + i, r] where k is column index, i is row index.
    
    Includes a time-bounded rank-reduction search that attempts to find a
    decomposition with rank < 56 by identifying linearly dependent columns
    in the U, V, W factor matrices.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2*2x2
    col = 0
    for s in range(2):
        for j in range(4):
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # U side
            U[a00, col] += 1;   U[a11, col] += 1
            U[a10, col+1] += 1; U[a11, col+1] += 1
            U[a00, col+2] += 1
            U[a11, col+3] += 1
            U[a00, col+4] += 1; U[a01, col+4] += 1
            U[a10, col+5] += 1; U[a00, col+5] += -1
            U[a01, col+6] += 1; U[a11, col+6] += -1
            
            # V side
            V[b00, col] += 1;   V[b11, col] += 1
            V[b00, col+1] += 1
            V[b01, col+2] += 1; V[b11, col+2] += -1
            V[b10, col+3] += 1; V[b00, col+3] += -1
            V[b11, col+4] += 1
            V[b00, col+5] += 1; V[b01, col+5] += 1
            V[b10, col+6] += 1; V[b11, col+6] += 1
            
            # W side
            W[c00, col] += 1;   W[c00, col+3] += 1; W[c00, col+4] += -1; W[c00, col+6] += 1
            W[c01, col+2] += 1; W[c01, col+4] += 1
            W[c10, col+1] += 1; W[c10, col+3] += 1
            W[c11, col] += 1;   W[c11, col+1] += -1; W[c11, col+2] += 1; W[c11, col+5] += 1
            
            col += 7
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # 110s safety margin before 120s limit
    
    # Attempt rank reduction by finding linearly dependent columns
    # The idea: if some columns in U, V, W are linearly dependent in a way
    # that preserves the tensor identity, we can eliminate them.
    
    def _validate(U, V, W, n=2, m=4, p=8):
        """Check that the decomposition exactly reproduces matrix multiplication."""
        rank = U.shape[1]
        # For the tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # should equal delta_{i,j}... no.
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # The decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * V[j*p+k,r] * W[k*n+i,r]
        # Wait, that's not right either. Let me think again.
        # 
        # Standard form: The multiplication tensor T has T[i,j,k] = 1 if we're computing
        # C[i,k] = sum_j A[i,j]B[j,k]. The tensor is T_{ijk} = delta_{j,j} (trivial).
        # 
        # In the U,V,W decomposition: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # where U is indexed by (i,j), V by (j,k), W by (k,i) [or (k*n+i)].
        # 
        # So for all i in [0,n), j in [0,m), k in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # (since the only non-zero term in the multiplication tensor is when the middle index matches)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def _try_reduce_rank(U, V, W, max_rank_to_try, deadline):
        """Try to reduce rank by eliminating linearly dependent columns."""
        current_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        for target_rank in range(current_rank - 1, max_rank_to_try - 1, -1):
            if time.time() > deadline:
                break
            
            # Try to find which column to eliminate
            # A column r can be eliminated if its contribution can be absorbed
            # by other columns. This happens when the column vectors in U, V, W
            # are linearly dependent in a specific way.
            
            # Simple approach: try zeroing out one column at a time and check
            # if the tensor identity still holds (it won't, but we can try
            # to find a linear combination that works).
            
            # More practical: use SVD or rank factorization to find if
            # the set of rank-1 tensors is linearly dependent.
            
            # For now, try a greedy approach: check if removing column r
            # and adjusting others can work. This is complex, so we'll
            # do a simple check: if column r in U is all zeros, it can be removed.
            
            removable = []
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                    removable.append(r)
            
            if removable:
                # Remove all zero columns
                keep = [r for r in range(current_rank) if r not in removable]
                U_new = U[:, keep]
                V_new = V[:, keep]
                W_new = W[:, keep]
                if _validate(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = len(keep)
                    U, V, W = U_new, V_new, W_new
                    current_rank = len(keep)
                    continue
            
            # Try linear dependency: if column r = sum of alpha_s * column_s for s != r
            # in all three matrices simultaneously, we can eliminate r.
            # This is a strong condition. Let's check for pairs first.
            
            found_reduction = False
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Check if column r is a linear combination of other columns
                # in U, V, W simultaneously
                other_cols = [c for c in range(current_rank) if c != r]
                
                # For U: solve U[:, r] = U[:, other_cols] @ x
                U_other = U[:, other_cols]
                U_r = U[:, r]
                
                # Check if U_r is in the span of U_other
                if np.linalg.matrix_rank(np.column_stack([U_other, U_r])) == np.linalg.matrix_rank(U_other):
                    continue
                
                # U_r is in span of U_other. Now check V and W.
                V_other = V[:, other_cols]
                V_r = V[:, r]
                if np.linalg.matrix_rank(np.column_stack([V_other, V_r])) == np.linalg.matrix_rank(V_other):
                    # V_r also in span. Check W.
                    W_other = W[:, other_cols]
                    W_r = W[:, r]
                    if np.linalg.matrix_rank(np.column_stack([W_other, W_r])) == np.linalg.matrix_rank(W_other):
                        # All three are in the span. Find the coefficients.
                        # Solve for x in U_other @ x = U_r
                        try:
                            x, residuals, _, _ = np.linalg.lstsq(U_other, U_r, rcond=None)
                            # Verify in V and W
                            if np.allclose(V_other @ x, V_r, atol=1e-9) and np.allclose(W_other @ x, W_r, atol=1e-9):
                                # Column r can be eliminated
                                new_U = U.copy()
                                new_V = V.copy()
                                new_W = W.copy()
                                for s, col_idx in enumerate(other_cols):
                                    new_U[:, col_idx] -= x[s] * U_r
                                    new_V[:, col_idx] -= x[s] * V_r
                                    new_W[:, col_idx] -= x[s] * W_r
                                # Remove column r
                                keep = [c for c in range(current_rank) if c != r]
                                new_U = new_U[:, keep]
                                new_V = new_V[:, keep]
                                new_W = new_W[:, keep]
                                if _validate(new_U, new_V, new_W):
                                    best_U, best_V, best_W = new_U, new_V, new_W
                                    best_rank = current_rank - 1
                                    U, V, W = new_U, new_V, new_W
                                    current_rank -= 1
                                    found_reduction = True
                                    break
                        except:
                            pass
            
            if not found_reduction:
                break
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank, but only if we have time
    if time.time() < deadline:
        U, V, W, reduced_rank = _try_reduce_rank(U, V, W, 40, deadline)
        rank = reduced_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
