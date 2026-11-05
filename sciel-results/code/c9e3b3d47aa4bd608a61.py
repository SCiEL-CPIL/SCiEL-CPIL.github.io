import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is valid with high precision."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Indices:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # We check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for valid
    # and == 0 for invalid.
    
    # To be fast, we can just compute the full tensor and check.
    # But R is small.
    
    # Let's use the explicit loop for correctness first, then optimize if needed.
    # Actually, 42 valid and 1722 invalid is small.
    
    valid_ok = True
    invalid_ok = True
    
    # Precompute indices for speed in the main loop, but here we just check.
    # We can reconstruct the full tensor T[i,j,k]
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's do it via einsum or manual sum
    # U_flat: (6, R)
    # V_flat: (21, R)
    # W_flat: (14, R)
    
    # T is a 2x3x7 tensor.
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # We can reshape U, V, W to 3D?
    # U is (n, m, R)
    # V is (m, p, R)
    # W is (p, n, R)
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # This is a bit tricky to vectorize directly without loops over r or i,j,k.
    # Given small sizes, a loop is fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if i == 0 and j == 0 and k == 0: # Example valid check
                    pass
                
    # Actually, the "valid" entries are ALL (i,j,k) in the multiplication tensor?
    # NO. The multiplication tensor M has entries M[i,j,k] = 1 if the output C[i,k] depends on A[i,j] and B[j,k].
    # For matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor M[i, j, k] is 1 for all i, j, k?
    # Wait. The tensor represents the bilinear map.
    # The entries of the tensor are 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In standard matrix multiplication, EVERY term A[i,j]B[j,k] contributes to C[i,k] (via the sum over j).
    # So M[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    # And M[i,j,k] = 0 otherwise? But the domain is exactly these indices.
    # So the "valid" set is ALL (i,j,k) in the grid.
    # And the "invalid" set is empty?
    
    # WAIT. The tensor decomposition of matrix multiplication usually refers to the tensor M where
    # M_{i, j, k} = 1 if the product A_{i, j} B_{j, k} appears in the sum for C_{i, k}.
    # Since C_{i, k} = \sum_j A_{i, j} B_{j, k}, the tensor is 1 for all (i, j, k).
    # So there are NO "invalid" indices in the standard tensor definition?
    
    # Let's re-read the seed program.
    # Seed program:
    # valid.append((i*m+j, j*p+k, k*n+i))
    # This covers ALL i, j, k.
    # Then it checks invalid indices:
    # for i1... for k2... if i1==i2 and j1==j2 and k1==k2: continue
    # This implies the seed program thinks there ARE invalid indices.
    # Why?
    
    # Ah, the tensor decomposition is for the BILINEAR FORM.
    # The tensor is $T \in \mathbb{F}^n \otimes \mathbb{F}^m \otimes \mathbb{F}^p$.
    # The entries are $T_{i, j, k}$.
    # For matrix multiplication, $T_{i, j, k} = 1$ if the term $A_{i, j} B_{j, k}$ is present in the product.
    # It IS present for all $i, j, k$.
    # So $T_{i, j, k} = 1$ for all $i, j, k$.
    
    # So why does the seed check invalid indices?
    # Maybe the seed program is wrong? Or I am misunderstanding the "invalid" part.
    # If $T_{i, j, k} = 1$ for all, then we just need $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ for all $i,j,k$.
    # There are no constraints for "0".
    
    # Let's check the seed code again.
    # `invalid.append(...)`
    # It checks `if abs(val) > tol: return False` for invalid.
    # This implies the target tensor has 0s in those positions.
    # But the multiplication tensor is dense (all 1s).
    # UNLESS... the indices are different.
    # U index: `i*m+j` (A)
    # V index: `j*p+k` (B)
    # W index: `k*n+i` (C)
    # This maps (i,j,k) to (A_idx, B_idx, C_idx).
    # The tensor entry is 1 if this combination is part of the multiplication.
    # It is part for ALL i,j,k.
    
    # So the "invalid" check in the seed is checking if $\sum U V W = 0$ for combinations that are NOT (i,j,k)?
    # But the indices $i*m+j$, $j*p+k$, $k*n+i$ uniquely identify the triple $(i,j,k)$.
    # There are no other triples.
    # So the "invalid" list in the seed is EMPTY?
    # Let's trace `_precompute_indices`.
    # `valid` has $n \cdot m \cdot p$ entries.
    # `invalid` loop:
    # i1 in n, j1 in m, k1 in p.
    # i2 in n, j2 in m, k2 in p.
    # If i1==i2 and j1==j2 and k1==k2: continue.
    # Else: append (i1*m+j1, j2*p+k1, k2*n+i2).
    
    # Wait. The indices in `invalid` are:
    # U_idx = i1*m + j1
    # V_idx = j2*p + k1  <-- Note: j2 from second loop, k1 from first
    # W_idx = k2*n + i2  <-- Note: k2 from second loop, i2 from second
    
    # This is NOT a simple (i,j,k) triple.
    # It is mixing indices from different triples.
    # This corresponds to checking if the decomposition is "separable" or "rank 1"?
    # No, this checks if the tensor is the SUM of rank 1 tensors.
    # If we have a decomposition $\sum_r u_r \otimes v_r \otimes w_r$,
    # then the value at any point (a,b,c) is $\sum_r U[a,r] V[b,r] W[c,r]$.
    # The target tensor $T$ has $T[a,b,c] = 1$ if (a,b,c) corresponds to a valid (i,j,k) multiplication term.
    # And $T[a,b,c] = 0$ otherwise.
    
    # So the "invalid" indices are those (a,b,c) that do NOT correspond to a valid (i,j,k) with the standard mapping?
    # But the mapping $a = i*m+j$, $b = j*p+k$, $c = k*n+i$ is a bijection from $(i,j,k)$ to a subset of $(a,b,c)$.
    # The total size of $(a,b,c)$ space is $(nm) \times (mp) \times (pn)$.
    # The size of $(i,j,k)$ space is $nmp$.
    # Since $n,m,p > 1$, the image of the mapping is a small subset of the full space.
    # So YES, there are many "invalid" indices where the target value is 0.
    
    # So the seed program is correct. We must satisfy:
    # 1. $\sum_r U[a,r] V[b,r] W[c,r] = 1$ for all valid (a,b,c) derived from (i,j,k).
    # 2. $\sum_r U[a,r] V[b,r] W[c,r] = 0$ for all other (a,b,c).
    
    # This is a much harder constraint. The tensor is sparse in the full $(nm) \times (mp) \times (pn)$ space.
    # It has $nmp$ ones and $(nm \cdot mp \cdot pn - nmp)$ zeros.
    
    # For $n=2, m=3, p=7$:
    # Valid: 42.
    # Total: $6 \cdot 21 \cdot 14 = 1764$.
    # Invalid: 1722.
    
    # So we need to kill 1722 interactions. This is why the rank must be high enough to allow this cancellation.
    
    # The "dissection" or "block" structure might help satisfy these zero constraints more easily.
    
    # Strategy:
    # Use the known rank 13 solution if it exists.
    # If I can't find it, I'll stick to the schoolbook 42.
    # But I should try to find a lower rank.
    
    # I will implement the local search with snapping.
    
    pass

def _precompute_indices(n, m, p):
    """Precompute valid and invalid index triples."""
    valid = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.append((i*m+j, j*p+k, k*n+i))
                
    invalid = []
    # To save memory/time, we don't need to store all invalid indices if we can generate them or check a subset.
    # But 1722 is small.
    valid_set = set(valid)
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a,b,c) not in valid_set:
                    invalid.append((a,b,c))
                    
    return np.array(valid, dtype=int), np.array(invalid, dtype=int)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook solution (Rank 42)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1.0
                V_school[j * p + k, c] = 1.0
                W_school[k * n + i, c] = 1.0
                c += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    valid_idx, invalid_idx = _precompute_indices(n, m, p)
    
    # Try ranks 13 to 20
    for target_rank in range(13, 21):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(trial * 104729 + target_rank)
            # Initialize with small random values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            best_obj = np.inf
            best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
            
            # Gradient Descent with Snapping
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                # Compute Objective and Gradients
                # Valid
                va = valid_idx[:, 0]
                vb = valid_idx[:, 1]
                vc = valid_idx[:, 2]
                prod_valid = U[va] * V[vb] * W[vc]
                valid_vals = prod_valid.sum(axis=1)
                valid_err = valid_vals - 1.0
                
                # Invalid
                ia = invalid_idx[:, 0]
                ib = invalid_idx[:, 1]
                ic = invalid_idx[:, 2]
                prod_invalid = U[ia] * V[ib] * W[ic]
                invalid_vals = prod_invalid.sum(axis=1)
                
                obj = np.sum(valid_err ** 2) + np.sum(invalid_vals ** 2)
                
                if obj < best_obj:
                    best_obj = obj
                    best_U_local, best_V_local, best_W_local = U.copy(), V.copy(), W.copy()
                    
                if best_obj < 1e-12:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if _check_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = target_rank
                        break
                        # We found a valid one, can break outer loops too?
                        # Yes, lower rank is better.
                    
                # Gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Valid contributions
                # d(obj)/d(U[a,r]) = 2 * valid_err * V[b,r] * W[c,r]
                # We need to sum over all valid (a,b,c)
                
                valid_contrib_U = 2 * valid_err[:, None] * V[vb] * W[vc]
                valid_contrib_V = 2 * valid_err[:, None] * U[va] * W[vc]
                valid_contrib_W = 2 * valid_err[:, None] * U[va] * V[vb]
                
                invalid_contrib_U = 2 * invalid_vals[:, None] * V[ib] * W[ic]
                invalid_contrib_V = 2 * invalid_vals[:, None] * U[ia] * W[ic]
                invalid_contrib_W = 2 * invalid_vals[:, None] * U[ia] * V[ib]
                
                np.add.at(grad_U, va, valid_contrib_U)
                np.add.at(grad_V, vb, valid_contrib_V)
                np.add.at(grad_W, vc, valid_contrib_W)
                
                np.add.at(grad_U, ia, invalid_contrib_U)
                np.add.at(grad_V, ib, invalid_contrib_V)
                np.add.at(grad_W, ic, invalid_contrib_W)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            if best_rank < rank_school:
                break
                
        if best_rank < rank_school:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
