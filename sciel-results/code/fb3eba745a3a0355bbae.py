"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Attempt to find a low-rank decomposition of the 2x4x7 matrix multiplication tensor
    # using randomized linear algebra (random projections and SVD) within the time budget.
    # If no better rank is found, fall back to the schoolbook decomposition.
    
    import time
    start_time = time.time()
    deadline = 115.0  # Leave a safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    def make_schoolbook():
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        # Convert to lists of integers/floats
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # The tensor entry is 1 iff j == j' and k == k' and i == i'
    # Wait, the multiplication tensor T_{i,j,k} for M_{n,m,p} is defined such that:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is a 3-tensor T in U*V*W* such that <T, u⊗v⊗w> = ...
    # Let's use the standard flattening.
    # Let T be of shape (n*m, m*p, p*n).
    # T[(i,m+j), (j',m+k), (k',n+i')] = 1 if j==j' and k==k' and i==i' ?
    # No. The matrix multiplication tensor M_{n,m,p} is an element of K^{n*m} ⊗ K^{m*p} ⊗ K^{p*n}
    # such that for any A (n x m), B (m x p), 
    # vec(A*B) = M_{n,m,p} (vec(A) ⊗ vec(B))  ??? No, that's bilinear.
    # The tensor T is defined by T_{i,j;k,l} ... no.
    # Let's use the standard definition:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (where indices are (i,j,k))
    # This T has shape (n, m, p).
    # To flatten to matrices U (n*m x rank), V (m*p x rank), W (p*n x rank):
    # We need T to be a 3-tensor of shape (n*m, m*p, p*n).
    # Let's construct T_flat of shape (n*m, m*p, p*n).
    # T_flat[a, b, c] corresponds to i = a // m, j = a % m; j' = b // p, k = b % p; k' = c // n, i' = c % n.
    # The multiplication tensor has a non-zero entry at (i, j, k) if we consider the bilinear map.
    # Actually, the tensor is T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (with appropriate flattening)
    # Let's just use the property: rank(T) is the minimum number of rank-1 tensors to sum to T.
    # A rank-1 tensor is u ⊗ v ⊗ w.
    # We want to approximate T with sum_{r=1}^R u_r ⊗ v_r ⊗ w_r.
    
    # Let's create the 3D tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The term in the tensor corresponding to indices (i,j,k) is 1
                # In the flattened form:
                # First index: i*m + j
                # Second index: j*p + k
                # Third index: k*n + i
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # We try to find a low rank decomposition.
    # Since T is sparse and structured, we can try random sampling or SVD on modes.
    # However, finding the exact tensor rank is NP-hard.
    # We can try a heuristic: 
    # 1. Try ranks from schoolbook down to some lower bound.
    # 2. For each rank, try to solve for U, V, W using a randomized approach or alternating least squares.
    
    # Given the small dimensions, we might try a direct search or a specific known decomposition.
    # For 2x4x7, is there a known better than 56?
    # 2*4*7 = 56.
    # Strassen for 2x2x2 is 7 (vs 8).
    # If we could block 2x4x7 into 2x2x2 blocks...
    # 2 = 1*2, 4 = 2*2, 7 is prime.
    # Maybe pad to 2x4x8? 2*4*8 = 64.
    # Strassen on the 4x8 part?
    # 4x8 can be seen as 2x2 blocks of 2x4.
    # This is getting complex.
    
    # Let's try a randomized SVD approach to estimate rank.
    # If we can find a rank R decomposition, we verify it.
    # Verification: Check if sum_{r} U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals T.
    
    def verify_decomp(U, V, W, T, tol=1e-9):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reconstruct T_approx
        # This is slow if R is large.
        # T_approx[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # We can do this efficiently using einsum.
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    # Let's try to reduce rank by looking for linear dependencies in the "modes".
    # But for exact decomposition, we need integer/half-integer entries ideally.
    # The schoolbook has all 1s.
    
    # If we can't find a better one quickly, return schoolbook.
    # The prompt asks for a "stronger algorithm" and "structural change".
    # The WFT (Winograd) hint suggests using Fourier transforms.
    # But implementing a full WFT for matrix multiplication tensor rank is non-trivial.
    
    # Let's try a simple rank reduction heuristic:
    # Check if any column in the mode-1 unfolding is linearly dependent on others.
    # Unfolding T to matrix T_(1) of shape (nm, mp*pn).
    # Rank of T_(1) is an upper bound on tensor rank? No, tensor rank >= max(mode ranks).
    # The mode ranks are:
    # T_(1) rank <= min(nm, mp*pn)
    # T_(2) rank <= min(mp, nm*pn)
    # T_(3) rank <= min(pn, nm*mp)
    
    # For 2,4,7:
    # nm=8, mp=28, pn=14.
    # T_(1) shape (8, 392). Rank <= 8.
    # T_(2) shape (28, 112). Rank <= 28.
    # T_(3) shape (14, 224). Rank <= 14.
    # So tensor rank >= 28.
    # Schoolbook is 56.
    # There is a gap [28, 56].
    
    # We can try to find a decomposition of rank 28, 29, ..., up to 55.
    # This is a hard search problem.
    # We'll try a randomized gradient descent or alternating least squares (ALS) for a few iterations.
    # ALS for tensor CP decomposition.
    
    def als_decompose(T, R, max_iter=100, tol=1e-6):
        # Initialize U, V, W randomly
        np.random.seed(42)
        nm, mp, pn = T.shape
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        for it in range(max_iter):
            # Update U
            # For each i, U[i,:] is solution to min ||T[i,:,:] - sum_r U[i,r] V[:,r]⊗W[:,r]||
            # This is a linear system.
            # Flatten V⊗W for each r?
            # Let's use the standard ALS update.
            # T_(1) is (nm, mp*pn).
            # V⊗W matrix of size (mp*pn, R).
            # Let's construct V_w = np.einsum('ir,jr->ij', V, W).reshape(mp*pn, R)?
            # No, V is (mp, R), W is (pn, R).
            # The outer product of V[:,r] and W[:,r] is a matrix of size (mp, pn).
            # We need to multiply U (nm, R) by (V⊗W) to get T.
            # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
            
            # Update U:
            # For each i, minimize ||T[i,:,:] - sum_r U[i,r] (V[:,r] ⊗ W[:,r])||_F
            # This is: T_(1)[i, :] = U[i, :] * (V⊗W)  (where V⊗W is (mp*pn, R))
            # Let's build M_VW = np.einsum('jr,kr->jk r', V, W).reshape(mp*pn, R)
            M_VW = np.einsum('jr,kr->jkr', V, W).reshape(mp*pn, R)
            # U = T_(1) @ M_VW^+ (pseudo-inverse)
            # T_(1) is (nm, mp*pn)
            T1 = T.reshape(nm, mp*pn)
            U_new, _, _, _ = np.linalg.lstsq(M_VW, T1.T, rcond=None)
            U = U_new.T
            
            # Update V:
            # For each j, minimize ||T[:,j,:] - sum_r V[j,r] (U[:,r] ⊗ W[:,r])||_F
            M_UW = np.einsum('ir,kr->ikr', U, W).reshape(nm*pn, R)
            T2 = T.transpose(1, 0, 2).reshape(mp, nm*pn)
            V_new, _, _, _ = np.linalg.lstsq(M_UW, T2.T, rcond=None)
            V = V_new.T
            
            # Update W:
            # For each k, minimize ||T[:,:,k] - sum_r W[k,r] (U[:,r] ⊗ V[:,r])||_F
            M_UV = np.einsum('ir,jr->ijr', U, V).reshape(nm*mp, R)
            T3 = T.transpose(2, 0, 1).reshape(pn, nm*mp)
            W_new, _, _, _ = np.linalg.lstsq(M_UV, T3.T, rcond=None)
            W = W_new.T
            
            # Check convergence
            # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            # diff = np.linalg.norm(T_approx - T) / np.linalg.norm(T)
            # if diff < tol: break
            
            if time.time() > deadline - 10:
                break
        
        return U, V, W
    
    best_result = make_schoolbook()
    best_rank = rank_school
    
    # Try ranks from 30 to 55 (arbitrary range, might be too slow)
    # Actually, let's just try a few random restarts for rank 28, 32, 40, 50.
    # And check if the result is valid.
    
    # Since exact verification with allclose might pass for non-exact (floating point) decompositions,
    # but the task requires "integer/half-integer entries".
    # The schoolbook has integers.
    # ALS will produce floats.
    # We can try to round the floats to nearest 0.5 and verify.
    
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    def verify_exact(U, V, W, T):
        # Check if U, V, W have integer/half-integer entries
        if not np.allclose(U, round_to_half(U)):
            return False
        if not np.allclose(V, round_to_half(V)):
            return False
        if not np.allclose(W, round_to_half(W)):
            return False
        
        # Check if they reconstruct T exactly
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=1e-9)
    
    # Try a few ranks
    for R in [28, 32, 40, 48, 56]:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
            
        # Run ALS
        U, V, W = als_decompose(T, R, max_iter=20)
        
        # Try rounding
        U_r = round_to_half(U)
        V_r = round_to_half(V)
        W_r = round_to_half(W)
        
        if verify_exact(U_r, V_r, W_r, T):
            best_rank = R
            best_result = {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            break # Found a valid one, lower ranks might be harder, but let's try to go lower?
            # Actually, if we found 48, we should try 40, 32, 28 again?
            # The loop is descending in effort? No, ascending in R.
            # We want minimal R. So we should try low R first.
            # But low R is harder to find.
            # Let's just try the ones we listed.
    
    return best_result
# EVOLVE-BLOCK-END
