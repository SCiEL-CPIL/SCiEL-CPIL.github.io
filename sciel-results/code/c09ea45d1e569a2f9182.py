import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T
    # T[I, J, K] = 1 if indices are consistent, 0 otherwise
    # I = i*m + j
    # J = j*p + k
    # K = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    def schoolbook():
        R = nm * mp * pn
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    U[I, col] = 1.0
                    V[J, col] = 1.0
                    W[K, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    def is_valid_exact(U, V, W):
        # Check half-integer
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-6):
            return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-6):
            return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-6):
            return False
            
        # Check exact decomposition
        # Reconstruct tensor
        # sum_r U[I, r] V[J, r] W[K, r]
        # We can do this efficiently with einsum or loops
        # Given small size, loops are fine
        for I in range(nm):
            for J in range(mp):
                for K in range(pn):
                    s = np.dot(U[I] * V[J], W[K])
                    if abs(s - T[I, J, K]) > 1e-6:
                        return False
        return True

    start_time = time.time()
    time_limit = 110.0
    
    # Try to find a decomposition with rank < 48
    for R in range(47, 20, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Greedy reconstruction
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        E = T.copy()
        
        success = False
        for r in range(R):
            # Find best rank-1 approximation of E
            # Matricize E
            # E is (nm, mp, pn)
            # M1 = E.reshape(nm, mp*pn)
            # M2 = E.reshape(mp, nm*pn)
            # M3 = E.reshape(pn, nm*mp)
            
            M1 = E.reshape(nm, mp * pn)
            U_svd, S, Vt_svd = np.linalg.svd(M1, full_matrices=False)
            u1 = U_svd[:, 0]
            v1_flat = Vt_svd[0, :]
            v1 = v1_flat.reshape(mp, pn)
            
            # Check if the top component is significant
            if S[0] < 1e-9:
                break
                
            # The best rank-1 approximation is u1 * v1 * S[0]
            # But we need to match the structure U, V, W
            # The tensor is u1 * v1 (where v1 is 2D)
            # We need to factor v1 into v2 and w?
            # No, the rank-1 tensor is u * v * w where v is vector of length mp, w is vector of length pn?
            # No, the factors are U (nm x R), V (mp x R), W (pn x R).
            # So the rank-1 component is outer product of u (nm), v (mp), w (pn).
            # The matricization M1 = E.reshape(nm, mp*pn) has rows indexed by nm, cols by (mp, pn).
            # The top right singular vector v1_flat is of length mp*pn.
            # This corresponds to the tensor v1 (mp, pn).
            # We need to factor v1 into v (mp) and w (pn) such that v1 = outer(v, w).
            # This is only possible if v1 has rank 1.
            # But v1 is the top right singular vector of M1, which is a vector.
            # Wait, SVD of M1 gives u (nm) and v (mp*pn).
            # The rank-1 tensor is u * v (reshaped to mp, pn).
            # This tensor is of the form u_i * (v_j * w_k).
            # So we need to find v (mp) and w (pn) such that outer(v, w) approximates v1.
            # Since v1 is a fixed matrix (mp x pn), we can take its top SVD.
            Uv, Sv, Wt = np.linalg.svd(v1, full_matrices=False)
            v = Uv[:, 0]
            w = Wt[0, :]
            
            # Scale to match
            # The component is u1 * S[0] * (v * w) ?
            # The reconstruction is sum_r U_r V_r W_r.
            # We want u * v * w to approximate E.
            # The optimal scaling is determined by the inner product.
            # Let's just use the vectors and scale them to be half-integers.
            
            # Scale u, v, w to have max abs value 1 or similar, then round
            # Or just round directly?
            # The magnitude matters.
            # We want to maximize <E, u*v*w>
            # Let's normalize u, v, w to unit length.
            u_norm = np.linalg.norm(u1)
            v_norm = np.linalg.norm(v)
            w_norm = np.linalg.norm(w)
            
            if u_norm < 1e-9 or v_norm < 1e-9 or w_norm < 1e-9:
                continue
                
            u1 = u1 / u_norm
            v = v / v_norm
            w = w / w_norm
            
            # The best scale is S[0] * (1/(u_norm*v_norm*w_norm)) ?
            # Actually, the component value is S[0] * (u1 * v1)
            # v1 = v * w (approx)
            # So the scale factor is S[0].
            
            # We need to choose integer/half-integer values.
            # Let's try to scale u, v, w by some factor k such that they become half-integers.
            # Or just round u, v, w directly? They are unit vectors, so entries are small.
            # We need larger values to get 1s.
            
            # Strategy: Find a scaling factor alpha such that alpha*u, alpha*v, alpha*w are close to half-integers.
            # This is hard.
            # Alternative: Just round the un-normalized vectors?
            # The un-normalized vectors from SVD are arbitrary scale.
            
            # Let's try a few random scalings or just use the SVD vectors and scale them to have max 1.
            # Then multiply by a large integer?
            
            # Simpler: Use the fact that the tensor entries are 0 or 1.
            # The components should be small integers.
            
            # Let's just try to round the vectors u, v, w after scaling them to have max abs 1.
            # Then multiply by 1?
            # This might not work.
            
            # Let's try a different approach:
            # Since the size is small, we can try to find a rank-1 tensor with integer entries that best fits E.
            # This is a combinatorial problem.
            # We can sample some random integer vectors and check.
            
            # For now, let's just use the SVD vectors, scale them to have max 1, and round to nearest 0.5.
            # If the result is zero, skip.
            
            u_max = np.max(np.abs(u1))
            v_max = np.max(np.abs(v))
            w_max = np.max(np.abs(w))
            
            if u_max < 1e-6 or v_max < 1e-6 or w_max < 1e-6:
                continue
                
            u1 = u1 / u_max
            v = v / v_max
            w = w / w_max
            
            # Round to nearest 0.5
            u_round = np.round(u1 * 2) / 2
            v_round = np.round(v * 2) / 2
            w_round = np.round(w * 2) / 2
            
            # Check if this reduces error
            comp = np.einsum('i,j,k->ijk', u_round, v_round, w_round)
            err_before = np.sum(E**2)
            E_new = E - comp
            err_after = np.sum(E_new**2)
            
            if err_after < err_before:
                U[:, r] = u_round
                V[:, r] = v_round
                W[:, r] = w_round
                E = E_new
            else:
                # Try scaling by -1?
                comp = np.einsum('i,j,k->ijk', -u_round, -v_round, -w_round)
                E_new = E - comp
                err_after = np.sum(E_new**2)
                if err_after < err_before:
                    U[:, r] = -u_round
                    V[:, r] = -v_round
                    W[:, r] = -w_round
                    E = E_new
                else:
                    # Skip this rank
                    pass
                    
        if np.max(np.abs(E)) < 1e-6:
            if is_valid_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return schoolbook()
# EVOLVE-BLOCK-END
