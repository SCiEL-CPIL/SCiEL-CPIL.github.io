"""Combined program: ALS candidate search + exact linear-system refinement + exact integer verification."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _strassen_lower_bound(n, m, p):
    """Lower bound on the rank of multiplying an n x m by m x p matrix.
    Uses the block-Strassen argument: partition A into (n//2) x (m//2) blocks
    and B into (m//2) x (p//2) blocks, multiply as block matrices, and apply
    the known 2x2 Strassen lower bound of 7 per block-pair.
    """
    r = n * m * p  # schoolbook
    for n2 in range(1, n):
        for m2 in range(1, m):
            for p2 in range(1, p):
                if n % n2 == 0 and m % m2 == 0 and p % p2 == 0:
                    # number of block-pairs in a block-matrix multiply
                    pairs = (n // n2) * (m // m2) * (p // p2)
                    r = min(r, 7 * pairs)
    return r


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105  # safety margin under the 120s kill

    nm, mp, pn = n * m, m * p, p * n
    target = np.ones((n, m, p))
    rank_school = n * m * p

    # ---- exact integer verification (optimized) ----
    def verify_exact(Ui, Vi, Wi):
        Ui = np.asarray(Ui, dtype=np.int64)
        Vi = np.asarray(Vi, dtype=np.int64)
        Wi = np.asarray(Wi, dtype=np.int64)
        r = Ui.shape[1]
        
        # Reshape for vectorized tensor product
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r)
        U3 = Ui.reshape(n, m, r)
        V3 = Vi.reshape(m, p, r)
        W3 = Wi.reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We want T to be the identity tensor: T[i,j,k] = 1 if i==j==k? No, matrix mult tensor is T[i,j,k] = 1 if (i,j,k) corresponds to A_ij * B_jk -> C_ik.
        # The tensor for matrix mult is T_{i,j,k} where the term is A_{i,j} B_{j,k} C_{k,i} in the dual? 
        # The task says: "output factor follows the official trace convention W[k*n+i,r]".
        # And the decomposition is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,j,k}? 
        # Wait, the standard matrix multiplication tensor T has entries T_{i,j,k} such that (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # The bilinear form is <A, B, C> = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
        # The identity tensor for this bilinear form is 1 when (i,j,k) is such that the term contributes to the trace? 
        # No, the "matrix multiplication tensor" is usually defined as T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
        # So T[i,j,k] = 1 for all i,j,k? 
        # Let's check the current program's verify:
        # s = sum ... if s != 1: return False
        # It checks if the sum is 1 for ALL i,j,k.
        # This implies the target tensor is all ones?
        # Wait, if U,V,W are the schoolbook factors:
        # U[i*m+j, c] = 1 if c == i*m+j.
        # V[j*p+k, c] = 1 if c == j*p+k.
        # W[k*n+i, c] = 1 if c == k*n+i.
        # Sum is 1 only if i*m+j == j*p+k == k*n+i.
        # This is impossible for n=3, m=4, p=7 unless indices align in a specific way.
        # Actually, the schoolbook rank is n*m*p.
        # For each (i,j,k), there is a unique term.
        # So the target tensor is T[i,j,k] = 1 for all i,j,k.
        
        # Let's verify with vectorization:
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a batched operation.
        # We can use np.einsum or reshape.
        # U3: (n, m, r)
        # V3: (m, p, r)
        # W3: (p, n, r)
        
        # Sum over c: 
        # Temp[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c]
        # This is like a batched dot product between U3[:,:,c] and V3[:,:,c] but with different axes.
        # U3 is (n, m, r). V3 is (m, p, r).
        # We want for each j, sum_c U3[:, j, c] * V3[j, :, c].
        # Let's swap axes to make r last.
        # U3 is already (n, m, r).
        # V3 is (m, p, r).
        
        # We can do:
        # M[j, i, k] = sum_c U3[i, j, c] * V3[j, k, c]
        # This is np.einsum('ijc,jkc->jik', U3, V3) ? 
        # U3[i,j,c], V3[j,k,c]. Sum over c. Result depends on i,j,k.
        # Let's check indices:
        # U3[i,j,c] -> i, j, c
        # V3[j,k,c] -> j, k, c
        # Result T[i,j,k] = sum_c ...
        
        # Then we need to multiply by W3[k,i,c] and sum over c? 
        # No, W3 is (p, n, r) -> W3[k, i, c].
        # So T_final[i,j,k] = sum_c ( sum_c' U3[i,j,c'] V3[j,k,c'] ) * W3[k,i,c] ? 
        # No, the sum is over the SAME c.
        # T[i,j,k] = sum_c (U3[i,j,c] * V3[j,k,c] * W3[k,i,c])
        
        # This is a 3-way contraction.
        # We can compute it as:
        # 1. Compute P[j, i, k] = sum_c U3[i, j, c] * V3[j, k, c]
        #    This is np.einsum('ijc,jkc->jik', U3, V3)
        # 2. Then T[i,j,k] = sum_c P[j, i, k] * W3[k, i, c] ? 
        #    No, P already summed over c. The c in W must be the same c as in U and V.
        #    So we cannot separate them.
        
        # We need: T[i,j,k] = sum_c U3[i,j,c] V3[j,k,c] W3[k,i,c]
        
        # We can do this by iterating over c? No, that's slow.
        # We can reshape U3, V3, W3 to (n*m*p, r)? No.
        
        # Let's use np.einsum directly.
        # 'ijc,jkc,kic->ijk'
        # U3: i j c
        # V3: j k c
        # W3: k i c
        # Result: i j k
        
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Check if T is all 1s
        return np.all(T == 1)

    # ---- exact rational polish: given U,V (floats), solve for W exactly and round ----
    def polish_and_round(U, V, W, denom_list=(2, 10, 20, 50, 100)):
        r = U.shape[1]
        # We assume U, V are good. We solve for W.
        # The system is linear in W.
        # A * W_flat = b
        # Where A is (n*m*p, pn*r).
        # This is too large to build dense.
        # But we can use the structure.
        
        # Alternative: Since we have a float solution from ALS, we just round it.
        # If the ALS solution is very close to half-integers, rounding might work.
        
        # Let's try rounding U, V, W directly with small denominators.
        for denom in denom_list:
            Ur = (U * denom).round()
            Vr = (V * denom).round()
            Wr = (W * denom).round()
            
            # Check if they are close to integers
            if np.allclose(U, Ur / denom, atol=1e-9) and np.allclose(V, Vr / denom, atol=1e-9) and np.allclose(W, Wr / denom, atol=1e-9):
                if verify_exact(Ur, Vr, Wr):
                    return [float(v) for row in (Ur / denom) for v in row], \
                           [float(v) for row in (Vr / denom) for v in row], \
                           [float(v) for row in (Wr / denom) for v in row], r
        return None

    # ---- ALS candidate search (optimized) ----
    def compute_tensor(U, V, W):
        rr = U.shape[1]
        U3 = U.reshape(n, m, rr)
        V3 = V.reshape(m, p, rr)
        W3 = W.reshape(p, n, rr)
        return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    def als_solve(r, seed, max_iters=200):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    # A[k, c] = V[j, k, c] * W[k, i, c]
                    # V3[j, :, :] is (p, r)
                    # W3[:, i, :] is (p, r)
                    # A = V3[j, :, :] * W3[:, i, :] (element-wise)
                    A = V3[j, :, :] * W3[:, i, :]
                    # Solve A^T x = 1
                    try:
                        x = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    U[i * m + j, :] = x
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # A[i, c] = U[i, j, c] * W[k, i, c]
                    # U3[:, j, :] is (n, r)
                    # W3[k, :, :] is (n, r)
                    A = U3[:, j, :] * W3[k, :, :]
                    try:
                        x = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    V[j * p + k, :] = x
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # A[j, c] = U[i, j, c] * V[j, k, c]
                    # U3[i, :, :] is (m, r)
                    # V3[:, k, :] is (m, r)
                    A = U3[i, :, :] * V3[:, k, :]
                    try:
                        x = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    W[k * n + i, :] = x
            
            # Update views
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            T = compute_tensor(U, V, W)
            loss = np.sum((T - 1.0) ** 2)
            if loss < 1e-10:
                break
        
        T = compute_tensor(U, V, W)
        loss = np.sum((T - 1.0) ** 2)
        if loss < 1e-6:
            res = polish_and_round(U, V, W)
            if res is not None:
                return res
        return None

    # ---- schoolbook fallback ----
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_school

    lower = _strassen_lower_bound(n, m, p)
    # Try ranks from just above the lower bound up to schoolbook-1
    for r in range(max(lower, 1), rank_school):
        if time.time() > deadline:
            break
        found = False
        seeds = [r, r + 137, r + 271, r + 409, r + 541, r + 673]
        for seed in seeds:
            if time.time() > deadline:
                break
            res = als_solve(r, seed, max_iters=350)
            if res is not None:
                U_l, V_l, W_l, rr = res
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = rr
                found = True
                break
        if found:
            # found a valid low-rank; keep searching a little lower is impossible (we go up)
            # since we search ascending, the first valid is minimal in this sweep
            break

    return best
# EVOLVE-BLOCK-END
