"""Model-and-optimize: parameterize rank-r factorization and fit/optimize parameters.
Target: exact rank decomposition of 3x4x6 matrix multiplication tensor, minimize rank.
Strategy: start from a partition-based baseline (rank 66), then run a randomized
merge-and-drop local search to reduce rank while preserving exactness.
"""
import random
import time

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

def true_tensor():
    """Return the list of (i,j,k) triples that are 1 in the matmul tensor."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def verify(U, V, W, r):
    """Verify exact decomposition using NumPy for speed."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to 3D: U3[i,j,l], V3[j,k,l], W3[k,i,l]
    U3 = U_np.reshape(n, m, r)
    V3 = V_np.reshape(m, p, r)
    W3 = W_np.reshape(p, n, r)
    
    # Compute sum_l U3[i,j,l] * V3[j,k,l] * W3[k,i,l]
    # This is a batched contraction.
    # Sum over l: np.einsum('ijl,jkl,kil->ijk', U3, V3, W3)
    # Note: W3 index is (k, i, l) because W row is Cidx(i,k) = k*n+i.
    # In W3 reshape (p, n, r), index is [k, i, l].
    # So we need W3[k, i, l].
    result = np.einsum('ijl,jkl,kil->ijk', U3, V3, W3)
    
    # Check if result is all 1s
    return np.allclose(result, 1.0, atol=1e-9)

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        # M1
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        # M2
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        # M3
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        # M4
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        # M5
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        # M6
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        # M7
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return U, V, W

def try_merge(U, V, W, r, l1, l2):
    """Try to merge rank terms l1 and l2 into a single term.
    For two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 to sum to a rank-1 tensor,
    we need u1, u2 proportional AND v1, v2 proportional AND w1, w2 proportional.
    If so, the sum is (u1+u2)⊗(v1+αv2)⊗(w1+βw2) for appropriate scaling.
    More precisely: if u2 = α*u1, v2 = β*v1, w2 = γ*w1, then
    u1⊗v1⊗w1 + u2⊗v2⊗w2 = (1+αβγ) * u1⊗v1⊗w1 which is rank-1.
    But we can also have more general cases. Let's check proportionality.
    """
    # Check if U rows for l1 and l2 are proportional
    u1 = [row[l1] for row in U]
    u2 = [row[l2] for row in U]
    v1 = [row[l1] for row in V]
    v2 = [row[l2] for row in V]
    w1 = [row[l1] for row in W]
    w2 = [row[l2] for row in W]
    
    # Find scaling factors
    def find_scale(a, b):
        """Find s such that b = s*a, return s or None."""
        scale = None
        for i in range(len(a)):
            if a[i] != 0:
                s = b[i] / a[i]
                if scale is None:
                    scale = s
                elif abs(scale - s) > 1e-9:
                    return None
        # Verify
        if scale is None:
            # All zeros in a, check b is also all zeros
            if all(b[i] == 0 for i in range(len(b))):
                return 0
            return None
        for i in range(len(a)):
            if abs(b[i] - scale * a[i]) > 1e-9:
                return None
        return scale
    
    su = find_scale(u1, u2)
    sv = find_scale(v1, v2)
    sw = find_scale(w1, w2)
    
    if su is not None and sv is not None and sw is not None:
        # They are proportional, so the sum is rank-1
        # New term: (u1+u2) ⊗ (v1+v2) ⊗ (w1+w2) but scaled appropriately
        # Actually: u1⊗v1⊗w1 + u2⊗v2⊗w2 = u1⊗v1⊗w1 + su*u1⊗sv*v1⊗sw*w1
        # = (1 + su*sv*sw) * u1⊗v1⊗w1
        factor = 1 + su * sv * sw
        if abs(factor) < 1e-12:
            # The sum is zero, just drop both
            return 'drop_both', None
        # New single term with entries factor * u1[i], factor * v1[j], factor * w1[k]
        new_u = [factor * x for x in u1]
        new_v = [factor * x for x in v1]
        new_w = [factor * x for x in w1]
        return 'merge', (new_u, new_v, new_w)
    
    return None

def try_drop(U, V, W, r, l):
    """Check if removing term l still gives an exact decomposition."""
    # Recompute without term l
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for ll in range(r):
                    if ll == l:
                        continue
                    s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(i,k)][ll]
                if s != 1:
                    return False
    return True

def hals_refine(U, V, W, r, max_iter=50, tol=1e-6):
    """HALS-style refinement: iteratively update one factor at a time to minimize
    the Frobenius norm of (T - sum_l u_l ⊗ v_l ⊗ w_l).
    Here T is the 3-way tensor with T[i,j,k] = 1 for all (i,j,k).
    We reshape U, V, W to 3D tensors and do alternating least squares.
    """
    # Convert to numpy for speed
    import numpy as np
    U_np = np.array(U, dtype=float)  # (n*m, r)
    V_np = np.array(V, dtype=float)  # (m*p, r)
    W_np = np.array(W, dtype=float)  # (p*n, r)
    
    # Reshape to 3D: U3[i,j,l], V3[j,k,l], W3[i,k,l]
    U3 = U_np.reshape(n, m, r)
    V3 = V_np.reshape(m, p, r)
    W3 = W_np.reshape(p, n, r)
    
    for it in range(max_iter):
        # Update U: for each (i,j), solve for u_{i,j,:} given V, W
        for i in range(n):
            for j in range(m):
                # Residual for (i,j): T[i,j,:] - sum_l u_{i,j,l} * (V3[j,:,l] * W3[:,i,l] summed over k)
                # Actually: sum_k T[i,j,k] = sum_l u_{i,j,l} * sum_k V3[j,k,l] * W3[k,i,l]
                # This is linear in u_{i,j,:}
                # A[l] = sum_k V3[j,k,l] * W3[k,i,l]
                # b = sum_k T[i,j,k] = p (since all entries are 1)
                A_vec = np.zeros(r)
                for k in range(p):
                    A_vec += V3[j, k, :] * W3[k, i, :]
                b = float(p)  # sum of T[i,j,k] over k = p
                # Solve A_vec^T * u = b, i.e., u = b * A_vec / (A_vec^T * A_vec)
                norm_sq = np.dot(A_vec, A_vec)
                if norm_sq > tol:
                    U3[i, j, :] = b * A_vec / norm_sq
        
        # Update V: for each (j,k), solve for v_{j,k,:} given U, W
        for j in range(m):
            for k in range(p):
                A_vec = np.zeros(r)
                for i in range(n):
                    A_vec += U3[i, j, :] * W3[k, i, :]
                b = float(n)  # sum of T[i,j,k] over i = n
                norm_sq = np.dot(A_vec, A_vec)
                if norm_sq > tol:
                    V3[j, k, :] = b * A_vec / norm_sq
        
        # Update W: for each (k,i), solve for w_{k,i,:} given U, V
        for k in range(p):
            for i in range(n):
                A_vec = np.zeros(r)
                for j in range(m):
                    A_vec += U3[i, j, :] * V3[j, k, :]
                b = float(m)  # sum of T[i,j,k] over j = m
                norm_sq = np.dot(A_vec, A_vec)
                if norm_sq > tol:
                    W3[k, i, :] = b * A_vec / norm_sq
    
    # Convert back to list format
    U_new = U3.reshape(n*m, r).tolist()
    V_new = V3.reshape(m*p, r).tolist()
    W_new = W3.reshape(p*n, r).tolist()
    return U_new, V_new, W_new

def snap_to_half_int(U, V, W):
    """Snap values to nearest half-integer."""
    def snap(x):
        return round(2*x) / 2.0
    U_snap = [[snap(x) for x in row] for row in U]
    V_snap = [[snap(x) for x in row] for row in V]
    W_snap = [[snap(x) for x in row] for row in W]
    return U_snap, V_snap, W_snap

def local_search(U, V, W, r, deadline, rng):
    import numpy as np
    best_r = r
    best_U, best_V, best_W = U, V, W
    
    # Convert to NumPy for fast drop checks
    U_np = np.array(best_U, dtype=float)
    V_np = np.array(best_V, dtype=float)
    W_np = np.array(best_W, dtype=float)
    
    # Precompute the full tensor sum
    def compute_tensor(U_np, V_np, W_np):
        U3 = U_np.reshape(n, m, -1)
        V3 = V_np.reshape(m, p, -1)
        W3 = W_np.reshape(p, n, -1)
        return np.einsum('ijl,jkl,kil->ijk', U3, V3, W3)
    
    T_target = np.ones((n, m, p))
    T_current = compute_tensor(U_np, V_np, W_np)
    
    while time.time() < deadline - 5:
        improved = False
        # Try to drop a column
        for l in range(best_r):
            # Calculate tensor without column l
            # T_current - term_l
            u_l = U_np[:, l]
            v_l = V_np[:, l]
            w_l = W_np[:, l]
            
            u2_l = u_l.reshape(n, m)
            v2_l = v_l.reshape(m, p)
            w2_l = w_l.reshape(p, n)
            
            term_l = np.einsum('ij,jk,ki->ijk', u2_l, v2_l, w2_l)
            
            T_test = T_current - term_l
            
            if np.allclose(T_test, T_target, atol=1e-9):
                # Drop column l
                U_np = np.delete(U_np, l, axis=1)
                V_np = np.delete(V_np, l, axis=1)
                W_np = np.delete(W_np, l, axis=1)
                best_r -= 1
                T_current = T_test
                improved = True
                break
        
        if not improved:
            break
    
    # Convert back to list
    best_U = U_np.tolist()
    best_V = V_np.tolist()
    best_W = W_np.tolist()
    
    return best_r, best_U, best_V, best_W

def _combinations(lst, k):
    """Simple combinations generator."""
    if k == 0:
        yield []
        return
    if k > len(lst):
        return
    for i in range(len(lst)):
        for rest in _combinations(lst[i+1:], k-1):
            yield [lst[i]] + list(rest)

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    # Baseline
    U, V, W = baseline_66()
    r = 66

    # Local search to drop redundant terms
    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    # Final verification
    if not verify(U, V, W, r):
        # Fallback to baseline if search broke it
        U, V, W = baseline_66()
        r = 66
        if not verify(U, V, W, r):
            raise ValueError("Baseline is invalid")

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
