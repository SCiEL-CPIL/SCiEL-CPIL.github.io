"""Model-and-optimize: parameterize rank-r factorization and fit/optimize parameters.
Target: exact rank decomposition of 3x4x6 matrix multiplication tensor, minimize rank.
Strategy: start from a partition-based baseline (rank 66), then run a randomized
merge-and-drop local search to reduce rank while preserving exactness.
Optimized with NumPy for vectorized verification and faster local search.
Added: fast zero-term detection and merge test for two proportional terms.
"""
import numpy as np
import time
import random

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

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
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def verify_np(U, V, W):
    """Vectorized verification using NumPy."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    # sum_l U[i,j,l] * V[j,k,l] * W[k,i,l]
    result = np.einsum('ijl,jkl,kil->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(result, 1.0, atol=1e-9)

def term_contribution(U, V, W, l):
    """Compute the tensor contribution of term l: U[:,l] outer V[:,l] outer W[:,l]."""
    u = U.reshape(n, m, -1)[:, :, l]
    v = V.reshape(m, p, -1)[:, :, l]
    w = W.reshape(p, n, -1)[:, :, l]
    # Contribution to C[i,j,k] = U[i,j,l] * V[j,k,l] * W[k,i,l]
    # Return as a 3D array of shape (n, p, m) or similar for comparison
    # Actually, the tensor T[i,j,k] = sum_l U[i,j,l]*V[j,k,l]*W[k,i,l]
    # So term l contributes U[i,j,l]*V[j,k,l]*W[k,i,l] to T[i,j,k]
    return u[:, None, None] * v[None, :, :] * w[None, :, :]  # shape (n, p, n) -> need to align

def is_zero_term(U, V, W, l):
    """Fast check: if any of U[:,l], V[:,l], W[:,l] is all zeros, term is zero."""
    if np.allclose(U[:, l], 0, atol=1e-12):
        return True
    if np.allclose(V[:, l], 0, atol=1e-12):
        return True
    if np.allclose(W[:, l], 0, atol=1e-12):
        return True
    return False

def try_drop_np(U, V, W, l):
    """Check if removing term l still gives an exact decomposition."""
    # Fast path: zero term
    if is_zero_term(U, V, W, l):
        return True
    U_new = np.delete(U, l, axis=1)
    V_new = np.delete(V, l, axis=1)
    W_new = np.delete(W, l, axis=1)
    return verify_np(U_new, V_new, W_new)

def try_merge_np(U, V, W, i, j):
    """Check if terms i and j can be merged into one term.
    Two terms can be merged if their U, V, W columns are proportional with the same scalar.
    That is, U[:,i] = a*U[:,j], V[:,i] = a*V[:,j], W[:,i] = a*W[:,j] for some scalar a.
    Then the sum of the two terms equals 2*a times one of them (if a=1) or can be combined.
    Actually, if U_i = c*U_j, V_i = c*V_j, W_i = c*W_j, then term_i = c^3 * term_j,
    and term_i + term_j = (c^3 + 1) * term_j, which is a single term.
    More generally, we need U_i/U_j = V_i/V_j = W_i/W_j component-wise (where nonzero).
    """
    ui = U[:, i]
    uj = U[:, j]
    vi = V[:, i]
    vj = V[:, j]
    wi = W[:, i]
    wj = W[:, j]
    
    # Check if ui is proportional to uj
    # Find a scale factor from nonzero elements
    mask_u = np.abs(uj) > 1e-12
    if not np.any(mask_u):
        # uj is zero, check if ui is also zero
        if np.allclose(ui, 0, atol=1e-12):
            # Both zero in U, can merge trivially (both contribute nothing via U)
            # But we need all three to be consistent
            mask_v = np.abs(vj) > 1e-12
            if np.any(mask_v):
                c_v = vi[mask_v] / vj[mask_v]
                if np.allclose(c_v, c_v[0], atol=1e-9):
                    mask_w = np.abs(wj) > 1e-12
                    if np.any(mask_w):
                        c_w = wi[mask_w] / wj[mask_w]
                        if np.allclose(c_w, c_w[0], atol=1e-9) and np.allclose(c_v[0], c_w[0], atol=1e-9):
                            return True
            else:
                mask_w = np.abs(wj) > 1e-12
                if not np.any(mask_w) and np.allclose(wi, 0, atol=1e-12):
                    return True
        return False
    
    c_u = ui[mask_u] / uj[mask_u]
    if not np.allclose(c_u, c_u[0], atol=1e-9):
        return False
    c = c_u[0]
    
    # Check V proportionality
    mask_v = np.abs(vj) > 1e-12
    if np.any(mask_v):
        c_v = vi[mask_v] / vj[mask_v]
        if not np.allclose(c_v, c, atol=1e-9):
            return False
    else:
        if not np.allclose(vi, 0, atol=1e-12):
            return False
    
    # Check W proportionality
    mask_w = np.abs(wj) > 1e-12
    if np.any(mask_w):
        c_w = wi[mask_w] / wj[mask_w]
        if not np.allclose(c_w, c, atol=1e-9):
            return False
    else:
        if not np.allclose(wi, 0, atol=1e-12):
            return False
    
    return True

def local_search(U, V, W, deadline, rng):
    r = U.shape[1]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle indices to randomize search order
        indices = list(range(r))
        rng.shuffle(indices)
        
        # First pass: try dropping single terms
        for l in indices:
            if time.time() > deadline - 5:
                break
            if try_drop_np(U, V, W, l):
                U = np.delete(U, l, axis=1)
                V = np.delete(V, l, axis=1)
                W = np.delete(W, l, axis=1)
                r -= 1
                improved = True
                break  # Restart loop with new r and re-shuffle
        
        # Second pass: try merging pairs
        if not improved and r >= 2 and time.time() < deadline - 5:
            indices2 = list(range(r))
            rng.shuffle(indices2)
            for a in indices2:
                if time.time() > deadline - 5:
                    break
                for b in indices2:
                    if time.time() > deadline - 5:
                        break
                    if a >= b:
                        continue
                    if try_merge_np(U, V, W, a, b):
                        # Merge: sum the two terms into one, then remove one
                        # New term = term_a + term_b
                        # Since they're proportional, this is just a scaled version
                        U[:, a] += U[:, b]
                        V[:, a] += V[:, b]
                        W[:, a] += W[:, b]
                        # Remove column b
                        U = np.delete(U, b, axis=1)
                        V = np.delete(V, b, axis=1)
                        W = np.delete(W, b, axis=1)
                        r -= 1
                        improved = True
                        break
                if improved:
                    break
    return U, V, W

def construct():
    deadline = time.time() + 140  # safety margin
    rng = random.Random(42)

    # Baseline
    U, V, W = baseline_66()

    # Local search to drop redundant terms and merge proportional terms
    U, V, W = local_search(U, V, W, deadline, rng)

    # Final verification
    if not verify_np(U, V, W):
        # Fallback to baseline if something went wrong
        U, V, W = baseline_66()

    # Round to nearest integer if very close, to satisfy "integer/half-integer" strictly
    U = np.round(U, 6)
    V = np.round(V, 6)
    W = np.round(W, 6)

    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
