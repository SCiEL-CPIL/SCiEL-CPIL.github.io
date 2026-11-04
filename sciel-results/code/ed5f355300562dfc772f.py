import time
import numpy as np
import random

N, M, P = 4, 4, 9
DEADLINE = 135.0

# --- Build the multiplication tensor T[i,j,k] = 1 (index by flattened a,b,c) ---
# a index: i*M + j  (0..15)
# b index: j*P + k  (0..35)
# c index: k*N + i  (0..35)
# We need U[a,r] V[b,r] W[c,r] such that sum_r U[a,r]V[b,r]W[c,r] = 1 iff (i,j,k) valid else 0.
# Actually the tensor is T_{a,b,c} = 1 if c = k*N+i where a=i*M+j, b=j*P+k.
# So for each (i,j,k) in [0..3]x[0..3]x[0..8], we have a = i*M+j, b = j*P+k, c = k*N+i.
# The constraint is: for all a,b,c: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
# where T[a,b,c] = 1 if there exists i,j,k with a=i*M+j, b=j*P+k, c=k*N+i, else 0.
# Note: for given a=i*M+j and b=j'*P+k, we need j==j' for it to be a valid entry, and c must equal k*N+i.
# So T[a,b,c] = 1 iff (a//M, b//P) == (a%M, b%P) is consistent... let's just build it directly.

def build_tensor():
    """Return T as a dict mapping (a,b,c) -> 1 for valid entries."""
    T = {}
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[(a, b, c)] = 1
    return T

def build_tensor_array():
    """Return T as a 3D numpy array of shape (N*M, M*P, P*N) with 0/1 entries."""
    T = np.zeros((N*M, M*P, P*N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T

T_ARR = None
def get_T():
    global T_ARR
    if T_ARR is None:
        T_ARR = build_tensor_array()
    return T_ARR

def validate(U, V, W):
    """Check if U,V,W (lists of lists or 2D arrays) form a valid decomposition.
    U: (N*M, r), V: (M*P, r), W: (P*N, r)
    Returns True if exact."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    r = U.shape[1]
    T = get_T()
    # Compute sum_r U[:,r] V[:,r] W[:,r] as a 3D tensor
    # For each r, outer product
    # To be efficient, we can iterate over r and accumulate
    # But for small r this is fine.
    # Actually, let's do it in a vectorized way:
    # Result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can compute this as: for each a, b, the vector over r is U[a,:]*V[b,:], then dot with W[:,c]
    # Or: Result = np.einsum('ar,br,cr->abc', U, V, W)
    try:
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    except Exception:
        return False
    return np.array_equal(result, T)

def schoolbook():
    """Return schoolbook decomposition as (U, V, W) with rank N*M*P."""
    r = N * M * P
    U = np.zeros((N*M, r), dtype=np.int64)
    V = np.zeros((M*P, r), dtype=np.int64)
    W = np.zeros((P*N, r), dtype=np.int64)
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1
    return U, V, W

def strassen_2x2x2_terms():
    """Return list of (a_idx, b_idx, c_idx, coeff) for 2x2*2x2->2x2 Strassen."""
    # Indices: a = i*2+j, b = j*2+k, c = k*2+i for i,j,k in 0..1
    terms = []
    # M1 = (A00+A11)(B00+B11)
    for (ai, aj) in [(0,0),(1,1)]:
        for (bi, bj) in [(0,0),(1,1)]:
            a = ai*2+aj
            b = bi*2+bj
            # This term contributes to C00,C01,C10,C11 via Strassen
            # We need to track which C entries it affects
            # Let's use the standard Strassen structure
    # Let me just hardcode the 7 terms with their (a,b) support and C coefficients
    # Term s: A-slice and B-slice, and C-combination
    # M1: A00+A11, B00+B11 -> C00+=M1, C01+=M1, C10+=M1, C11-=M1
    # M2: A10+A11, B00 -> C10+=M2, C11+=M2
    # M3: A00, B10+B11 -> C00+=M3, C10+=M3
    # M4: A11, B01-B11 -> C01+=M4, C11+=M4
    # M5: A00+A01, B11 -> C01+=M5, C11+=M5
    # M6: A10-A00, B00+B01 -> C00-=M6, C10+=M6
    # M7: A01-A11, B10+B11 -> C00+=M7, C11-=M7
    
    # For each term, we have a set of (a, coeff_a) and (b, coeff_b)
    # and a set of (c, coeff_c)
    strassen = [
        # M1
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(0,1,1),(1,0,1),(1,1,-1))]),
        # M2
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1),(1,1,1))]),
        # M3
        ([((0,0,1))], [((1,0,1),(1,1,1))], [((0,0,1),(1,0,1))]),
        # M4
        ([((1,1,1))], [((0,1,1),(1,1,-1))], [((0,1,1),(1,1,1))]),
        # M5
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,1,1),(1,1,1))]),
        # M6
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((0,0,-1),(1,0,1))]),
        # M7
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,1),(1,1,-1))]),
    ]
    # Now expand to global indices for 2x2
    # a_local = i*2+j, b_local = j*2+k, c_local = k*2+i
    result = []
    for (a_list, b_list, c_list) in strassen:
        for (ai, aj, ac) in a_list:
            a_global = ai * 2 + aj
            for (bi, bj, bc) in b_list:
                b_global = bi * 2 + bj
                for (ci, cj, cc) in c_list:
                    c_global = ci * 2 + cj
                    coeff = ac * bc * cc
                    if coeff != 0:
                        result.append((a_global, b_global, c_global, coeff))
    return result

def build_strassen_4x4x4():
    """Build 4x4*4x4->4x4 using Strassen-of-Strassen. Returns (U,V,W) arrays."""
    # Use the hybrid approach from the current program but build as arrays
    # Let's just use the schoolbook for 4x4x4 for simplicity in the seed,
    # then local search will try to reduce it.
    # Actually, let's build a proper Strassen 4x4x4.
    # For now, use schoolbook as seed.
    r = 16
    U = np.zeros((16, r), dtype=np.int64)
    V = np.zeros((16, r), dtype=np.int64)
    W = np.zeros((16, r), dtype=np.int64)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i*4+j
                b = j*4+k
                c = k*4+i
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1
    return U, V, W

def build_seed_4x4x9():
    """Build a seed decomposition for 4x4*4x9->4x9.
    Use schoolbook for the first 4 columns (4x4*4x4) and schoolbook for last 5 columns.
    Total rank = 4*4*4 + 4*4*5 = 64 + 80 = 144.
    """
    r = N*M*P  # 144
    U = np.zeros((N*M, r), dtype=np.int64)
    V = np.zeros((M*P, r), dtype=np.int64)
    W = np.zeros((P*N, r), dtype=np.int64)
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i*M + j
                b = j*P + k
                c = k*N + i
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1
    return U, V, W

def try_merge(U, V, W, r, deadline):
    """Try to reduce rank by finding linearly dependent columns.
    Returns (U', V', W', r') with potentially lower rank."""
    t0 = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try to find pairs of columns that can be merged
    # Column c and column c' can be merged if there exists a scalar lambda
    # such that U[:,c] = lambda*U[:,c'], V[:,c] = lambda*V[:,c'], W[:,c] = lambda*W[:,c']
    # Or more generally, if the 3-tuples of column vectors are linearly dependent.
    
    # Simpler: check if any two columns are identical (lambda=1)
    # Or proportional.
    
    removed = 0
    changed = True
    while changed and time.time() - t0 < deadline - 5:
        changed = False
        for c in range(r - removed):
            if time.time() - t0 > deadline - 5:
                break
            Uc = U[:, c]
            Vc = V[:, c]
            Wc = W[:, c]
            if np.all(Uc == 0) and np.all(Vc == 0) and np.all(Wc == 0):
                # Remove zero column
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                r -= 1
                removed += 1
                changed = True
                continue
            for c2 in range(c+1, r - removed):
                U2c = U[:, c2]
                V2c = V[:, c2]
                W2c = W[:, c2]
                # Check if proportional
                # Find lambda from first non-zero element
                lambda_val = None
                for a in range(N*M):
                    if Uc[a] != 0:
                        if U2c[a] == 0:
                            break
                        lambda_val = U2c[a] / Uc[a]
                        break
                    elif U2c[a] != 0:
                        break
                if lambda_val is None:
                    # Try from V
                    for b in range(M*P):
                        if Vc[b] != 0:
                            if V2c[b] == 0:
                                break
                            lambda_val = V2c[b] / Vc[b]
                            break
                        elif V2c[b] != 0:
                            break
                if lambda_val is None:
                    for cidx in range(P*N):
                        if Wc[cidx] != 0:
                            if W2c[cidx] == 0:
                                break
                            lambda_val = W2c[cidx] / Wc[cidx]
                            break
                        elif W2c[cidx] != 0:
                            break
                if lambda_val is None:
                    continue
                # Verify proportionality
                try:
                    if not np.allclose(Uc * lambda_val, U2c, atol=1e-9):
                        continue
                    if not np.allclose(Vc * lambda_val, V2c, atol=1e-9):
                        continue
                    if not np.allclose(Wc * lambda_val, W2c, atol=1e-9):
                        continue
                except Exception:
                    continue
                # Merge: subtract lambda * column c from column c2 (or just remove one)
                # Since they're proportional, we can remove one
                U[:, c2] -= lambda_val * Uc
                V[:, c2] -= lambda_val * Vc
                W[:, c2] -= lambda_val * Wc
                # Now column c2 should be zero
                U = np.delete(U, c2, axis=1)
                V = np.delete(V, c2, axis=1)
                W = np.delete(W, c2, axis=1)
                r -= 1
                removed += 1
                changed = True
                break
    return U, V, W, r

def random_pivot_swap(U, V, W, r, deadline):
    """Try random modifications to reduce rank."""
    t0 = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    best_rank = r
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(1000):
        if time.time() - t0 > deadline - 10:
            break
        # Try to zero out a random non-zero entry by adjusting other entries
        # This is complex, so let's just try small random perturbations
        # and see if rank can be reduced.
        # For now, just do column merging.
        U, V, W, r = try_merge(U, V, W, r, deadline)
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U, best_V, best_W, best_rank

def construct():
    t0 = time.time()
    
    # Build seed
    U, V, W = build_seed_4x4x9()
    r = U.shape[1]
    
    # Validate seed
    if not validate(U, V, W):
        # Shouldn't happen, but fallback
        U, V, W = schoolbook()
        r = U.shape[1]
    
    # Try to reduce rank
    deadline = t0 + DEADLINE
    U, V, W, r = try_merge(U, V, W, r, deadline)
    
    # Validate after reduction
    if not validate(U, V, W):
        # Revert to seed
        U, V, W = build_seed_4x4x9()
        r = U.shape[1]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
