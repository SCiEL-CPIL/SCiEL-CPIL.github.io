"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0  # Safety margin


def _strassen_2x2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple (U_dict, V_dict, W_dict) mapping (row, col) to coefficient.
    U is n x m, V is m x p, W is p x n.
    Indices are 0-based.
    """
    terms = []
    # M1 = (A11 + A22) * (B11 + B22)
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    # M2 = (A21 + A22) * B11
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    # M3 = A11 * (B12 - B22)
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    # M4 = A22 * (B21 - B11)
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    # M5 = (A11 + A12) * B22
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    # M6 = (A21 - A11) * (B11 + B12)
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    # M7 = (A12 - A22) * (B21 + B22)
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


def _generate_contiguous_partitions(dim):
    """
    Generate all contiguous partitions of 'dim' into positive integers.
    Returns a list of tuples, e.g., for dim=3: [(3,), (1,2), (2,1), (1,1,1)]
    """
    results = []
    
    def rec(start, cur):
        if start == dim:
            if cur:
                results.append(tuple(cur))
            return
        # Try all possible lengths for the next block
        for length in range(1, dim - start + 1):
            cur.append(length)
            rec(start + length, cur)
            cur.pop()
            
    rec(0, [])
    return results


def _calculate_rank(ip, jp, kp):
    """
    Calculate the total rank for a given set of partitions.
    ip, jp, kp are tuples of block sizes.
    """
    total_rank = 0
    for si in ip:
        for sj in jp:
            for sk in kp:
                if si == 2 and sj == 2 and sk == 2:
                    total_rank += 7
                else:
                    total_rank += si * sj * sk
    return total_rank


def _build_decomposition(n, m, p, ip, jp, kp):
    """
    Build U, V, W matrices for the given partitions.
    Returns U, V, W as lists of lists (or numpy arrays).
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Pre-calculate start indices for each partition
    def get_starts(partition):
        starts = []
        current = 0
        for size in partition:
            starts.append(current)
            current += size
        return starts
    
    i_starts = get_starts(ip)
    j_starts = get_starts(jp)
    k_starts = get_starts(kp)
    
    # Estimate max rank to allocate (use theoretical rank)
    max_rank = _calculate_rank(ip, jp, kp)
    
    U = [[0] * max_rank for _ in range(nm)]
    V = [[0] * max_rank for _ in range(mp)]
    W = [[0] * max_rank for _ in range(pn)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for i_idx, si in enumerate(ip):
        i_start = i_starts[i_idx]
        for j_idx, sj in enumerate(jp):
            j_start = j_starts[j_idx]
            for k_idx, sk in enumerate(kp):
                k_start = k_starts[k_idx]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen
                    for (u_pat, v_pat, w_pat) in strassen_terms:
                        for (li, lj), val_u in u_pat.items():
                            gi = i_start + li
                            gj = j_start + lj
                            U[gi * m + gj][col] = val_u
                        for (lj, lk), val_v in v_pat.items():
                            gj = j_start + lj
                            gk = k_start + lk
                            V[gj * p + gk][col] = val_v
                        for (lk, li), val_w in w_pat.items():
                            gk = k_start + lk
                            gi = i_start + li
                            W[gk * n + gi][col] = val_w
                        col += 1
                else:
                    # Use Schoolbook
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    # Trim if necessary (should match exactly)
    if col < max_rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if the decomposition exactly reproduces the matrix multiplication tensor.
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    
    # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Using einsum: 'ar,br,cr->abc'
    # But our indices are flattened.
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # Target T is (n, m, p) where T[i,j,k] = 1 if we are at the corresponding position?
    # No, the tensor for matrix multiplication is T[i,j,k] = 1 if the indices correspond to the multiplication.
    # Actually, the tensor T_{ijk} is 1 if i,j,k are the indices of A, B, C such that C_{ik} += A_{ij}B_{jk}.
    # So T_{ijk} = 1 for all i,j,k? No.
    # The tensor is defined such that C = A B is computed by T(A,B) = C.
    # T_{ijk} A_{ij} B_{jk} = C_{ik}.
    # So T_{ijk} = 1 if the indices match the multiplication structure.
    # Specifically, T is the identity tensor for the bilinear map.
    # T_{i,j,k} = 1 if i,j,k are valid indices for the multiplication?
    # Yes, for all i in 0..n-1, j in 0..m-1, k in 0..p-1, T_{i,j,k} = 1.
    
    # Let's check the reconstruction.
    # M_all[a, b, c] where a is index in U (i*m+j), b in V (j*p+k), c in W (k*n+i)
    # This doesn't directly give T[i,j,k].
    # We need to map back.
    
    # Let's construct the expected tensor E[i,j,k] = 1.
    # And check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1.
    
    # Efficient check:
    # Compute M = W @ (U^T * V^T)? No.
    # Let's use einsum to get T_reconstructed[i,j,k]
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # This is hard to do directly with einsum on flattened indices without reshaping.
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U_reshaped = Ua.reshape(n, m, -1)
    V_reshaped = Va.reshape(m, p, -1)
    W_reshaped = Wa.reshape(p, n, -1)
    
    # T_reconstructed[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Expected T is all 1s of shape (n, m, p)
    T_exp = np.ones((n, m, p), dtype=float)
    
    return np.allclose(T_rec, T_exp, atol=1e-9)


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=20):
    """
    Try to reduce rank by removing columns that are redundant.
    A column is redundant if the tensor can still be represented without it.
    This is checked by verifying the decomposition after removal.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    r = len(U[0])
    
    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        # Try removing each column
        for target in range(r):
            if time.time() > _DEADLINE:
                break
            
            # Create copies without the target column
            U_new = [row[:target] + row[target+1:] for row in U]
            V_new = [row[:target] + row[target+1:] for row in V]
            W_new = [row[:target] + row[target+1:] for row in W]
            
            # Verify
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                r -= 1
                reduced = True
                break
        
        if not reduced:
            break
            
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full_rank = n * m * p
    
    # Fallback to schoolbook
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    best = _schoolbook(n, m, p)
    best_rank = full_rank

    # Generate partitions
    ip_list = _generate_contiguous_partitions(n)
    jp_list = _generate_contiguous_partitions(m)
    kp_list = _generate_contiguous_partitions(p)
    
    # Generate all combinations and calculate rank
    candidates = []
    for ip in ip_list:
        for jp in jp_list:
            for kp in kp_list:
                rank = _calculate_rank(ip, jp, kp)
                candidates.append((rank, ip, jp, kp))
                
    # Sort by rank
    candidates.sort(key=lambda x: x[0])
    
    # Try the top candidates
    # We can try all of them if time permits, as construction is fast
    # But verification + reduction is slower.
    # Let's try the top 50 unique ranks or top 100 candidates.
    tried = 0
    for rank, ip, jp, kp in candidates:
        if time.time() > _DEADLINE:
            break
        if rank >= best_rank:
            break # No point in trying higher rank candidates
            
        # Construct
        U, V, W = _build_decomposition(n, m, p, ip, jp, kp)
        
        # Verify
        if not _verify_decomposition(U, V, W, n, m, p):
            continue
            
        # Try to reduce
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_iterations=10)
        
        # Verify reduced
        if _verify_decomposition(U_red, V_red, W_red, n, m, p):
            new_rank = len(U_red[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best = {"U": U_red, "V": V_red, "W": W_red}
                
        tried += 1
        if tried > 50:
            break

    return best

# EVOLVE-BLOCK-END
