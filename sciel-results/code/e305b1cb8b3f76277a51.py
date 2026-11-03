"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+l, :], V_arr[l*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_skip(U, V, W, n, m, p, rank, skip_term):
    """Verify decomposition with term skip_term zeroed out."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_col = U_arr[:, skip_term].copy()
    V_col = V_arr[:, skip_term].copy()
    W_col = W_arr[:, skip_term].copy()
    
    # Compute the contribution of the skipped term
    # residual[i,l,k] = U[i*m+l,skip] * V[l*p+k,skip] * W[k*n+i,skip]
    # We need: sum_{r != skip} U[i*m+l,r]*V[l*p+k,r]*W[k*n+i,r] == 1
    # i.e., full_sum - skip_contribution == 1
    # i.e., skip_contribution == 0 for all (i,l,k) where full_sum == 1
    
    # More directly: check if full_sum - skip_contribution == 1 for all (i,l,k)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                full_s = np.dot(U_arr[i*m+l, :], V_arr[l*p+k, :] * W_arr[k*n+i, :])
                skip_s = U_arr[i*m+l, skip_term] * V_arr[l*p+k, skip_term] * W_arr[k*n+i, skip_term]
                if abs(full_s - skip_s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs, k_singles=None):
    """Build a Strassen-based decomposition with given pairings.
    
    k_pairs: list of (k0,k1) pairs for Strassen 2x2 blocks
    k_singles: list of single k indices (default: all k not in k_pairs)
    """
    if k_singles is None:
        used_k = set()
        for k0, k1 in k_pairs:
            used_k.add(k0)
            used_k.add(k1)
        k_singles = [k for k in range(p) if k not in used_k]
    
    # Rank calculation:
    # For each j_pair and k_pair: 7 terms (Strassen 2x2)
    # For each j_pair and k_single: 4 terms (standard 2x1)
    # For each j_single and k: 2 terms
    rank = 7 * len(j_pairs) * len(k_pairs) + 4 * len(j_pairs) * len(k_singles) + 2 * (m - 2*len(j_pairs)) * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
        
        # For each k_single with this j_pair: 4 standard products
        for ks in k_singles:
            U[0*m + j0][col] += 1
            V[j0*p + ks][col] += 1
            W[ks*n + 0][col] += 1
            col += 1
            
            U[0*m + j1][col] += 1
            V[j1*p + ks][col] += 1
            W[ks*n + 0][col] += 1
            col += 1
            
            U[1*m + j0][col] += 1
            V[j0*p + ks][col] += 1
            W[ks*n + 1][col] += 1
            col += 1
            
            U[1*m + j1][col] += 1
            V[j1*p + ks][col] += 1
            W[ks*n + 1][col] += 1
            col += 1
    
    # Remaining j indices: standard products
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    
    for j in range(m):
        if j in used_j:
            continue
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val - round(val)) < tol:
                new_row.append(float(round(val)))
            elif abs(val - (round(val*2)/2.0)) < tol:
                new_row.append(round(val*2)/2.0)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def _remove_column(U, V, W, idx, rank):
    """Remove column idx from U, V, W."""
    U_final = []
    for row in U:
        U_final.append([row[j] for j in range(rank) if j != idx])
    V_final = []
    for row in V:
        V_final.append([row[j] for j in range(rank) if j != idx])
    W_final = []
    for row in W:
        W_final.append([row[j] for j in range(rank) if j != idx])
    return U_final, V_final, W_final, rank - 1

def _eliminate_redundant_terms(U, V, W, n, m, p, rank, time_fn):
    """Try to eliminate redundant terms. Returns (U, V, W, new_rank)."""
    current_U, current_V, current_W, current_rank = U, V, W, rank
    eliminated = set()
    
    # Sort terms by L2 norm (ascending) - smaller norms more likely redundant
    for iteration in range(current_rank):
        if time_fn() < 3:
            break
        
        norms = []
        for r in range(current_rank):
            if r in eliminated:
                norms.append(float('inf'))
                continue
            norm_sq = sum(current_U[i][r]**2 for i in range(n*m))
            norm_sq += sum(current_V[l][r]**2 for l in range(m*p))
            norm_sq += sum(current_W[k][r]**2 for k in range(p*n))
            norms.append(norm_sq)
        
        # Find minimum norm term not yet eliminated
        min_idx = int(np.argmin(norms))
        if norms[min_idx] == float('inf'):
            break
        
        if _verify_skip(current_U, current_V, current_W, n, m, p, current_rank, min_idx):
            current_U, current_V, current_W, current_rank = _remove_column(
                current_U, current_V, current_W, min_idx, current_rank
            )
            eliminated.add(min_idx)
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin before 150s limit
    
    def _time_left():
        return _deadline - time.time()
    
    best_rank = float('inf')
    best_result = None
    
    # Generate all possible k-pair configurations for p=6
    # We can have 0, 1, 2, or 3 k-pairs
    # For each number of pairs, generate all matchings
    
    def _gen_k_configs(p):
        """Generate all ways to partition k indices into pairs and singles."""
        configs = []
        
        # 3 pairs (all paired)
        k_list = list(range(p))
        for perm in set(combinations(k_list, 2)):
            remaining = [x for x in k_list if x not in perm]
            for perm2 in set(combinations(remaining, 2)):
                remaining2 = [x for x in remaining if x not in perm2]
                if len(remaining2) == 2:
                    perm3 = tuple(sorted(remaining2))
                    k_pairs = [tuple(sorted(perm)), tuple(sorted(perm2)), perm3]
                    k_pairs = tuple(sorted(k_pairs))
                    configs.append((list(k_pairs), []))
        
        # 2 pairs + 2 singles
        for pair1 in combinations(k_list, 2):
            remaining1 = [x for x in k_list if x not in pair1]
            for pair2 in combinations(remaining1, 2):
                remaining2 = [x for x in remaining1 if x not in pair2]
                k_pairs = [tuple(sorted(pair1)), tuple(sorted(pair2))]
                k_pairs = tuple(sorted(k_pairs))
                k_singles = sorted(remaining2)
                configs.append((list(k_pairs), k_singles))
        
        # 1 pair + 4 singles
        for pair in combinations(k_list, 2):
            remaining = [x for x in k_list if x not in pair]
            k_pairs = [tuple(sorted(pair))]
            k_singles = sorted(remaining)
            configs.append((k_pairs, k_singles))
        
        # 0 pairs + 6 singles
        configs.append(([], list(range(p))))
        
        # Deduplicate
        seen = set()
        unique_configs = []
        for kp, ks in configs:
            key = (tuple(sorted(kp)), tuple(ks))
            if key not in seen:
                seen.add(key)
                unique_configs.append((kp, ks))
        
        return unique_configs
    
    k_configs = _gen_k_configs(p)
    
    # Generate j-pair configurations for m=5
    j_configs = []
    # 2 pairs + 1 single
    for jp1 in combinations(range(m), 2):
        remaining1 = [x for x in range(m) if x not in jp1]
        for jp2 in combinations(remaining1, 2):
            remaining2 = [x for x in remaining1 if x not in jp2]
            j_pairs = [tuple(sorted(jp1)), tuple(sorted(jp2))]
            j_pairs = tuple(sorted(j_pairs))
            j_singles = sorted(remaining2)
            j_configs.append((list(j_pairs), j_singles))
    
    # 1 pair + 3 singles
    for jp in combinations(range(m), 2):
        remaining = [x for x in range(m) if x not in jp]
        j_pairs = [tuple(sorted(jp))]
        j_singles = sorted(remaining)
        j_configs.append((j_pairs, j_singles))
    
    # 0 pairs + 5 singles
    j_configs.append(([], list(range(m))))
    
    # Deduplicate j_configs
    seen_j = set()
    unique_j_configs = []
    for jp, js in j_configs:
        key = (tuple(sorted(jp)), tuple(js))
        if key not in seen_j:
            seen_j.add(key)
            unique_j_configs.append((jp, js))
    j_configs = unique_j_configs
    
    # Try all combinations
    total_configs = len(j_configs) * len(k_configs)
    configs_tried = 0
    
    for j_pairs, j_singles in j_configs:
        if _time_left() < 10:
            break
        for k_pairs, k_singles in k_configs:
            if _time_left() < 5:
                break
            
            configs_tried += 1
            
            U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs, k_singles)
            rank = len(U[0])
            
            if rank >= best_rank:
                continue
            
            if not _verify_decomp(U, V, W, n, m, p, rank):
                continue
            
            # Try to eliminate redundant terms
            U2, V2, W2, rank2 = _eliminate_redundant_terms(U, V, W, n, m, p, rank, _time_left)
            
            if rank2 < best_rank:
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    # Fallback: standard configuration
    if best_result is None:
        U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
        best_result = {"U": U, "V": V, "W": W}
        best_rank = len(U[0])
    
    # Snap to half-integers for cleaner output
    best_result["U"] = _snap_to_half_integer(best_result["U"])
    best_result["V"] = _snap_to_half_integer(best_result["V"])
    best_result["W"] = _snap_to_half_integer(best_result["W"])
    
    # Final verification
    if not _verify_decomp(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
        # Fallback to standard
        U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
        best_result = {"U": U, "V": V, "W": W}
        best_rank = len(U[0])
    
    return best_result
# EVOLVE-BLOCK-END
