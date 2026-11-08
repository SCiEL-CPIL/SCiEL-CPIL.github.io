"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    t0 = time.time()
    deadline = t0 + 140

    def validate(U, V, W):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col

    def schoolbook_block(U, V, W, col, i_list, j_list, k_list):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col

    def build_decomposition(j_split, k_split):
        """Build a decomposition given how to split j and k indices."""
        total_rank = 0
        for jg in j_split:
            for kg in k_split:
                if len(jg) == 2 and len(kg) == 2:
                    total_rank += 7
                else:
                    total_rank += len(jg) * len(kg)
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        col = 0
        for jg in j_split:
            for kg in k_split:
                if len(jg) == 2 and len(kg) == 2:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, jg[0], jg[1], kg[0], kg[1])
                else:
                    col = schoolbook_block(U, V, W, col, [0, 1], jg, kg)
        return U, V, W, total_rank

    # Generate all valid partitions of 0..4 into groups
    def gen_partitions(indices):
        """Generate all set partitions of the given indices."""
        if not indices:
            yield []
            return
        first = indices[0]
        rest = indices[1:]
        for part in gen_partitions(rest):
            # Put first in its own group
            yield [[first]] + part
            # Put first into each existing group
            for i in range(len(part)):
                new_part = [list(g) for g in part]
                new_part[i].append(first)
                yield new_part

    # Generate valid j_splits (partitions of 0..4) and k_splits (partitions of 0..5)
    j_indices = list(range(m))
    k_indices = list(range(p))
    
    j_splits = list(gen_partitions(j_indices))
    k_splits = list(gen_partitions(k_indices))
    
    # Sort by estimated rank (lower is better) to find good solutions faster
    def est_rank(j_split, k_split):
        r = 0
        for jg in j_split:
            for kg in k_split:
                if len(jg) == 2 and len(kg) == 2:
                    r += 7
                else:
                    r += len(jg) * len(kg)
        return r
    
    # Get all (j_split, k_split) pairs sorted by estimated rank
    all_pairs = []
    for js in j_splits:
        for ks in k_splits:
            r = est_rank(js, ks)
            all_pairs.append((r, js, ks))
    all_pairs.sort(key=lambda x: x[0])
    
    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    for est_r, js, ks in all_pairs:
        if est_r >= best_rank:
            break
        if time.time() > deadline - 5:
            break
        
        U, V, W, rank = build_decomposition(js, ks)
        if validate(U, V, W):
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break  # Found the minimum possible rank
    
    if best_U is None:
        # Fallback: pure schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Convert to lists
    U_list = [[int(best_U[i, c]) if abs(best_U[i, c] - round(best_U[i, c])) < 1e-9 else float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, c]) if abs(best_V[i, c] - round(best_V[i, c])) < 1e-9 else float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, c]) if abs(best_W[i, c] - round(best_W[i, c])) < 1e-9 else float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
