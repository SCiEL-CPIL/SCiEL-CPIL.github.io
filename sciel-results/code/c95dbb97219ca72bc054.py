import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # Term 1: M1 = (A00+A11)(B00+B11) -> C00, C11
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # Term 2: M2 = (A10+A11)B00 -> C10, -C11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(0,1): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # Term 3: M3 = A00(B01-B11) -> C01, C11
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(1,0): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # Term 4: M4 = A11(B10-B00) -> C00, C10
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (0,1): 1}
    terms.append((u4, v4, w4))
    
    # Term 5: M5 = (A00+A01)B11 -> -C00, C01
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (1,0): 1}
    terms.append((u5, v5, w5))
    
    # Term 6: M6 = (A10-A00)(B00+B01) -> C11
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # Term 7: M7 = (A01-A11)(B10+B11) -> C00
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms


def _build_decomposition(n, m, p, i_groups, j_groups, k_groups):
    """Build a block decomposition with Strassen on 2x2x2 blocks."""
    strassen_terms = _strassen_2x2x2_terms()
    
    # Count total rank
    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                if si == 2 and sj == 2 and sk == 2:
                    r = 7
                else:
                    r = si * sj * sk
                total_rank += r
                block_info.append((ig, jg, kg, r, si, sj, sk))
    
    # Allocate matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ig, jg, kg, r, si, sj, sk in block_info:
        if si == 2 and sj == 2 and sk == 2:
            # Use Strassen
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            # Schoolbook for this sub-block
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}, total_rank


def _verify_exact(U, V, W, n, m, p):
    """Verify decomposition using exact arithmetic."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for rr in range(r):
                    u = U[i*m+j][rr]
                    v = V[j*p+k][rr]
                    w = W[k*n+i][rr]
                    if u != 0 and v != 0 and w != 0:
                        total += Fraction(u) * Fraction(v) * Fraction(w)
                if total != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _generate_partitions(d, max_size=2):
    """Generate all partitions of d into parts of size 1 or 2."""
    if d == 0:
        yield []
        return
    if d == 1:
        yield [[0]]
        return
    if d >= 2:
        # Start with a pair
        yield from _generate_partitions_pair(d, 2)

def _generate_partitions_pair(d, current_size):
    """Generate partitions where parts are 1 or 2."""
    if d == 0:
        yield []
        return
    if d == 1:
        yield [1]
        return
    if d >= 2:
        yield [2] + list(_generate_partitions_pair(d-2, 2))
        yield [1] + list(_generate_partitions_pair(d-1, 2))

def _partitions_to_groups(d, part_sizes):
    """Convert part sizes to actual index groups."""
    groups = []
    start = 0
    for size in part_sizes:
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Fallback: schoolbook
    schoolbook_result = _schoolbook(n, m, p)
    best_result = schoolbook_result
    best_rank = n * m * p
    
    # Generate all possible partitions
    i_part_sizes = list(_generate_partitions(n))
    j_part_sizes = list(_generate_partitions(m))
    k_part_sizes = list(_generate_partitions(p))
    
    # Try all combinations of partitions
    for i_parts in i_part_sizes:
        if time.time() > deadline:
            break
        i_groups = _partitions_to_groups(n, i_parts)
        
        for j_parts in j_part_sizes:
            if time.time() > deadline:
                break
            j_groups = _partitions_to_groups(m, j_parts)
            
            for k_parts in k_part_sizes:
                if time.time() > deadline:
                    break
                k_groups = _partitions_to_groups(p, k_parts)
                
                # Calculate expected rank
                expected_rank = 0
                for ig in i_groups:
                    for jg in j_groups:
                        for kg in k_groups:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                expected_rank += 7
                            else:
                                expected_rank += si * sj * sk
                
                if expected_rank >= best_rank:
                    continue
                
                # Build and verify
                result, rank = _build_decomposition(n, m, p, i_groups, j_groups, k_groups)
                
                if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
