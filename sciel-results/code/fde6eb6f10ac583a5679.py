"""Rewritten program: structured greedy decomposition for 4x4x9 matrix multiplication tensor."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    dim_a = n * m  # 16
    dim_b = m * p  # 36
    dim_c = p * n  # 36
    
    # Build the set of non-zero entries of the multiplication tensor
    # T[a,b,c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    nonzeros = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzeros.add((a, b, c))
    
    remaining = set(nonzeros)
    terms = []  # list of (u, v, w) as dicts or sparse vectors
    
    # Strategy: try structured rank-1 terms that cover grids of entries
    # A term is defined by (A_support, B_support, C_support) with coefficients
    # For simplicity, use coefficient 1 for all supported entries
    
    # Precompute: for each (a,b), which c values are in the tensor
    ab_to_c = {}
    for (a, b, c) in nonzeros:
        key = (a, b)
        if key not in ab_to_c:
            ab_to_c[key] = set()
        ab_to_c[key].add(c)
    
    # For each a, which (b,c) pairs exist
    a_to_bc = {}
    for (a, b, c) in nonzeros:
        if a not in a_to_bc:
            a_to_bc[a] = set()
        a_to_bc[a].add((b, c))
    
    # For each b, which (a,c) pairs exist  
    b_to_ac = {}
    for (a, b, c) in nonzeros:
        if b not in b_to_ac:
            b_to_ac[b] = set()
        b_to_ac[b].add((a, c))
    
    # For each c, which (a,b) pairs exist
    c_to_ab = {}
    for (a, b, c) in nonzeros:
        if c not in c_to_ab:
            c_to_ab[c] = set()
        c_to_ab[c].add((a, b))
    
    def coverage_count(A_sup, B_sup, C_sup):
        """Count how many remaining entries are covered by this rank-1 term."""
        count = 0
        for a in A_sup:
            for b in B_sup:
                for c in C_sup:
                    if (a, b, c) in remaining:
                        count += 1
        return count
    
    # Generate candidate structured terms
    # Type 1: single (a,b) with all its c values
    # Type 2: multiple a's with same b, all c values
    # Type 3: full grid patterns
    
    max_iterations = 200
    iteration = 0
    
    while remaining and iteration < max_iterations:
        iteration += 1
        best_term = None
        best_count = 0
        
        # Try single-entry terms (fallback)
        for (a, b, c) in list(remaining)[:50]:  # limit for speed
            cnt = 1  # covers just this one
            if cnt > best_count:
                best_count = cnt
                best_term = ({a}, {b}, {c})
        
        # Try type 1: fix (a,b), take all c in remaining for this (a,b)
        for (a, b) in list(ab_to_c.keys()):
            c_set = set()
            for c in ab_to_c[(a, b)]:
                if (a, b, c) in remaining:
                    c_set.add(c)
            if len(c_set) >= 2:
                cnt = len(c_set)
                if cnt > best_count:
                    best_count = cnt
                    best_term = ({a}, {b}, c_set)
        
        # Try type 2: fix b, take multiple a's that share c values
        for b in range(dim_b):
            if b not in b_to_ac:
                continue
            ac_set = b_to_ac[b]
            # Group by c
            c_to_as = {}
            for (a, c) in ac_set:
                if (a, b, c) in remaining:
                    if c not in c_to_as:
                        c_to_as[c] = set()
                    c_to_as[c].add(a)
            # Try taking all a's for a given c
            for c, a_set in c_to_as.items():
                if len(a_set) >= 2:
                    cnt = len(a_set)
                    if cnt > best_count:
                        best_count = cnt
                        best_term = (a_set, {b}, {c})
        
        # Try type 3: fix a, take multiple (b,c) that form a grid
        for a in range(dim_a):
            if a not in a_to_bc:
                continue
            bc_set = a_to_bc[a]
            # Group by b
            b_to_cs = {}
            for (b, c) in bc_set:
                if (a, b, c) in remaining:
                    if b not in b_to_cs:
                        b_to_cs[b] = set()
                    b_to_cs[b].add(c)
            # Try taking all b's with common c values
            if len(b_to_cs) >= 2:
                # Find common c values across multiple b's
                all_cs = [cs for cs in b_to_cs.values()]
                if all_cs:
                    common_cs = set.intersection(*all_cs) if len(all_cs) > 1 else all_cs[0]
                    if common_cs:
                        b_set = set(b_to_cs.keys())
                        cnt = coverage_count({a}, b_set, common_cs)
                        if cnt > best_count:
                            best_count = cnt
                            best_term = ({a}, b_set, common_cs)
        
        # Try type 4: small grids (2x2 or 2x3)
        # For pairs of a values
        a_list = list(range(dim_a))
        for i in range(len(a_list)):
            a1 = a_list[i]
            if a1 not in a_to_bc:
                continue
            for j in range(i+1, len(a_list)):
                a2 = a_list[j]
                if a2 not in a_to_bc:
                    continue
                bc1 = a_to_bc[a1]
                bc2 = a_to_bc[a2]
                # Find common b values
                b1_set = set(b for (b, c) in bc1 if (a1, b, c) in remaining)
                b2_set = set(b for (b, c) in bc2 if (a2, b, c) in remaining)
                common_b = b1_set & b2_set
                if len(common_b) >= 2:
                    # Find common c for these b's
                    c1_set = set(c for (b, c) in bc1 if b in common_b and (a1, b, c) in remaining)
                    c2_set = set(c for (b, c) in bc2 if b in common_b and (a2, b, c) in remaining)
                    common_c = c1_set & c2_set
                    if common_c:
                        cnt = coverage_count({a1, a2}, common_b, common_c)
                        if cnt > best_count:
                            best_count = cnt
                            best_term = ({a1, a2}, common_b, common_c)
        
        # If no good term found, take single entry
        if best_term is None or best_count <= 1:
            (a, b, c) = next(iter(remaining))
            best_term = ({a}, {b}, {c})
            best_count = 1
        
        # Apply the term
        A_sup, B_sup, C_sup = best_term
        for a in A_sup:
            for b in B_sup:
                for c in C_sup:
                    if (a, b, c) in remaining:
                        remaining.discard((a, b, c))
        terms.append((A_sup, B_sup, C_sup))
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(dim_a)]
    V = [[0] * rank for _ in range(dim_b)]
    W = [[0] * rank for _ in range(dim_c)]
    
    for r, (A_sup, B_sup, C_sup) in enumerate(terms):
        for a in A_sup:
            U[a][r] = 1
        for b in B_sup:
            V[b][r] = 1
        for c in C_sup:
            W[c][r] = 1
    
    return {"U": U, "V": V, "W": W}
