"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Base rank 89, then greedy local search attempts to reduce rank.
"""
import time
from fractions import Fraction
from math import gcd

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
    def add_strassen_2x2x2(r1, c1, k1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Build base decomposition with partition [4, 1]
    blocks = [(0, 4), (4, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 4 and ac == 4 and bc == 4:
                    for ri in range(0, 4, 2):
                        for ci in range(0, 4, 2):
                            for ki in range(0, 4, 2):
                                add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
    
    base_rank = len(terms)
    
    # Greedy merge search: try to find pairs of terms whose sum is rank-1
    def term_to_flat(U_dict, V_dict, W_dict):
        """Convert term to flat 3D dict."""
        flat = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    flat[key] = flat.get(key, 0) + u_val * v_val * w_val
        return flat
    
    def check_rank1(flat):
        """Check if a flat 3D tensor is rank-1. Return (a, b, c) dicts or None."""
        if not flat:
            return None
        
        u_set = set(k[0] for k in flat.keys())
        v_set = set(k[1] for k in flat.keys())
        w_set = set(k[2] for k in flat.keys())
        
        if len(u_set) <= 1 and len(v_set) <= 1 and len(w_set) <= 1:
            if len(flat) == 1:
                (i, j, k), val = list(flat.items())[0]
                if val == 0:
                    return None
                return ({i: 1}, {j: 1}, {k: val})
            return None
        
        # Need at least 2 in each dimension for non-trivial rank-1 check
        # Find a reference element with nonzero value
        ref = None
        for key, val in flat.items():
            if val != 0:
                ref = key
                break
        if ref is None:
            return None
        
        ri, rj, rk = ref
        rval = flat[ref]
        
        # Compute a, b, c vectors
        a = {}
        b = {}
        c = {}
        
        for i in u_set:
            val = flat.get((i, rj, rk), 0)
            a[i] = Fraction(val) / Fraction(rval)
        
        for j in v_set:
            val = flat.get((ri, j, rk), 0)
            b[j] = Fraction(val) / Fraction(rval)
        
        for k in w_set:
            val = flat.get((ri, rj, k), 0)
            c[k] = Fraction(val) / Fraction(rval)
        
        # Verify
        for (i, j, k), val in flat.items():
            expected = a.get(i, Fraction(0)) * b.get(j, Fraction(0)) * c.get(k, Fraction(0))
            if expected != Fraction(val):
                return None
        
        return (a, b, c)
    
    def try_merge(i, j):
        """Try to merge terms[i] and terms[j]. Return (new_term, scale) or None."""
        if time.time() > deadline - 5:
            return None
        
        t1 = term_to_flat(*terms[i])
        t2 = term_to_flat(*terms[j])
        
        combined = dict(t1)
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Remove zeros
        combined = {k: v for k, v in combined.items() if v != 0}
        
        result = check_rank1(combined)
        if result is None:
            return None
        
        a, b, c = result
        return (a, b, c)
    
    # Greedy merge loop
    merged = True
    while merged and time.time() < deadline - 10:
        merged = False
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                
                result = try_merge(i, j)
                if result is not None:
                    a, b, c = result
                    # Convert to int dicts
                    a_dict = {k: int(v) for k, v in a.items() if v != 0}
                    b_dict = {k: int(v) for k, v in b.items() if v != 0}
                    c_dict = {k: int(v) for k, v in c.items() if v != 0}
                    
                    if a_dict and b_dict and c_dict:
                        # Remove terms j and i, add merged term
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_dict, b_dict, c_dict))
                        merged = True
                        break
            if merged:
                break
    
    rank = len(terms)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
