"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Initial rank ~117, then greedy local search attempts to reduce rank.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

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
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
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
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Partition: [4, 1]
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
                    # Decompose 4x4x4 into four 2x2x2 Strassen blocks
                    for ri in range(0, 4, 2):
                        for ci in range(0, 4, 2):
                            for ki in range(0, 4, 2):
                                add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Greedy local search: try to merge pairs of terms
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert term to a flat 3D tensor representation using Fraction for exactness."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    val = Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    tensor[key] = tensor.get(key, Fraction(0)) + val
        return tensor
    
    def check_rank1(combined):
        """Check if a sparse tensor is rank-1. Returns (a_dict, b_dict, c_dict) as Fractions or None."""
        if not combined:
            return None
        
        # Find a nonzero reference element
        ref_val = None
        ref_i = ref_j = ref_k = None
        for (i, j, k), val in combined.items():
            if val != 0:
                ref_val = val
                ref_i, ref_j, ref_k = i, j, k
                break
        
        if ref_val is None:
            return None
        
        # Derive factors using Fraction arithmetic
        # a[i] = T[i, ref_j, ref_k] / ref_val
        # b[j] = T[ref_i, j, ref_k] / ref_val  
        # c[k] = T[ref_i, ref_j, k] / ref_val
        # Then T[i,j,k] should equal a[i]*b[j]*c[k] for all (i,j,k)
        
        a_dict = {}
        b_dict = {}
        c_dict = {}
        
        # Collect all indices
        all_i = set()
        all_j = set()
        all_k = set()
        for (i, j, k) in combined.keys():
            all_i.add(i)
            all_j.add(j)
            all_k.add(k)
        
        # Compute a[i] for all i
        for i in all_i:
            ti = combined.get((i, ref_j, ref_k), Fraction(0))
            if ti != 0:
                a_dict[i] = ti / ref_val
        
        # Compute b[j] for all j
        for j in all_j:
            tj = combined.get((ref_i, j, ref_k), Fraction(0))
            if tj != 0:
                b_dict[j] = tj / ref_val
        
        # Compute c[k] for all k
        for k in all_k:
            tk = combined.get((ref_i, ref_j, k), Fraction(0))
            if tk != 0:
                c_dict[k] = tk / ref_val
        
        # Verify: T[i,j,k] == a[i]*b[j]*c[k] for all (i,j,k)
        for (i, j, k), val in combined.items():
            expected = a_dict.get(i, Fraction(0)) * b_dict.get(j, Fraction(0)) * c_dict.get(k, Fraction(0))
            if val != expected:
                return None
        
        # Check that all entries are integer or half-integer
        def is_valid(val):
            if val == 0:
                return True
            num, den = val.numerator, val.denominator
            if den == 1:
                return True
            if den == 2:
                return True
            return False
        
        for d in [a_dict, b_dict, c_dict]:
            for v in d.values():
                if not is_valid(v):
                    return None
        
        return (a_dict, b_dict, c_dict)
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor."""
        if time.time() > deadline:
            return None
        
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, Fraction(0)) + v
        
        # Remove zeros
        combined = {k: v for k, v in combined.items() if v != 0}
        
        return check_rank1(combined)
    
    # Try merging pairs - multiple passes with time limit
    def frac_to_val(f):
        if f.denominator == 1:
            return f.numerator
        elif f.denominator == 2:
            return f.numerator / 2
        else:
            return float(f)
    
    max_passes = 100
    for _ in range(max_passes):
        if time.time() > deadline - 10:
            break
        merged = False
        i = 0
        while i < len(terms) and time.time() < deadline - 8:
            j = i + 1
            while j < len(terms) and time.time() < deadline - 8:
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    a_new, b_new, c_new = result
                    
                    a_conv = {k: frac_to_val(v) for k, v in a_new.items()}
                    b_conv = {k: frac_to_val(v) for k, v in b_new.items()}
                    c_conv = {k: frac_to_val(v) for k, v in c_new.items()}
                    
                    # Remove zeros
                    a_conv = {k: v for k, v in a_conv.items() if v != 0}
                    b_conv = {k: v for k, v in b_conv.items() if v != 0}
                    c_conv = {k: v for k, v in c_conv.items() if v != 0}
                    
                    if a_conv and b_conv and c_conv:
                        # Remove terms i and j, add new term
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_conv, b_conv, c_conv))
                        merged = True
                        break
                j += 1
            if merged:
                break
            i += 1
    
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
    
    # Verify correctness
    def verify():
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j]*V[j*p+k]*W[k*n+i]
        # Check a few random entries
        import random
        random.seed(42)
        for _ in range(20):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            # Expected contribution from this (i,j,k) triple
            expected = 1  # In matrix mult tensor, T[i,j,k]=1
            actual = 0
            for r in range(rank):
                actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            if abs(actual - expected) > 1e-9:
                return False
        return True
    
    if not verify():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    r = ii * m * p + jj * p + kk
                    U[ii * m + jj][r] = 1
                    V[jj * p + kk][r] = 1
                    W[kk * n + ii][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
