"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with partition [4,1] and Strassen for 2x2x2 subblocks.
Initial rank ~89, then greedy local search attempts to reduce rank.
Improved merge logic and additional reduction passes.
"""
import time
import numpy as np
from fractions import Fraction
from math import gcd

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
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
    
    def term_to_tensor(U_dict, V_dict, W_dict):
        """Convert term to a flat 3D tensor representation."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return tensor
    
    def can_merge(term1, term2):
        """Check if term1 + term2 is a rank-1 tensor.
        Returns (scale, a_int, b_int, c_int) if mergeable, else None."""
        if time.time() > deadline:
            return None
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        combined = {}
        for k, v in t1.items():
            combined[k] = v
        for k, v in t2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Remove zeros
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            return None
        
        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))
        
        if len(u_indices) <= 1 and len(v_indices) <= 1 and len(w_indices) <= 1:
            if len(u_indices) == 1 and len(v_indices) == 1 and len(w_indices) == 1:
                val = combined[(u_indices[0], v_indices[0], w_indices[0])]
                if abs(val) < 1e-12:
                    return None
                return (1, {u_indices[0]: int(val)}, {v_indices[0]: 1}, {w_indices[0]: 1})
            return None
        
        # For rank-1 check: tensor[i,j,k] = a[i]*b[j]*c[k]
        # Pick a reference element that is nonzero
        ref_i, ref_j, ref_k = None, None, None
        ref_val = None
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    val = combined.get((i, j, k), 0)
                    if abs(val) > 1e-12:
                        ref_i, ref_j, ref_k = i, j, k
                        ref_val = val
                        break
                if ref_val is not None:
                    break
            if ref_val is not None:
                break
        
        if ref_val is None:
            return None
        
        # Compute a[i], b[j], c[k] using the reference
        a = {}
        b = {}
        c = {}
        a[ref_i] = 1.0
        b[ref_j] = 1.0
        c[ref_k] = 1.0
        
        for i in u_indices:
            if i != ref_i:
                val = combined.get((i, ref_j, ref_k), 0)
                a[i] = val / ref_val
        
        for j in v_indices:
            if j != ref_j:
                val = combined.get((ref_i, j, ref_k), 0)
                b[j] = val / ref_val
        
        for k in w_indices:
            if k != ref_k:
                val = combined.get((ref_i, ref_j, k), 0)
                c[k] = val / ref_val
        
        # Verify: tensor[i,j,k] should equal a[i]*b[j]*c[k]*ref_val
        for (i, j, k), val in combined.items():
            expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
            if abs(val - expected) > 1e-9:
                return None
        
        # Scale to get integer/half-integer entries
        all_vals = list(a.values()) + list(b.values()) + list(c.values())
        a_frac = {k: Fraction(v).limit_denominator(1000) for k, v in a.items()}
        b_frac = {k: Fraction(v).limit_denominator(1000) for k, v in b.items()}
        c_frac = {k: Fraction(v).limit_denominator(1000) for k, v in c.items()}
        
        # Find common denominator
        denoms = [d.numerator for d in a_frac.values()] + \
                 [d.numerator for d in b_frac.values()] + \
                 [d.numerator for d in c_frac.values()]
        if not denoms:
            return None
        lcm = 1
        for d in denoms:
            lcm = lcm * d // gcd(lcm, d)
        
        # Scale to integers
        a_int = {k: int(v * lcm) for k, v in a_frac.items()}
        b_int = {k: int(v * lcm) for k, v in b_frac.items()}
        c_int = {k: int(v * lcm) for k, v in c_frac.items()}
        
        # Verify integer version
        for (i, j, k), val in combined.items():
            expected = a_int.get(i, 0) * b_int.get(j, 0) * c_int.get(k, 0)
            if abs(val * lcm - expected) > 1e-9:
                return None
        
        return (lcm, a_int, b_int, c_int)
    
    # Try merging pairs - multiple passes
    for pass_num in range(10):
        if time.time() > deadline - 10:
            break
        merged = False
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    _, a_new, b_new, c_new = result
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    
    # Try triple merges: check if three terms sum to a rank-1 tensor
    for pass_num in range(5):
        if time.time() > deadline - 5:
            break
        found = False
        for i in range(len(terms)):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 3:
                    break
                for k in range(j + 1, len(terms)):
                    if time.time() > deadline - 3:
                        break
                    # Check if terms[i] + terms[j] + terms[k] is rank-1
                    t1 = term_to_tensor(*terms[i])
                    t2 = term_to_tensor(*terms[j])
                    t3 = term_to_tensor(*terms[k])
                    combined = {}
                    for tt in [t1, t2, t3]:
                        for key, val in tt.items():
                            combined[key] = combined.get(key, 0) + val
                    
                    combined = {key: val for key, val in combined.items() if abs(val) > 1e-12}
                    if not combined:
                        continue
                    
                    u_indices = sorted(set(key[0] for key in combined.keys()))
                    v_indices = sorted(set(key[1] for key in combined.keys()))
                    w_indices = sorted(set(key[2] for key in combined.keys()))
                    
                    if len(u_indices) <= 1 and len(v_indices) <= 1 and len(w_indices) <= 1:
                        if len(u_indices) == 1 and len(v_indices) == 1 and len(w_indices) == 1:
                            val = combined[(u_indices[0], v_indices[0], w_indices[0])]
                            if abs(val) > 1e-12:
                                # Remove all three, add one
                                terms.pop(k)
                                terms.pop(j)
                                terms.pop(i)
                                terms.append(({u_indices[0]: int(val)}, {v_indices[0]: 1}, {w_indices[0]: 1}))
                                found = True
                                break
                        continue
                    
                    # Find reference
                    ref_i, ref_j, ref_k, ref_val = None, None, None, None
                    for i_idx in u_indices:
                        for j_idx in v_indices:
                            for k_idx in w_indices:
                                val = combined.get((i_idx, j_idx, k_idx), 0)
                                if abs(val) > 1e-12:
                                    ref_i, ref_j, ref_k, ref_val = i_idx, j_idx, k_idx, val
                                    break
                            if ref_val is not None:
                                break
                        if ref_val is not None:
                            break
                    
                    if ref_val is None:
                        continue
                    
                    a = {ref_i: 1.0}
                    b = {ref_j: 1.0}
                    c = {ref_k: 1.0}
                    
                    for i_idx in u_indices:
                        if i_idx != ref_i:
                            val = combined.get((i_idx, ref_j, ref_k), 0)
                            a[i_idx] = val / ref_val
                    
                    for j_idx in v_indices:
                        if j_idx != ref_j:
                            val = combined.get((ref_i, j_idx, ref_k), 0)
                            b[j_idx] = val / ref_val
                    
                    for k_idx in w_indices:
                        if k_idx != ref_k:
                            val = combined.get((ref_i, ref_j, k_idx), 0)
                            c[k_idx] = val / ref_val
                    
                    valid = True
                    for (i_idx, j_idx, k_idx), val in combined.items():
                        expected = a.get(i_idx, 0) * b.get(j_idx, 0) * c.get(k_idx, 0) * ref_val
                        if abs(val - expected) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        a_frac = {k: Fraction(v).limit_denominator(1000) for k, v in a.items()}
                        b_frac = {k: Fraction(v).limit_denominator(1000) for k, v in b.items()}
                        c_frac = {k: Fraction(v).limit_denominator(1000) for k, v in c.items()}
                        
                        denoms = [d.numerator for d in a_frac.values()] + \
                                 [d.numerator for d in b_frac.values()] + \
                                 [d.numerator for d in c_frac.values()]
                        lcm = 1
                        for d in denoms:
                            lcm = lcm * d // gcd(lcm, d)
                        
                        a_int = {k: int(v * lcm) for k, v in a_frac.items()}
                        b_int = {k: int(v * lcm) for k, v in b_frac.items()}
                        c_int = {k: int(v * lcm) for k, v in c_frac.items()}
                        
                        # Verify
                        valid2 = True
                        for (i_idx, j_idx, k_idx), val in combined.items():
                            expected = a_int.get(i_idx, 0) * b_int.get(j_idx, 0) * c_int.get(k_idx, 0)
                            if abs(val * lcm - expected) > 1e-9:
                                valid2 = False
                                break
                        
                        if valid2:
                            terms.pop(k)
                            terms.pop(j)
                            terms.pop(i)
                            terms.append((a_int, b_int, c_int))
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            break
    
    rank = len(terms)
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
