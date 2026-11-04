"""
5x5 matrix multiplication tensor decomposition.
Structural change: start from a finer block decomposition with multiple
Strassen 2x2x2 blocks, then aggressively merge pairs and triples into
rank-1 terms.
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
    deadline = start_time + 130.0  # Safety margin before 150s kill

    terms = []  # List of (U_dict, V_dict, W_dict)

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

    # Partition: [2, 1, 2] to create multiple 2x2x2 Strassen blocks
    blocks = [(0, 2), (2, 3), (3, 5)]

    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1

                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2x2(r1, c1, k1)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    def term_to_sparse(U_dict, V_dict, W_dict):
        """Convert term to dict of (u,v,w) -> value."""
        tensor = {}
        for u_idx, u_val in U_dict.items():
            for v_idx, v_val in V_dict.items():
                for w_idx, w_val in W_dict.items():
                    key = (u_idx, v_idx, w_idx)
                    tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        return {k: v for k, v in tensor.items() if abs(v) > 1e-12}

    def is_rank1_and_extract(combined):
        """
        Check if combined tensor (dict of (i,j,k)->val) is rank-1.
        Returns (a_dict, b_dict, c_dict) of Fraction values if rank-1, else None.
        """
        if not combined:
            return None

        u_indices = sorted(set(k[0] for k in combined.keys()))
        v_indices = sorted(set(k[1] for k in combined.keys()))
        w_indices = sorted(set(k[2] for k in combined.keys()))

        if not u_indices or not v_indices or not w_indices:
            return None

        # Find a reference element
        ref_i, ref_j, ref_k, ref_val = None, None, None, None
        for i in u_indices:
            for j in v_indices:
                for k in w_indices:
                    val = combined.get((i, j, k), 0)
                    if abs(val) > 1e-12:
                        ref_i, ref_j, ref_k, ref_val = i, j, k, val
                        break
                if ref_val is not None:
                    break
            if ref_val is not None:
                break

        if ref_val is None:
            return None

        # Extract factors
        a = {ref_i: 1.0}
        b = {ref_j: 1.0}
        c = {ref_k: 1.0}

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

        # Verify rank-1: tensor[i,j,k] == a[i]*b[j]*c[k]*ref_val
        for (i, j, k), val in combined.items():
            expected = a.get(i, 0) * b.get(j, 0) * c.get(k, 0) * ref_val
            if abs(val - expected) > 1e-9:
                return None

        # Convert to Fractions
        a_frac = {k: Fraction(v).limit_denominator(10000) for k, v in a.items()}
        b_frac = {k: Fraction(v).limit_denominator(10000) for k, v in b.items()}
        c_frac = {k: Fraction(v).limit_denominator(10000) for k, v in c.items()}

        return (a_frac, b_frac, c_frac)

    def scale_to_ints(a_frac, b_frac, c_frac):
        """Scale fraction dicts to integers with common LCM multiplier."""
        denoms = []
        for d in list(a_frac.values()) + list(b_frac.values()) + list(c_frac.values()):
            denoms.append(d.denominator)
        if not denoms:
            return 1, {}, {}, {}
        lcm = 1
        for d in denoms:
            lcm = lcm * d // gcd(lcm, d)
        a_int = {k: int(v * lcm) for k, v in a_frac.items()}
        b_int = {k: int(v * lcm) for k, v in b_frac.items()}
        c_int = {k: int(v * lcm) for k, v in c_frac.items()}
        return lcm, a_int, b_int, c_int

    # Merge passes: try to combine pairs of terms into a single rank-1 term
    max_passes = 30
    for pass_num in range(max_passes):
        if time.time() > deadline - 15:
            break
        merged_any = False
        changed = True
        while changed:
            if time.time() > deadline - 10:
                break
            changed = False
            for i in range(len(terms)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(terms)):
                    if time.time() > deadline - 10:
                        break
                    t1 = term_to_sparse(*terms[i])
                    t2 = term_to_sparse(*terms[j])
                    combined = {}
                    for k, v in t1.items():
                        combined[k] = v
                    for k, v in t2.items():
                        combined[k] = combined.get(k, 0) + v
                    combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}

                    if not combined:
                        # They cancel out entirely - remove both
                        terms.pop(j)
                        terms.pop(i)
                        merged_any = True
                        changed = True
                        break
                    continue

                    result = is_rank1_and_extract(combined)
                    if result is not None:
                        a_frac, b_frac, c_frac = result
                        _, a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_int, b_int, c_int))
                        merged_any = True
                        changed = True
                        break
                if changed:
                    break
        if not merged_any:
            break

    # Try triple merges
    for pass_num in range(15):
        if time.time() > deadline - 10:
            break
        found = False
        for i in range(len(terms)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 5:
                    break
                for k in range(j + 1, len(terms)):
                    if time.time() > deadline - 5:
                        break
                    t1 = term_to_sparse(*terms[i])
                    t2 = term_to_sparse(*terms[j])
                    t3 = term_to_sparse(*terms[k])
                    combined = {}
                    for tt in [t1, t2, t3]:
                        for key, val in tt.items():
                            combined[key] = combined.get(key, 0) + val
                    combined = {key: val for key, val in combined.items() if abs(val) > 1e-12}

                    if not combined:
                        terms.pop(k)
                        terms.pop(j)
                        terms.pop(i)
                        found = True
                        break
                    continue

                    result = is_rank1_and_extract(combined)
                    if result is not None:
                        a_frac, b_frac, c_frac = result
                        _, a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
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
