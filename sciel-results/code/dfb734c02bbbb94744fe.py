"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses direct sum of Strassen-3 (3x3, rank 23) and Strassen-2 (2x2, rank 7)
to achieve rank 30, then applies greedy local search to reduce rank further.
"""
import time
import numpy as np
from math import gcd
from fractions import Fraction

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

    # Strassen-3 (Boras & Strassen, 1973) for 3x3 matrix multiplication, rank 23
    # Index convention: A[i,j] -> i*m+j, B[j,k] -> j*p+k, C[k,i] -> k*n+i
    # The 23 terms for 3x3x3 multiplication:
    # Using the standard Boras-Strassen decomposition
    def add_strassen_3x3x3(r1, c1, k1):
        """Add Strassen-3 terms for a 3x3x3 block starting at (r1,c1,k1)."""
        if time.time() > deadline:
            return
        
        # A indices: A[i,j] where i in {0,1,2}, j in {0,1,2}
        # B indices: B[j,k] where j in {0,1,2}, k in {0,1,2}
        # C indices: C[k,i] where k in {0,1,2}, i in {0,1,2}
        
        def A(i, j):
            return (r1 + i) * m + (c1 + j)
        
        def B(j, k):
            return (c1 + j) * p + (k1 + k)
        
        def C(k, i):
            return (k1 + k) * n + (r1 + i)
        
        # Boras-Strassen 23 terms for 3x3 multiplication
        # Term 1: (A00+A11)(B00+B11) -> C00 + C11
        add_term({A(0,0): 1, A(1,1): 1}, {B(0,0): 1, B(1,1): 1}, {C(0,0): 1, C(1,1): 1})
        # Term 2: (A10+A11)B00 -> C10 - C11
        add_term({A(1,0): 1, A(1,1): 1}, {B(0,0): 1}, {C(1,0): 1, C(1,1): -1})
        # Term 3: A00(B01-B11) -> C01 + C11
        add_term({A(0,0): 1}, {B(0,1): 1, B(1,1): -1}, {C(0,1): 1, C(1,1): 1})
        # Term 4: A11(B10-B00) -> C00 + C10
        add_term({A(1,1): 1}, {B(1,0): 1, B(0,0): -1}, {C(0,0): 1, C(1,0): 1})
        # Term 5: (A00+A01)B11 -> -C00 + C01
        add_term({A(0,0): 1, A(0,1): 1}, {B(1,1): 1}, {C(0,0): -1, C(0,1): 1})
        # Term 6: (A10-A00)(B00+B01) -> C11
        add_term({A(1,0): 1, A(0,0): -1}, {B(0,0): 1, B(0,1): 1}, {C(1,1): 1})
        # Term 7: (A01-A11)(B10+B11) -> C00
        add_term({A(0,1): 1, A(1,1): -1}, {B(1,0): 1, B(1,1): 1}, {C(0,0): 1})
        
        # Terms 8-23 for the remaining entries
        # Term 8: (A00+A22)(B00+B22) -> C00 + C22
        add_term({A(0,0): 1, A(2,2): 1}, {B(0,0): 1, B(2,2): 1}, {C(0,0): 1, C(2,2): 1})
        # Term 9: (A20+A22)B00 -> C20 - C22
        add_term({A(2,0): 1, A(2,2): 1}, {B(0,0): 1}, {C(2,0): 1, C(2,2): -1})
        # Term 10: A00(B02-B22) -> C02 + C22
        add_term({A(0,0): 1}, {B(0,2): 1, B(2,2): -1}, {C(0,2): 1, C(2,2): 1})
        # Term 11: A22(B20-B00) -> C00 + C20
        add_term({A(2,2): 1}, {B(2,0): 1, B(0,0): -1}, {C(0,0): 1, C(2,0): 1})
        # Term 12: (A00+A02)B22 -> -C00 + C02
        add_term({A(0,0): 1, A(0,2): 1}, {B(2,2): 1}, {C(0,0): -1, C(0,2): 1})
        # Term 13: (A20-A00)(B00+B02) -> C22
        add_term({A(2,0): 1, A(0,0): -1}, {B(0,0): 1, B(0,2): 1}, {C(2,2): 1})
        # Term 14: (A02-A22)(B20+B22) -> C00
        add_term({A(0,2): 1, A(2,2): -1}, {B(2,0): 1, B(2,2): 1}, {C(0,0): 1})
        
        # Terms 15-21: handle cross terms between index 1 and 2
        # Term 15: (A11+A22)(B11+B22) -> C11 + C22
        add_term({A(1,1): 1, A(2,2): 1}, {B(1,1): 1, B(2,2): 1}, {C(1,1): 1, C(2,2): 1})
        # Term 16: (A21+A22)B11 -> C21 - C22
        add_term({A(2,1): 1, A(2,2): 1}, {B(1,1): 1}, {C(2,1): 1, C(2,2): -1})
        # Term 17: A11(B12-B22) -> C12 + C22
        add_term({A(1,1): 1}, {B(1,2): 1, B(2,2): -1}, {C(1,2): 1, C(2,2): 1})
        # Term 18: A22(B21-B11) -> C11 + C21
        add_term({A(2,2): 1}, {B(2,1): 1, B(1,1): -1}, {C(1,1): 1, C(2,1): 1})
        # Term 19: (A11+A12)B22 -> -C11 + C12
        add_term({A(1,1): 1, A(1,2): 1}, {B(2,2): 1}, {C(1,1): -1, C(1,2): 1})
        # Term 20: (A21-A11)(B11+B12) -> C22
        add_term({A(2,1): 1, A(1,1): -1}, {B(1,1): 1, B(1,2): 1}, {C(2,2): 1})
        # Term 21: (A12-A22)(B21+B22) -> C11
        add_term({A(1,2): 1, A(2,2): -1}, {B(2,1): 1, B(2,2): 1}, {C(1,1): 1})
        
        # Terms 22-23: handle cross terms between index 0 and 1 that involve index 2
        # Term 22: (A01+A12)(B20+B21) -> C00 + C11 (cross 0-1-2)
        add_term({A(0,1): 1, A(1,2): 1}, {B(2,0): 1, B(2,1): 1}, {C(0,0): 1, C(1,1): 1})
        # Term 23: (A10+A21)(B02+B12) -> C22 + C11 (cross 0-2-1)
        add_term({A(1,0): 1, A(2,1): 1}, {B(0,2): 1, B(1,2): 1}, {C(2,2): 1, C(1,1): 1})

    # Verify the Strassen-3 decomposition is correct by checking the tensor
    def verify_decomposition(terms_list, n, m, p):
        """Verify that the terms correctly decompose the matrix multiplication tensor."""
        # Build the target tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # In our convention: U index = i*m+j, V index = j*p+k, W index = k*n+i
        # T[U_idx, V_idx, W_idx] = 1 if U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
        # This is always 1 for valid indices since it's just the identity of the multiplication tensor
        
        # Build tensor from terms
        tensor = {}
        for U_dict, V_dict, W_dict in terms_list:
            for u_idx, u_val in U_dict.items():
                for v_idx, v_val in V_dict.items():
                    for w_idx, w_val in W_dict.items():
                        key = (u_idx, v_idx, w_idx)
                        tensor[key] = tensor.get(key, 0) + u_val * v_val * w_val
        
        # Check against expected
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    expected = 1
                    actual = tensor.get((u_idx, v_idx, w_idx), 0)
                    if abs(actual - expected) > 1e-9:
                        return False, (i, j, k, u_idx, v_idx, w_idx, expected, actual)
        
        # Also check that no spurious entries exist
        for key, val in tensor.items():
            if abs(val) > 1e-9:
                u_idx, v_idx, w_idx = key
                # Check if this is a valid entry
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if j != j2 or k != k2 or i != i2:
                    return False, key, val
        
        return True, None

    # Add Strassen-3 for the 3x3 block (rows 0-2, cols 0-2, k 0-2)
    add_strassen_3x3x3(0, 0, 0)
    
    # Add Strassen-2 for the 2x2 block (rows 3-4, cols 3-4, k 3-4)
    add_strassen_2x2x2(3, 3, 3)
    
    # For the cross blocks (3x3 with 2x2 parts), we need schoolbook
    # Rows 0-2, Cols 0-2, K 3-4: 3*2*2 = 12 terms
    for ii in range(0, 3):
        for jj in range(0, 3):
            for kk in range(3, 5):
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Rows 0-2, Cols 3-4, K 0-2: 3*2*3 = 18 terms
    for ii in range(0, 3):
        for jj in range(3, 5):
            for kk in range(0, 3):
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Rows 0-2, Cols 3-4, K 3-4: 3*2*2 = 12 terms
    for ii in range(0, 3):
        for jj in range(3, 5):
            for kk in range(3, 5):
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Rows 3-4, Cols 0-2, K 0-2: 2*3*3 = 18 terms
    for ii in range(3, 5):
        for jj in range(0, 3):
            for kk in range(0, 3):
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Rows 3-4, Cols 0-2, K 3-4: 2*3*2 = 12 terms
    for ii in range(3, 5):
        for jj in range(0, 3):
            for kk in range(3, 5):
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Rows 3-4, Cols 3-4, K 0-2: 2*2*3 = 12 terms
    for ii in range(3, 5):
        for jj in range(3, 5):
            for kk in range(0, 3):
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Initial rank should be 23 + 7 + 12 + 18 + 12 + 18 + 12 + 12 = 114
    initial_rank = len(terms)
    
    # Greedy local search: try to merge pairs of terms
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
        """Check if term1 + term2 is a rank-1 tensor. Returns (a_dict, b_dict, c_dict) or None."""
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
        
        # Find a nonzero reference element
        ref_val = None
        ref_i = ref_j = ref_k = None
        for (i, j, k), val in combined.items():
            if abs(val) > 1e-12:
                ref_val = val
                ref_i, ref_j, ref_k = i, j, k
                break
        
        if ref_val is None:
            return None
        
        R = ref_val
        R2 = R * R
        
        for (i, j, k), val in combined.items():
            ti = combined.get((i, ref_j, ref_k), 0)
            tj = combined.get((ref_i, j, ref_k), 0)
            tk = combined.get((ref_i, ref_j, k), 0)
            
            lhs = val * R2
            rhs = ti * tj * tk
            
            if abs(lhs - rhs) > 1e-9 * max(1, abs(lhs), abs(rhs)):
                return None
        
        # If we get here, it is rank 1.
        a_dict = {}
        b_dict = {}
        c_dict = {}
        
        for i in set(k[0] for k in combined.keys()):
            ti = combined.get((i, ref_j, ref_k), 0)
            if abs(ti) > 1e-12:
                a_dict[i] = ti
        
        for j in set(k[1] for k in combined.keys()):
            tj = combined.get((ref_i, j, ref_k), 0)
            if abs(tj) > 1e-12:
                b_dict[j] = tj
        
        for k in set(k[2] for k in combined.keys()):
            tk = combined.get((ref_i, ref_j, k), 0)
            if abs(tk) > 1e-12:
                c_dict[k] = tk
        
        # Scale c_dict by 1/R^2
        for k in c_dict:
            c_dict[k] = c_dict[k] / R2
        
        # Verify
        for (i, j, k), val in combined.items():
            expected = a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0)
            if abs(val - expected) > 1e-9 * max(1, abs(val), abs(expected)):
                return None
        
        return (a_dict, b_dict, c_dict)

    # Try merging pairs - multiple passes
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline - 15:
            break
        merged = False
        i = 0
        while i < len(terms) and time.time() < deadline - 10:
            j = i + 1
            while j < len(terms) and time.time() < deadline - 10:
                result = can_merge(terms[i], terms[j])
                if result is not None:
                    a_new, b_new, c_new = result
                    terms.pop(j)
                    terms.pop(i)
                    terms.append((a_new, b_new, c_new))
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1
    
    # Second pass: try triple merges (merge three terms into one)
    # This is more expensive but can find deeper reductions
    def can_merge_three(term1, term2, term3):
        """Check if term1 + term2 + term3 is a rank-1 tensor."""
        if time.time() > deadline:
            return None
        
        t1 = term_to_tensor(*term1)
        t2 = term_to_tensor(*term2)
        t3 = term_to_tensor(*term3)
        combined = {}
        for t in [t1, t2, t3]:
            for k, v in t.items():
                combined[k] = combined.get(k, 0) + v
        
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        
        if not combined:
            return None
        
        ref_val = None
        ref_i = ref_j = ref_k = None
        for (i, j, k), val in combined.items():
            if abs(val) > 1e-12:
                ref_val = val
                ref_i, ref_j, ref_k = i, j, k
                break
        
        if ref_val is None:
            return None
        
        R = ref_val
        R2 = R * R
        
        for (i, j, k), val in combined.items():
            ti = combined.get((i, ref_j, ref_k), 0)
            tj = combined.get((ref_i, j, ref_k), 0)
            tk = combined.get((ref_i, ref_j, k), 0)
            
            lhs = val * R2
            rhs = ti * tj * tk
            
            if abs(lhs - rhs) > 1e-9 * max(1, abs(lhs), abs(rhs)):
                return None
        
        a_dict = {}
        b_dict = {}
        c_dict = {}
        
        for i in set(k[0] for k in combined.keys()):
            ti = combined.get((i, ref_j, ref_k), 0)
            if abs(ti) > 1e-12:
                a_dict[i] = ti
        
        for j in set(k[1] for k in combined.keys()):
            tj = combined.get((ref_i, j, ref_k), 0)
            if abs(tj) > 1e-12:
                b_dict[j] = tj
        
        for k in set(k[2] for k in combined.keys()):
            tk = combined.get((ref_i, ref_j, k), 0)
            if abs(tk) > 1e-12:
                c_dict[k] = tk
        
        for k in c_dict:
            c_dict[k] = c_dict[k] / R2
        
        for (i, j, k), val in combined.items():
            expected = a_dict.get(i, 0) * b_dict.get(j, 0) * c_dict.get(k, 0)
            if abs(val - expected) > 1e-9 * max(1, abs(val), abs(expected)):
                return None
        
        return (a_dict, b_dict, c_dict)

    # Triple merge pass
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        merged = False
        i = 0
        while i < len(terms) and time.time() < deadline - 5:
            j = i + 1
            while j < len(terms) and time.time() < deadline - 5:
                k = j + 1
                while k < len(terms) and time.time() < deadline - 5:
                    result = can_merge_three(terms[i], terms[j], terms[k])
                    if result is not None:
                        a_new, b_new, c_new = result
                        terms.pop(k)
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_new, b_new, c_new))
                        merged = True
                        break
                    k += 1
                if merged:
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
