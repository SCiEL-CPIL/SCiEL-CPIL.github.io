import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_matmul(U, V, W):
    """
    Verifies if U, V, W decompose 3x3 matmul.
    Convention:
    C[i, k] = sum_r (sum_j A[i, j] U[i*m+j, r]) (sum_j B[j, k] V[j*p+k, r]) W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    R = len(U[0])
    
    # Test with small integer matrices
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.int64)
    B = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int64)
    C_expected = A @ B
    
    C_computed = np.zeros((3, 3), dtype=np.int64)
    
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    A_flat = A.flatten() # i*m+j
    B_flat = B.flatten() # j*p+k
    
    U_proj = U_arr.T @ A_flat
    V_proj = V_arr.T @ B_flat
    
    for r in range(R):
        p_r = U_proj[r] * V_proj[r]
        if p_r != 0:
            w_col = W_arr[:, r]
            w_matrix = w_col.reshape(3, 3) # row k, col i
            C_computed += p_r * w_matrix.T
            
    if not np.allclose(C_computed, C_expected):
        return False
    return True

def _rank24_construct():
    # Rank 24 decomposition
    # Structure:
    # 1. Strassen 2x2 (7 terms) for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. Scalar (1 term) for A[2,2] * B[2,2] -> C[2,2]
    # 3. Cross terms (16 terms)
    #    - 4 terms for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]  (2x1 * 1x2)
    #    - 4 terms for A[2, 0:2] * B[0:2, 2] -> C[0:2, 2]    (1x2 * 2x1)
    #    - 4 terms for A[0:2, 2] * B[2, 2] -> C[0:2, 2]      (2x1 * 1x1)
    #    - 4 terms for A[2, 2] * B[0:2, 2] -> C[0:2, 0:2]... wait, indices.
    
    # Let's be precise with indices.
    # A indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2), 6=(2,0), 7=(2,1), 8=(2,2)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2), 6=(2,0), 7=(2,1), 8=(2,2)
    # W indices (k*n+i): 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2), 6=(2,0), 7=(2,1), 8=(2,2)
    
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
        
    # --- Block 1: Strassen for top-left 2x2 ---
    # A sub: 0,1,3,4. B sub: 0,1,3,4. C sub: 0,1,3,4.
    # M1 = (A00+A11)(B00+B11) -> 0,1,3,4
    # M2 = (A10+A11)B00 -> 1,3
    # M3 = A00(B10-B11) -> 0,1
    # M4 = A11(B01-B00) -> 3,4
    # M5 = (A00+A01)B11 -> 0,4
    # M6 = (A10-A00)(B00+B01) -> 1,3
    # M7 = (A01-A11)(B10+B11) -> 0,1,3,4... wait, standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Let's check standard Strassen indices for 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(h-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5-M7, M3+M5], [M2+M4, M1-M2+M6+M7]]
    
    # Map to 3x3 indices:
    # a=0, b=1, c=3, d=4
    # e=0, f=1, g=3, h=4
    # C00 (idx 0) = M1+M4-M5-M7
    # C01 (idx 1) = M3+M5
    # C10 (idx 3) = M2+M4
    # C11 (idx 4) = M1-M2+M6+M7
    
    # M1: U={0:1, 4:1}, V={0:1, 4:1}, W={0:1, 4:1}
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    
    # M2: U={3:1, 4:1}, V={0:1}, W={3:1}
    add({3:1, 4:1}, {0:1}, {3:1})
    
    # M3: U={0:1}, V={1:1, 3:-1}, W={1:1}
    add({0:1}, {1:1, 3:-1}, {1:1})
    
    # M4: U={4:1}, V={4:1, 1:-1}, W={0:1, 4:1}
    add({4:1}, {4:1, 1:-1}, {0:1, 4:1})
    
    # M5: U={0:1, 1:1}, V={4:1}, W={0:-1, 1:1}
    add({0:1, 1:1}, {4:1}, {0:-1, 1:1})
    
    # M6: U={3:1, 0:-1}, V={0:1, 1:1}, W={4:1}
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    
    # M7: U={1:1, 4:-1}, V={3:1, 4:1}, W={0:-1, 1:-1, 3:-1, 4:1}
    # Wait, C00 has -M7, C11 has +M7.
    # C00 (0): -1
    # C11 (4): +1
    # Does M7 affect others? No.
    add({1:1, 4:-1}, {3:1, 4:1}, {0:-1, 4:1})
    
    # --- Block 2: Scalar for bottom-right ---
    # A8 * B8 -> C8
    add({8:1}, {8:1}, {8:1})
    
    # --- Block 3: Cross terms ---
    # We need to handle:
    # 1. A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    #    A indices: 2, 5. B indices: 6, 7. C indices: 0, 1, 3, 4.
    #    This is (2x1) * (1x2) -> 2x2. Rank 4.
    #    Terms:
    #    T1: A2*B6 -> C0
    #    T2: A2*B7 -> C1
    #    T3: A5*B6 -> C3
    #    T4: A5*B7 -> C4
    #    Wait, is rank 4 sufficient?
    #    (a1, a2) * (b1, b2) -> [[a1b1, a1b2], [a2b1, a2b2]]
    #    Rank 4 is schoolbook for 2x1x2.
    #    Is there a lower rank? 2x1x2 rank is 4.
    #    So we need 4 terms.
    add({2:1}, {6:1}, {0:1})
    add({2:1}, {7:1}, {1:1})
    add({5:1}, {6:1}, {3:1})
    add({5:1}, {7:1}, {4:1})
    
    # 2. A[2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    #    A indices: 6, 7. B indices: 2, 5. C indices: 2, 5.
    #    Wait, C[0:2, 2] means C[0,2] and C[1,2]. Indices 2 and 5.
    #    This is (1x2) * (2x1) -> 1x1? No.
    #    A is 1x2 (row 2, cols 0,1). B is 2x1 (rows 0,1, col 2).
    #    Result is 1x1? No, C[0:2, 2] is 2x1.
    #    Wait.
    #    C[i, 2] = sum_j A[i, j] B[j, 2].
    #    Here we are looking at A[2, j] and B[j, 2].
    #    So C[2, 2] gets contribution from A[2,0]B[0,2] + A[2,1]B[1,2].
    #    But we already handled A8*B8 for C8.
    #    The cross term is A[2, 0:2] * B[0:2, 2] -> C[2, 2]?
    #    No, the task is to decompose the WHOLE tensor.
    #    My block decomposition was:
    #    C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2]B[2, 0:2]
    #    C[0:2, 2]   = A[0:2, 0:2]B[0:2, 2] + A[0:2, 2]B[2, 2]
    #    C[2, 0:2]   = A[2, 0:2]B[0:2, 0:2] + A[2, 2]B[2, 0:2]
    #    C[2, 2]     = A[2, 0:2]B[0:2, 2] + A[2, 2]B[2, 2]
    
    # I already did:
    # - A[0:2, 0:2]B[0:2, 0:2] (7 terms)
    # - A[2, 2]B[2, 2] (1 term)
    # - A[0:2, 2]B[2, 0:2] (4 terms)
    
    # Remaining:
    # 3. A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    #    A 2x2, B 2x1, C 2x1. Rank 4.
    #    A indices: 0,1,3,4. B indices: 2,5. C indices: 2,5.
    #    Terms:
    #    A0*B2 -> C2
    #    A1*B5 -> C5? No.
    #    C[0,2] = A0*B2 + A1*B5
    #    C[1,2] = A3*B2 + A4*B5
    #    Rank 4 terms:
    add({0:1}, {2:1}, {2:1})
    add({1:1}, {5:1}, {5:1})
    add({3:1}, {2:1}, {5:1})
    add({4:1}, {5:1}, {5:1}) # Wait, C[1,2] gets A4*B5.
    # Let's check C[1,2] index 5.
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] = A3*B2 + A4*B5.
    # So term 4 should be {4:1}, {5:1}, {5:1}.
    # And term 3 should be {3:1}, {2:1}, {5:1}.
    # Term 1: {0:1}, {2:1}, {2:1}.
    # Term 2: {1:1}, {5:1}, {5:1}.
    # This covers C[0,2] and C[1,2].
    
    # 4. A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    #    A 2x1, B 1x1, C 2x1. Rank 2.
    #    A indices: 2,5. B index: 8. C indices: 2,5.
    add({2:1}, {8:1}, {2:1})
    add({5:1}, {8:1}, {5:1})
    
    # 5. A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    #    A 1x2, B 2x2, C 1x2. Rank 4.
    #    A indices: 6,7. B indices: 0,1,3,4. C indices: 6,7.
    #    C[2,0] = A6*B0 + A7*B3
    #    C[2,1] = A6*B1 + A7*B4
    add({6:1}, {0:1}, {6:1})
    add({7:1}, {3:1}, {6:1})
    add({6:1}, {1:1}, {7:1})
    add({7:1}, {4:1}, {7:1})
    
    # 6. A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    #    A 1x1, B 1x2, C 1x2. Rank 2.
    #    A index: 8. B indices: 6,7. C indices: 6,7.
    add({8:1}, {6:1}, {6:1})
    add({8:1}, {7:1}, {7:1})
    
    # 7. A[2, 0:2] * B[0:2, 2] -> C[2, 2]
    #    A 1x2, B 2x1, C 1x1. Rank 4? No, dot product rank is 2.
    #    Wait, (1x2)*(2x1) is 1x1.
    #    C[2,2] += A6*B2 + A7*B5.
    #    Rank 2 terms:
    add({6:1}, {2:1}, {8:1})
    add({7:1}, {5:1}, {8:1})
    
    # Total terms: 7 + 1 + 4 + 4 + 2 + 4 + 2 + 2 = 26.
    # This is rank 26.
    # I wanted rank 24.
    # Can we do better on the cross terms?
    # A[0:2, 2] * B[2, 0:2] is 2x1 * 1x2. Rank 4 is minimal.
    # A[0:2, 0:2] * B[0:2, 2] is 2x2 * 2x1. Rank 4 is minimal.
    # A[0:2, 2] * B[2, 2] is 2x1 * 1x1. Rank 2 is minimal.
    # A[2, 0:2] * B[0:2, 0:2] is 1x2 * 2x2. Rank 4 is minimal.
    # A[2, 2] * B[2, 0:2] is 1x1 * 1x2. Rank 2 is minimal.
    # A[2, 0:2] * B[0:2, 2] is 1x2 * 2x1. Rank 2 is minimal.
    
    # Sum of minimal ranks: 7 + 1 + 4 + 4 + 2 + 4 + 2 + 2 = 26.
    # So this block decomposition gives rank 26.
    # Is there a way to share terms between blocks?
    # The tensor is not a direct sum if blocks overlap in A or B indices.
    # Here, the A blocks are disjoint (0:2,0:2), (0:2,2), (2,0:2), (2,2).
    # The B blocks are disjoint.
    # So the tensor IS a direct sum of these 4 sub-tensors?
    # No, C blocks overlap?
    # C[0:2, 0:2] gets contributions from A[0:2, 0:2]B[0:2, 0:2] AND A[0:2, 2]B[2, 0:2].
    # So C blocks are NOT disjoint.
    # But A and B indices ARE disjoint across the 4 terms.
    # If A and B indices are disjoint, can we just sum the ranks?
    # Yes, because the tensor T = sum T_i.
    # If supp(A_i) and supp(B_i) are disjoint for all i != j?
    # Here supp(A_1)={0,1,3,4}, supp(A_2)={2,5}. Disjoint.
    # supp(B_1)={0,1,3,4}, supp(B_2)={6,7}. Disjoint.
    # If A_i and B_i are disjoint from A_j and B_j, then T_i and T_j are "orthogonal" in the factor space.
    # So rank(T) <= sum rank(T_i).
    # So rank 26 is a valid upper bound.
    
    # To get rank 24, we need a smarter decomposition.
    # Maybe combine some cross terms?
    # For now, return rank 26.
    # If we can't get 24, 26 is still better than 27.
    
    # Construct matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

def construct():
    best = _schoolbook()
    
    if _time_left() > 5.0:
        cand = _rank24_construct()
        if cand is not None:
            if _verify_matmul(cand["U"], cand["V"], cand["W"]):
                best = cand
    
    return best
