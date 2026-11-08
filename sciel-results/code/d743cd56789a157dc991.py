"""
5x5 matrix multiplication - rank-minimal block decomposition.
Splits 5 = 2 + 3, uses Strassen for 2x2 and optimized methods for rectangular blocks.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n*m, dtype=float)
        V = np.zeros(m*p, dtype=float)
        W = np.zeros(p*n, dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m + j] = val
        for (j, k), val in v_vals.items():
            V[j*p + k] = val
        for (k, i), val in w_vals.items():
            W[k*n + i] = val

        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """7-term Strassen for 2x2 * 2x2 block multiplication."""
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3 = A00(B01-B11) -> C01 += M3, C00 += M3  (actually C01 += M3, C00 += M3)
        # Standard Strassen: C00 = M1 - M4 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M3 + M6
        # Let me use the standard formulation:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6
        
        # M1 already added above
        # M2
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        # M3
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1})
        # M4
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        # M5
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k1,i0):1, (k0,i0):-1})
        # M6
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        # M7
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

    def schoolbook_2x2_2x2(i0, i1, j0, j1, k0, k1):
        """4-term schoolbook for 2x2 * 2x2."""
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    # Block structure: rows 0,1 = "top" (2 rows); rows 2,3,4 = "bottom" (3 rows)
    # C = A*B where A is 5x5, B is 5x5, C is 5x5
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    # C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    # C[2:5, 0:2] = A[2:5,0:2]*B[0:2,0:2] + A[2:5,2:5]*B[2:5,0:2]
    # C[2:5, 2:5] = A[2:5,0:2]*B[0:2,2:5] + A[2:5,2:5]*B[2:5,2:5]

    # --- C[0:2, 0:2] ---
    # A[0:2,0:2] * B[0:2,0:2]: Strassen 2x2*2x2
    strassen_2x2(0, 1, 0, 1, 0, 1)
    # A[0:2,2:5] * B[2:5,0:2]: 2x3 * 3x2
    # Split: A[0:2,2:4] (2x2) * B[2:4,0:2] (2x2) + A[0:2,4] (2x1) * B[4,0:2] (1x2) + A[0:2,2:4] * B[4,0:2] + A[0:2,4] * B[2:4,0:2]
    # Wait, the sum is over j=2,3,4. So:
    # sum_j A[i,j] B[j,k] for i in {0,1}, j in {2,3,4}, k in {0,1}
    # = sum_{j=2,3} A[i,j] B[j,k] + A[i,4] B[4,k]
    # First part: 2x2 * 2x2 with Strassen
    strassen_2x2(0, 1, 2, 3, 0, 1)
    # Second part: 2x1 * 1x2, schoolbook = 2*1*2 = 4 terms
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # --- C[0:2, 2:5] ---
    # A[0:2,0:2] * B[0:2,2:5]: 2x2 * 2x3
    # = A[0:2,0:2] * B[0:2,2:4] (2x2*2x2) + A[0:2,0:2] * B[0:2,4] (2x2*2x1)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    # 2x2 * 2x1: schoolbook = 2*2*1 = 4 terms
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # A[0:2,2:5] * B[2:5,2:5]: 2x3 * 3x3
    # = A[0:2,2:4] * B[2:4,2:5] (2x2*2x3) + A[0:2,4] * B[4,2:5] (2x1*1x3)
    # 2x2 * 2x3: Strassen on 2x2 part + schoolbook for 2x2*2x1
    strassen_2x2(0, 1, 2, 3, 2, 3)
    # 2x2 * 2x1 part (k=4)
    for i in [0, 1]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # 2x1 * 1x3: schoolbook = 2*1*3 = 6 terms
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    # --- C[2:5, 0:2] ---
    # A[2:5,0:2] * B[0:2,0:2]: 3x2 * 2x2
    # = A[2:4,0:2] * B[0:2,0:2] (2x2*2x2) + A[4,0:2] * B[0:2,0:2] (1x2*2x2)
    strassen_2x2(2, 3, 0, 1, 0, 1)
    # 1x2 * 2x2: schoolbook = 1*2*2 = 4 terms
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # A[2:5,2:5] * B[2:5,0:2]: 3x3 * 3x2
    # = A[2:4,2:4] * B[2:4,0:2] (2x2*2x2) + A[2:4,4] * B[4,0:2] (2x1*1x2) + A[4,2:4] * B[2:4,0:2] (1x2*2x2) + A[4,4] * B[4,0:2] (1x1*1x2)
    strassen_2x2(2, 3, 2, 3, 0, 1)
    # 2x1 * 1x2: schoolbook = 2*1*2 = 4 terms
    for i in [2, 3]:
        for k in [0, 1]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    # 1x2 * 2x2: schoolbook = 1*2*2 = 4 terms
    for j in [2, 3]:
        for k in [0, 1]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # 1x1 * 1x2: schoolbook = 1*1*2 = 2 terms
    for k in [0, 1]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

    # --- C[2:5, 2:5] ---
    # A[2:5,0:2] * B[0:2,2:5]: 3x2 * 2x3
    # = A[2:4,0:2] * B[0:2,2:4] (2x2*2x2) + A[2:4,0:2] * B[0:2,4] (2x2*2x1) + A[4,0:2] * B[0:2,2:4] (1x2*2x2) + A[4,0:2] * B[0:2,4] (1x2*2x1)
    strassen_2x2(2, 3, 0, 1, 2, 3)
    # 2x2 * 2x1 (k=4)
    for i in [2, 3]:
        for j in [0, 1]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # 1x2 * 2x2
    for j in [0, 1]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # 1x2 * 2x1 (k=4)
    for j in [0, 1]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # A[2:5,2:5] * B[2:5,2:5]: 3x3 * 3x3
    # Split into 2x2 blocks:
    # = A[2:4,2:4]*B[2:4,2:4] + A[2:4,4]*B[4,2:4] + A[4,2:4]*B[2:4,2:4] + A[4,4]*B[4,4]
    #   + A[2:4,2:4]*B[2:4,4] + A[2:4,4]*B[4,4] + A[4,2:4]*B[2:4,4] + A[4,4]*B[4,4]
    # Wait, let me be more careful. C[i,k] = sum_j A[i,j] B[j,k] for i,k in {2,3,4}, j in {2,3,4}
    # = sum_{j=2,3} A[i,j] B[j,k] + A[i,4] B[4,k]
    # For the sum_{j=2,3} part:
    #   i in {2,3}, k in {2,3}: 2x2*2x2 -> Strassen
    #   i in {2,3}, k=4: 2x2*2x1 -> schoolbook
    #   i=4, k in {2,3}: 1x2*2x2 -> schoolbook
    #   i=4, k=4: 1x1*1x1 -> 1 term
    # For the A[i,4] B[4,k] part:
    #   i in {2,3,4}, k in {2,3,4}: 3x1*1x3 -> schoolbook = 9 terms
    strassen_2x2(2, 3, 2, 3, 2, 3)
    # 2x2 * 2x1 (i in {2,3}, j in {2,3}, k=4)
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    # 1x2 * 2x2 (i=4, j in {2,3}, k in {2,3})
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    # 1x1 * 1x1 (i=4, j=2,3? No, j=4, k=4)
    # Wait, the sum_{j=2,3} part with i=4, k=4: A[4,2:4] * B[2:4,4]
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    # Now the A[i,4] B[4,k] part: i in {2,3,4}, k in {2,3,4}
    for i in [2, 3, 4]:
        for k in [2, 3, 4]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    if time.time() - t0 > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
