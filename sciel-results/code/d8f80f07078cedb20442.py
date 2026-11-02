"""Matrix multiplication 6x6x6 - rank-minimal tensor decomposition search."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_decomposition_fast(Um, Vm, Wm):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        R = Um.shape[1]
        if R == 0:
            return False
        Um_3d = Um.reshape(n, m, R)  # Um[i, j, r]
        Vm_3d = Vm.reshape(m, p, R)  # Vm[j, k, r]
        Wm_3d = Wm.reshape(p, n, R)  # Wm[k, i, r]
        # Use einsum to compute the tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # We need T[i,j,k] == 1 for all i,j,k
        T = np.einsum('ijr,jkr,kir->ijk', Um_3d, Vm_3d, Wm_3d)
        if not np.allclose(T, 1.0, atol=1e-7):
            return False
        return True

    # Build the Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2():
        """Returns U, V, W matrices for 2x2x2 Strassen decomposition (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # P2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        # P3: A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        # P4: A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        # P5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        # P6: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        # P7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    # Build schoolbook 3x3x3 decomposition (rank 27)
    def schoolbook_3x3():
        """Returns U, V, W matrices for 3x3x3 schoolbook decomposition (rank 27)."""
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j, col] = 1.0
                    V[j * 3 + k, col] = 1.0
                    W[k * 3 + i, col] = 1.0
                    col += 1
        return U, V, W

    # Kronecker product of two decompositions
    def kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        """Kronecker product of two matrix multiplication decompositions.
        U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        Returns U: (n1*n2*m1*m2, r1*r2), V: (m1*m2*p1*p2, r1*r2), W: (p1*p2*n1*n2, r1*r2)
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        U = np.zeros((n1 * n2 * m1 * m2, r1 * r2))
        V = np.zeros((m1 * m2 * p1 * p2, r1 * r2))
        W = np.zeros((p1 * p2 * n1 * n2, r1 * r2))
        
        # Precompute non-zero entries for efficiency
        U1_nz = [(i1, j1, U1[i1*m1+j1, r]) for r in range(r1) for i1 in range(n1) for j1 in range(m1) if U1[i1*m1+j1, r] != 0]
        V1_nz = [(j1, k1, V1[j1*p1+k1, r]) for r in range(r1) for j1 in range(m1) for k1 in range(p1) if V1[j1*p1+k1, r] != 0]
        W1_nz = [(k1, i1, W1[k1*n1+i1, r]) for r in range(r1) for k1 in range(p1) for i1 in range(n1) if W1[k1*n1+i1, r] != 0]
        
        U2_nz = [(i2, j2, U2[i2*m2+j2, s]) for s in range(r2) for i2 in range(n2) for j2 in range(m2) if U2[i2*m2+j2, s] != 0]
        V2_nz = [(j2, k2, V2[j2*p2+k2, s]) for s in range(r2) for j2 in range(m2) for k2 in range(p2) if V2[j2*p2+k2, s] != 0]
        W2_nz = [(k2, i2, W2[k2*n2+i2, s]) for s in range(r2) for k2 in range(p2) for i2 in range(n2) if W2[k2*n2+i2, s] != 0]
        
        for r in range(r1):
            for s in range(r2):
                col = r * r2 + s
                # U part
                for i1, j1, u1_val in U1_nz:
                    if u1_val == 0 or U1[i1*m1+j1, r] == 0:
                        continue
                    for i2, j2, u2_val in U2_nz:
                        if u2_val == 0 or U2[i2*m2+j2, s] == 0:
                            continue
                        row = (i1 * n2 + i2) * m1 * m2 + (j1 * m2 + j2)
                        U[row, col] += u1_val * u2_val
                # V part
                for j1, k1, v1_val in V1_nz:
                    if v1_val == 0 or V1[j1*p1+k1, r] == 0:
                        continue
                    for j2, k2, v2_val in V2_nz:
                        if v2_val == 0 or V2[j2*p2+k2, s] == 0:
                            continue
                        row = (j1 * m2 + j2) * p1 * p2 + (k1 * p2 + k2)
                        V[row, col] += v1_val * v2_val
                # W part
                for k1, i1, w1_val in W1_nz:
                    if w1_val == 0 or W1[k1*n1+i1, r] == 0:
                        continue
                    for k2, i2, w2_val in W2_nz:
                        if w2_val == 0 or W2[k2*n2+i2, s] == 0:
                            continue
                        row = (k1 * p2 + k2) * n1 * n2 + (i1 * n2 + i2)
                        W[row, col] += w1_val * w2_val
        return U, V, W

    # Build Strassen 2x2 and Schoolbook 3x3
    U2, V2, W2 = strassen_2x2()
    U3, V3, W3 = schoolbook_3x3()
    
    # Kronecker: 6x6x6 = (2x3) x (2x3) x (2x3)
    # n=6=2*3, m=6=2*3, p=6=2*3
    # U: (n*m, rank) = (36, 7*27=189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U_kron, V_kron, W_kron = kron_decomp(U2, V2, W2, U3, V3, W3, 2, 2, 2, 3, 3, 3)
    
    # Verify the Kronecker decomposition
    if not check_decomposition_fast(U_kron, V_kron, W_kron):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    best_U = U_kron.copy()
    best_V = V_kron.copy()
    best_W = W_kron.copy()
    best_rank = U_kron.shape[1]
    
    # Try to reduce rank by removing columns
    def try_remove_columns(Um, Vm, Wm, col_indices_to_try):
        """Try to remove a subset of columns. Returns (success, new_U, new_V, new_W)."""
        R = Um.shape[1]
        keep = [r for r in range(R) if r not in col_indices_to_try]
        if not keep:
            return False, None, None, None
        U_new = Um[:, keep]
        V_new = Vm[:, keep]
        W_new = Wm[:, keep]
        if check_decomposition_fast(U_new, V_new, W_new):
            return True, U_new, V_new, W_new
        return False, None, None, None
    
    U_cur = U_kron.copy()
    V_cur = V_kron.copy()
    W_cur = W_kron.copy()
    current_rank = best_rank
    
    col_order = list(range(current_rank))
    random.shuffle(col_order)
    
    # Phase 1: Greedy single-column removal
    for idx in col_order:
        if elapsed() > deadline - 20:
            break
        if idx >= current_rank:
            continue
        success, U_new, V_new, W_new = try_remove_columns(U_cur, V_cur, W_cur, {idx})
        if success:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            current_rank -= 1
            if current_rank < best_rank:
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
                best_rank = current_rank
            col_order = [c for c in col_order if c != idx and c < current_rank]
            random.shuffle(col_order)
    
    # Phase 2: Try removing pairs of columns
    if elapsed() < deadline - 30:
        pairs = list(zip(col_order, col_order[1:]))
        random.shuffle(pairs)
        for a, b in pairs[:300]:
            if elapsed() > deadline - 20:
                break
            if a >= current_rank or b >= current_rank:
                continue
            if a == b:
                continue
            success, U_new, V_new, W_new = try_remove_columns(U_cur, V_cur, W_cur, {a, b})
            if success:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                current_rank -= 2
                if current_rank < best_rank:
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
                    best_rank = current_rank
                col_order = [c for c in col_order if c not in {a, b} and c < current_rank]
                random.shuffle(col_order)
    
    # Final verification
    if not check_decomposition_fast(best_U, best_V, best_W):
        # Fallback
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
