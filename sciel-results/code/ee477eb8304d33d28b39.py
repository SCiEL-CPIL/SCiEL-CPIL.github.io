"""Matrix multiplication 4x4x4 tensor decomposition."""
import numpy as np
import time
import random

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Uses Kronecker product of Strassen (rank 7) to get rank 49, then attempts
    iterative column removal to reduce rank. Falls back to rank 49 if removal fails.
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    # Strassen 2x2 decomposition (rank 7)
    # Indices for 2x2: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00+A11)(B00+B11): C00 += M0, C11 += M0
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1

    # M1 = (A10+A11)B00: C10 += M1
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1

    # M2 = A00(B10-B11): C01 += M2
    U2[0,2]=1
    V2[2,2]=1; V2[3,2]=-1
    W2[2,2]=1

    # M3 = A11(B01-B00): C01 -= M3, C11 += M3
    U2[3,3]=1
    V2[1,3]=1; V2[0,3]=-1
    W2[2,3]=1; W2[3,3]=1

    # M4 = (A00+A01)B11: C00 -= M4, C01 += M4
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1

    # M5 = (A10-A00)(B00+B01): C10 -= M5
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[1,5]=-1

    # M6 = (A01-A11)(B10+B11): C10 += M6
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[1,6]=1

    # Verify 2x2 Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(7):
                    actual += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if actual != expected:
                    # Fallback to naive rank 64
                    U = np.zeros((16, 64), dtype=int)
                    V = np.zeros((16, 64), dtype=int)
                    W = np.zeros((16, 64), dtype=int)
                    col = 0
                    for ii in range(4):
                        for jj in range(4):
                            for kk in range(4):
                                U[ii*4+jj, col] = 1
                                V[jj*4+kk, col] = 1
                                W[kk*4+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Kronecker product to get 4x4 decomposition (rank 49)
    U4_kron = np.kron(U2, U2)  # shape (16, 49)
    V4_kron = np.kron(V2, V2)  # shape (16, 49)
    W4_kron = np.kron(W2, W2)  # shape (16, 49)

    # Apply bit-interleaving permutation to reorder rows
    def compute_perm_u(S):
        i = S // 4
        j = S % 4
        i1 = i // 2
        i2 = i % 2
        j1 = j // 2
        j2 = j % 2
        K = (i1 * 2 + j1) * 4 + (i2 * 2 + j2)
        return K

    def compute_perm_v(S):
        j = S // 4
        k = S % 4
        j1 = j // 2
        j2 = j % 2
        k1 = k // 2
        k2 = k % 2
        K = (j1 * 2 + k1) * 4 + (j2 * 2 + k2)
        return K

    def compute_perm_w(S):
        k = S // 4
        i = S % 4
        k1 = k // 2
        k2 = k % 2
        i1 = i // 2
        i2 = i % 2
        K = (k1 * 2 + i1) * 4 + (k2 * 2 + i2)
        return K

    perm_U = [compute_perm_u(S) for S in range(16)]
    perm_V = [compute_perm_v(S) for S in range(16)]
    perm_W = [compute_perm_w(S) for S in range(16)]

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # Verify the 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(49):
                    actual += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if actual != expected:
                    # Fallback to naive rank 64
                    U = np.zeros((16, 64), dtype=int)
                    V = np.zeros((16, 64), dtype=int)
                    W = np.zeros((16, 64), dtype=int)
                    col = 0
                    for ii in range(4):
                        for jj in range(4):
                            for kk in range(4):
                                U[ii*4+jj, col] = 1
                                V[jj*4+kk, col] = 1
                                W[kk*4+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now attempt to reduce rank by removing columns
    # We want to find a subset of columns that still satisfies the tensor equation
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(i,k)
    
    # Strategy: greedy removal. Try removing columns one by one.
    # For each column, check if it's "redundant" given the others.
    # A column is redundant if its contribution can be absorbed by other columns.
    # More simply: try removing each column and check if the tensor is still satisfied.
    
    # Build the tensor constraint matrix.
    # For each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta(i,k)
    # This is a system of 64 equations (some are 0, some are 1).
    # The equations are:
    # For i==k: sum_c U[i,j,c]*V[j,i,c]*W[i,i,c] = 1 for all j
    # For i!=k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 0 for all j
    
    # Let's build a linear system where each column c contributes a vector to the 64-dimensional space.
    # vec_c = [U[i,j,c]*V[j,k,c]*W[k,i,c] for all (i,j,k)]
    # We need sum_c vec_c = target_vec where target_vec[i,j,k] = delta(i,k)
    
    # Precompute target vector
    target = np.zeros(64, dtype=int)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i == k:
                    target[idx] = 1
                idx += 1
    
    # Precompute contribution vectors for each column
    contrib = np.zeros((49, 64), dtype=int)
    for c in range(49):
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    contrib[c, idx] = U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                    idx += 1
    
    # Verify sum of all contrib equals target
    if not np.allclose(contrib.sum(axis=0), target):
        # Should not happen, but fallback
        U = np.zeros((16, 64), dtype=int)
        V = np.zeros((16, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        col = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U[ii*4+jj, col] = 1
                    V[jj*4+kk, col] = 1
                    W[kk*4+ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Greedy removal: try to remove columns one by one
    # We keep a set of active columns. Start with all 49.
    # For each column in some order, try removing it and check if the remaining
    # columns can still sum to target.
    # Since the entries are integers, we can check exactly.
    # But the remaining columns must sum to target exactly.
    # This is a subset sum problem in 64 dimensions, which is hard.
    # Instead, we use a heuristic: a column is removable if its vector is in the
    # span of the other vectors? No, we need exact integer sum.
    
    # Better heuristic: try removing columns in order of "importance" or randomly.
    # For each candidate column to remove, check if target - sum(remaining) == 0.
    # This is O(49 * 64) per check, which is fast.
    
    active = list(range(49))
    # Try removing columns in a random order, multiple passes
    random.seed(42)
    
    for iteration in range(100):  # up to 100 passes
        if time.time() > DEADLINE:
            break
        if not active:
            break
        # Shuffle active columns
        random.shuffle(active)
        removed_this_pass = False
        for c in active[:]:
            if time.time() > DEADLINE:
                break
            # Try removing column c
            new_active = [x for x in active if x != c]
            if not new_active:
                # Can't remove if it's the only one
                continue
            # Check if sum of new_active equals target
            s = contrib[new_active].sum(axis=0)
            if np.array_equal(s, target):
                active = new_active
                removed_this_pass = True
                break  # restart the pass with the new active set
        if not removed_this_pass:
            break  # no more columns can be removed
    
    # Build final matrices from active columns
    if active:
        U_final = U4[:, active]
        V_final = V4[:, active]
        W_final = W4[:, active]
    else:
        # Should not happen, but fallback to full 49
        U_final = U4
        V_final = V4
        W_final = W4
    
    # Final verification
    rank_final = U_final.shape[1]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(rank_final):
                    actual += U_final[i*4+j, c] * V_final[j*4+k, c] * W_final[k*4+i, c]
                if actual != expected:
                    # Fallback to full 49
                    U_final = U4
                    V_final = V4
                    W_final = W4
                    rank_final = 49
                    break
        else:
            continue
        break
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
