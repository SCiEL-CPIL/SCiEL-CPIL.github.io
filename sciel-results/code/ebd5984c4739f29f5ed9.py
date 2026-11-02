import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # 1. Define Strassen factors for 2x2x2
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # 2. Build 4x4x4 factors using tensor product (Kronecker)
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
                    
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
                    
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            for s in range(7):
                for r in range(7):
                    R = 7 * s + r
                    W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # 3. Build 4x4x8 factors with rank 98
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, :49] = U4.reshape(16, 49)
    U[:, 49:] = U4.reshape(16, 49)
    
    for j in range(4):
        for k in range(4):
            idx = j * 8 + k
            V[idx, :49] = V4[j, k, :]
            idx2 = j * 8 + (4 + k)
            V[idx2, 49:] = V4[j, k, :]
            
    for k in range(4):
        for i in range(4):
            idx = k * 4 + i
            W[idx, :49] = W4[k, i, :]
            idx2 = (4 + k) * 4 + i
            W[idx2, 49:] = W4[k, i, :]

    # 4. Try to reduce rank via column elimination
    # The multiplication tensor T[i,j,k] = delta_{i,k} for A(4x4) * B(4x8)
    # T has rank at most 128 (schoolbook). Strassen gives 98.
    # We can try to find linear dependencies among the rank-1 tensors.
    
    # Build the full multiplication tensor
    # T[i, j, k] = 1 if i == k, else 0 (for the identity part)
    # Actually, for matrix multiplication C = A * B where A is 4x4, B is 4x8:
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{i,k} ... no wait.
    # The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k where j is the contraction index.
    # Actually T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,8).
    
    # Let's verify the current decomposition is correct
    # C[i,k] = sum_r U[i,j? ... 
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the indices need to be consistent. Let me re-examine.
    
    # U has shape (n*m, rank) = (16, 98), indexed by (i*m + j, r)
    # V has shape (m*p, rank) = (32, 98), indexed by (j*p + k, r)
    # W has shape (p*n, rank) = (32, 98), indexed by (k*n + i, r)
    
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with a small test
    if time.time() < deadline - 5:
        # Verify current solution
        U_np = U
        V_np = V
        W_np = W
        
        # Check a few entries
        correct = True
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    # T[i,j,k] = 1 (all entries are 1 in the multiplication tensor)
                    # Actually no - the multiplication tensor for C = A*B is:
                    # T[i,j,k] = 1 if this (i,j,k) triple contributes, which is always true
                    # since C[i,k] = sum_j A[i,j]*B[j,k]
                    # So T[i,j,k] = 1 for all i,j,k
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*4+j, r] * V_np[j*8+k, r] * W_np[k*4+i, r]
                    if abs(val - 1.0) > 1e-10:
                        correct = False
                        break
                if not correct:
                    break
            if not correct:
                break
        
        if not correct:
            # Fallback: return schoolbook with rank 128
            U_school = np.zeros((16, 128))
            V_school = np.zeros((32, 128))
            W_school = np.zeros((32, 128))
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        r = i * 32 + j * 8 + k
                        U_school[i*4+j, r] = 1
                        V_school[j*8+k, r] = 1
                        W_school[k*4+i, r] = 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try rank reduction: check if any columns are linearly dependent
    # For each output entry (i,k), the contribution is:
    # C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = sum_r W[k,i,r] * sum_j U[i,j,r] * V[j,k,r]
    
    # Let's try to find a smaller rank by checking if we can set some W entries to 0
    # and compensate. This is complex, so let's try a simpler approach:
    # Check if the rank-98 solution can be reduced by finding null space vectors.
    
    # Alternative: try a different factorization approach
    # Use the fact that B is 4x8, so we can split B into two 4x4 blocks
    # B = [B1 | B2] where B1, B2 are 4x4
    # C = A * B = [A*B1 | A*B2]
    # We need rank(A*B1) + rank(A*B2) but they share the same A factors.
    
    # The current approach duplicates U for the two halves.
    # U[:, :49] = U4_flat and U[:, 49:] = U4_flat
    # This means the same A factors are used for both halves.
    
    # Can we do better? The rank of multiplying a 4x4 by a 4x8 is at most
    # 2 * rank(4x4x4) = 2*49 = 98 with the naive split.
    # But maybe we can share more structure.
    
    # Let's try: instead of duplicating U, use a single set of 49 factors
    # and have V and W handle the 8 columns differently.
    
    # Actually, the issue is that V and W also need to be split.
    # V[j, k] for k in 0..3 uses V4[j, k], and k in 4..7 uses V4[j, k-4]
    # W[k, i] for k in 0..3 uses W4[k, i], and k in 4..7 uses W4[k-4, i]
    
    # The duplication in U is necessary because the same A factors multiply
    # different B factors. Unless we can find a way to combine them.
    
    # Let's try rank 80 or so by looking for linear dependencies.
    # Build a matrix where each row corresponds to an (i,j,k) entry and each column
    # to a rank-1 tensor, then find the rank.
    
    if time.time() < deadline - 10:
        # Build the constraint matrix
        # For each (i,j,k), we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # This is nonlinear in U,V,W.
        
        # Instead, let's try to verify and return the current solution.
        # The current solution with rank 98 should be correct.
        
        # Let's do a more thorough verification
        U_np = U
        V_np = V
        W_np = W
        
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*4+j, r] * V_np[j*8+k, r] * W_np[k*4+i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-8:
            # Solution is correct, return it
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U_school = np.zeros((16, 128))
    V_school = np.zeros((32, 128))
    W_school = np.zeros((32, 128))
    for i in range(4):
        for j in range(4):
            for k in range(8):
                r = i * 32 + j * 8 + k
                U_school[i*4+j, r] = 1
                V_school[j*8+k, r] = 1
                W_school[k*4+i, r] = 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
