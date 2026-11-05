import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        rank = U.shape[1]
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj, col] = 1.0
                    V[jj * p + kk, col] = 1.0
                    W[kk * n + ii, col] = 1.0
                    col += 1
        return U, V, W

    # Construct rank-33 solution
    U = np.zeros((n * m, 33))
    V = np.zeros((m * p, 33))
    W = np.zeros((p * n, 33))
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] - 12 terms (2x6 output)
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: A[:,1:3] * B[1:3,:] using Strassen for each 2x2 block of B
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3 = A00(B10-B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4 = A11(B01-B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1
    
    # Verify rank-33
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U_s, V_s, W_s = schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try to reduce rank by removing columns
    # For each column, check if it can be removed
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 33
    
    # Try removing each single column
    for col_remove in range(33):
        if time.time() > deadline:
            break
        mask = np.ones(33, dtype=bool)
        mask[col_remove] = False
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        if verify_decomposition(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
            best_rank = 32
            break
    
    # If we found rank 32, try to remove more
    if best_rank == 32:
        # Try removing each column from the rank-32 solution
        for col_remove in range(32):
            if time.time() > deadline:
                break
            mask = np.ones(32, dtype=bool)
            mask[col_remove] = False
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            if verify_decomposition(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                best_rank = 31
                break
    
    # If still 33, try a different approach: use a 2x3x2 decomposition with rank < 12
    # For 2x3x2: C[0,0]=a0b0+a1b2+a2b4, C[0,1]=a0b1+a1b3+a2b5
    #            C[1,0]=a3b0+a4b2+a5b4, C[1,1]=a3b1+a4b3+a5b5
    # This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix
    # The tensor has 4 outputs, each with 3 terms = 12 monomials
    # Can we do it in rank 11 or less?
    
    # Try to find a rank-11 decomposition for 2x3x2 by solving a system
    # The tensor T is 2x3x2 with all entries 1
    # We need U (2x3xr), V (3x2xr), W (2x2xr) such that
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # This is a system of 12 equations (one per (i,j,k) pair)
    # with variables U[i,j,r], V[j,k,r], W[k,i,r] for r=1..rank
    
    # For rank 11: 2*3*11 + 3*2*11 + 2*2*11 = 66+66+44 = 176 variables, 12 equations
    # This is highly underdetermined, so solutions likely exist
    # But we need them to be "generic" enough to also work for the full 2x3x6 problem
    
    # Actually, the key insight is that the 3 blocks of 2x3x2 are independent
    # (they use different columns of B and different rows of A partially)
    # So if we can find a rank-r decomposition for 2x3x2, we can use 3r for the full problem
    # But the first part (A[:,0]*B[0,:]) uses A[:,0] and B[0,:] which overlap with the Strassen blocks
    
    # Let me try a completely different decomposition:
    # Split A into A1 (2x1) and A2 (2x2)
    # Split B into B1 (1x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x6, each entry is a single product: 12 terms
    # A2*B2: 2x6, can be split into 3 blocks of 2x2x2, each rank 7 = 21
    # Total: 33
    
    # Alternative: Can we share terms between A1*B1 and A2*B2?
    # A1*B1 gives terms a_i * b_k for i in {0,1}, k in {0..5}
    # A2*B2 gives terms involving a_{i,1}, a_{i,2} and b_{j,k} for j in {1,2}
    # These are disjoint sets of monomials, so no sharing is possible
    # Thus 33 seems to be the minimum for this decomposition strategy
    
    # But maybe a completely different decomposition exists with rank 32?
    # Let's try a random search for rank 32
    
    if best_rank == 33:
        # Try to find rank 32 by random perturbation
        np.random.seed(42)
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Start from rank 33 and try to merge two columns
            # Pick two random columns and see if their sum/difference can be absorbed
            i, j = np.random.choice(33, 2, replace=False)
            
            # Try: new_col = col_i + col_j, and adjust other columns
            # This is complex, so instead try a direct approach
            # Create a rank-32 solution by zeroing out one entry and solving for the rest
            
            # Actually, let's try: take the rank-33 solution and for each pair of columns,
            # check if one is a scalar multiple of the other (unlikely) or if their
            # difference is in the span of the others
            
            # Simpler: just try all single-removal again with a slightly different
            # initial solution
            
            if attempt % 50 == 0:
                # Try a different Strassen variant
                U2 = np.zeros((n * m, 33))
                V2 = np.zeros((m * p, 33))
                W2 = np.zeros((p * n, 33))
                
                col2 = 0
                # Different split: A[:,0:2] * B[0:2,:] + A[:,2] * B[2,:]
                # A[:,0:2] * B[0:2,:]: 2x2 * 2x6 = 2x6, split into 3 blocks of 2x2x2
                for t in range(3):
                    k0 = 2 * t
                    k1 = 2 * t + 1
                    # Strassen for A[:,0:2] * B[0:2, k0:k1]
                    # M1 = (A00+A11)(B00+B11)
                    U2[0, col2] += 1.0; U2[3, col2] += 1.0
                    V2[k0, col2] += 1.0; V2[4 + k1, col2] += 1.0
                    W2[k0 * n + 0, col2] += 1.0; W2[k1 * n + 1, col2] += 1.0
                    col2 += 1
                    # M2 = (A10+A11)B00
                    U2[2, col2] += 1.0; U2[3, col2] += 1.0
                    V2[k0, col2] += 1.0
                    W2[k0 * n + 1, col2] += 1.0; W2[k1 * n + 1, col2] += -1.0
                    col2 += 1
                    # M3 = A00(B10-B11)
                    U2[0, col2] += 1.0
                    V2[4 + k0, col2] += 1.0; V2[4 + k1, col2] += -1.0
                    W2[k1 * n + 0, col2] += 1.0; W2[k1 * n + 1, col2] += 1.0
                    col2 += 1
                    # M4 = A11(B01-B00)
                    U2[3, col2] += 1.0
                    V2[4 + k1, col2] += 1.0; V2[4 + k0, col2] += -1.0
                    W2[k0 * n + 0, col2] += 1.0; W2[k0 * n + 1, col2] += 1.0
                    col2 += 1
                    # M5 = (A00+A01)B11
                    U2[0, col2] += 1.0; U2[1, col2] += 1.0
                    V2[4 + k1, col2] += 1.0
                    W2[k0 * n + 0, col2] += -1.0; W2[k1 * n + 0, col2] += 1.0
                    col2 += 1
                    # M6 = (A10-A00)(B00+B01)
                    U2[2, col2] += 1.0; U2[0, col2] += -1.0
                    V2[k0, col2] += 1.0; V2[k1, col2] += 1.0
                    W2[k1 * n + 1, col2] += 1.0
                    col2 += 1
                    # M7 = (A01-A11)(B10+B11)
                    U2[1, col2] += 1.0; U2[3, col2] += -1.0
                    V2[4 + k0, col2] += 1.0; V2[4 + k1, col2] += 1.0
                    W2[k0 * n + 0, col2] += 1.0
                    col2 += 1
                
                # A[:,2] * B[2,:]: 2x1 * 1x6 = 2x6, 12 terms
                for k in range(p):
                    for i in range(n):
                        U2[i * m + 2, col2] = 1.0
                        V2[2 * p + k, col2] = 1.0
                        W2[k * n + i, col2] = 1.0
                        col2 += 1
                
                if verify_decomposition(U2, V2, W2, n, m, p):
                    # Try to remove a column
                    for col_remove in range(33):
                        mask = np.ones(33, dtype=bool)
                        mask[col_remove] = False
                        U_test = U2[:, mask]
                        V_test = V2[:, mask]
                        W_test = W2[:, mask]
                        if verify_decomposition(U_test, V_test, W_test, n, m, p):
                            best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                            best_rank = 32
                            break
                    if best_rank == 32:
                        break
    
    # Final verification
    if verify_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback
        U_s, V_s, W_s = schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
