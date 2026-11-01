"""Matrix multiplication 2x4x8 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Tensor T has shape (n*m, m*p, p*n) = (8, 32, 16)
    # T[i*m+j, j*p+k, k*n+i] = 1 (schoolbook pattern)
    # Output factor W uses trace convention W[k*n+i, r]
    
    # Build the multiplication tensor
    # For C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent (same j in a and b, same k in b and c)
    
    nm = n * m      # 8
    mp = m * p      # 32
    pn = p * n      # 16
    
    # Build tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with schoolbook decomposition (rank 64)
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    # Now try to reduce rank using Strassen-like combinations
    # For 2x2x2 blocks, Strassen uses 7 instead of 8
    # Our tensor is 2x4x8. We can view it as 8 independent 2x2x2 blocks?
    # Actually: A is 2x4, B is 4x8, C is 2x8
    # Split A into 2 blocks of 2x2: A = [A1, A2] where A1, A2 are 2x2
    # Split B into 4 blocks of 2x2: B = [B1, B2, B3, B4]
    # C = A1*B1 + A2*B3 (first 2 cols) + A1*B2 + A2*B4 (next 2 cols) etc.
    # Actually C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # 
    # Let me think of it differently. We have 4 "middle" indices j=0,1,2,3.
    # For each pair (i,k), C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # 
    # Strassen for 2x2x2: instead of 8 mults, use 7.
    # We can group the 4 j's into 2 pairs: (j=0,j=1) and (j=2,j=3)
    # For each pair, we have a 2x2x2 subproblem (i in {0,1}, j in pair, k in {0..7})
    # Wait, k has 8 values, not 2.
    
    # Better: view as 8 independent 2x2x2 problems by fixing k
    # For fixed k, we have C[:,k] = A * B[:,k] which is 2x1 = 2x4 * 4x1
    # That's not 2x2x2.
    
    # Alternative: The knowledge says "8 independent 2x2x2x2 sub-problems" giving 56.
    # Let me just use the known Strassen-56 construction directly.
    
    # Actually, let me try a different structural approach:
    # Use the fact that M_{2,4,8} can be embedded in M_{4,4,8} or similar.
    # 
    # For now, let me try to reduce from 64 using linear algebra.
    # The key idea: if one rank-1 term is in the span of others, we can remove it.
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by checking linear dependencies
    # A term c is redundant if U[:,c] is in span of other U columns,
    # V[:,c] is in span of other V columns, AND W[:,c] is in span of other W columns.
    # But that's not quite right - the tensor constraint is more subtle.
    
    # Better approach: use the "rank reduction" via the following:
    # If T = sum_{r=1}^{R} u_r v_r w_r (outer products), and if there's a linear
    # dependency among the u_r's, say u_s = sum_{r != s} alpha_r u_r, then we can
    # potentially reduce rank if the corresponding v and w also have dependencies.
    
    # For a cleaner approach, let me use the known result:
    # M_{2,4,8} has rank at most 56 (Strassen). Let me construct it.
    
    # Strassen for 2x4x8:
    # Think of A as 2x4, split into left 2x2 (A_L) and right 2x2 (A_R)
    # B is 4x8, split into top 2x8 (B_T) and bottom 2x8 (B_B)
    # C = A_L * B_T + A_R * B_B (each is 2x2 * 2x8 = 2x8)
    # 
    # For A_L * B_T: this is 2x2 * 2x8. Split B_T into 4 blocks of 2x2.
    # A_L * B_T = A_L * [B_T1, B_T2, B_T3, B_T4] = [A_L*B_T1, A_L*B_T2, A_L*B_T3, A_L*B_T4]
    # Each A_L*B_Ti is 2x2*2x2, rank 7 by Strassen.
    # So A_L * B_T uses 4*7 = 28 multiplications.
    # Similarly A_R * B_B uses 28.
    # Total: 56.
    
    # Let me construct this explicitly.
    
    def strassen_2x2x2():
        """Return the 7 Strassen multiplication terms for 2x2*2x2.
        A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
        C = A*B
        M1 = a11*(b11+b22)
        M2 = (a11+a12)*b22
        M3 = (a21+a22)*b11
        M4 = a22*(b11-b21)
        M5 = (a11+a21)*(b12+b22)
        M6 = (a12-a21)*(b11+b21)
        M7 = (a11-a22)*(b12+b22)
        C11 = M1+M4-M5+M7
        C12 = M2+M5
        C21 = M3+M6
        C22 = M1-M2+M3+M4
        """
        # Return U, V, W for 2x2x2 Strassen
        # A indices: (0,0),(0,1),(1,0),(1,1) -> 0,1,2,3
        # B indices: (0,0),(0,1),(1,0),(1,1) -> 0,1,2,3
        # C indices: (0,0),(0,1),(1,0),(1,1) -> 0,1,2,3
        U_s = np.zeros((4, 7), dtype=np.float64)
        V_s = np.zeros((4, 7), dtype=np.float64)
        W_s = np.zeros((4, 7), dtype=np.float64)
        
        # M1 = a11*(b11+b22): U[0]=1, V[0]=1,V[3]=1, W: C11+=1, C22+=1
        U_s[0,0] = 1; V_s[0,0] = 1; V_s[3,0] = 1; W_s[0,0] = 1; W_s[3,0] = 1
        # M2 = (a11+a12)*b22: U[0]=1,U[1]=1, V[3]=1, W: C12+=1, C22-=1
        U_s[0,1] = 1; U_s[1,1] = 1; V_s[3,1] = 1; W_s[1,1] = 1; W_s[3,1] = -1
        # M3 = (a21+a22)*b11: U[2]=1,U[3]=1, V[0]=1, W: C21+=1, C22+=1
        U_s[2,2] = 1; U_s[3,2] = 1; V_s[0,2] = 1; W_s[2,2] = 1; W_s[3,2] = 1
        # M4 = a22*(b11-b21): U[3]=1, V[0]=1,V[2]=-1, W: C11+=1, C22+=1
        U_s[3,3] = 1; V_s[0,3] = 1; V_s[2,3] = -1; W_s[0,3] = 1; W_s[3,3] = 1
        # M5 = (a11+a21)*(b12+b22): U[0]=1,U[2]=1, V[1]=1,V[3]=1, W: C11-=1, C12+=1
        U_s[0,4] = 1; U_s[2,4] = 1; V_s[1,4] = 1; V_s[3,4] = 1; W_s[0,4] = -1; W_s[1,4] = 1
        # M6 = (a12-a21)*(b11+b21): U[1]=1,U[2]=-1, V[0]=1,V[2]=1, W: C21+=1
        U_s[1,5] = 1; U_s[2,5] = -1; V_s[0,5] = 1; V_s[2,5] = 1; W_s[2,5] = 1
        # M7 = (a11-a22)*(b12+b22): U[0]=1,U[3]=-1, V[1]=1,V[3]=1, W: C11+=1, C22+=1
        U_s[0,6] = 1; U_s[3,6] = -1; V_s[1,6] = 1; V_s[3,6] = 1; W_s[0,6] = 1; W_s[3,6] = 1
        
        return U_s, V_s, W_s
    
    # Now build the 2x4x8 decomposition using 8 Strassen 2x2x2 blocks
    # A is 2x4: indices (i,j) with i in {0,1}, j in {0,1,2,3}
    # B is 4x8: indices (j,k) with j in {0,1,2,3}, k in {0..7}
    # C is 2x8: indices (i,k) with i in {0,1}, k in {0..7}, W index = k*n+i = 2*k+i
    
    # Group j into pairs: (0,1) and (2,3)
    # Group k into pairs: (0,1), (2,3), (4,5), (6,7)
    # For each j-pair and k-pair, we have a 2x2x2 block
    
    U_full = np.zeros((nm, 56), dtype=np.float64)
    V_full = np.zeros((mp, 56), dtype=np.float64)
    W_full = np.zeros((pn, 56), dtype=np.float64)
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    col = 0
    for j_pair in range(2):  # j in {0,1} or {2,3}
        j0 = j_pair * 2
        j1 = j_pair * 2 + 1
        for k_pair in range(4):  # k in {0,1}, {2,3}, {4,5}, {6,7}
            k0 = k_pair * 2
            k1 = k_pair * 2 + 1
            for r in range(7):
                # U: A[i, j0] and A[i, j1] for i=0,1
                # A index: i*m + j = i*4 + j
                for i in range(2):
                    # j0 corresponds to local index 0, j1 to local index 1
                    U_full[i*4 + j0, col] = U_s[i*2 + 0, r]  # local j=0
                    U_full[i*4 + j1, col] = U_s[i*2 + 1, r]  # local j=1
                # V: B[j, k0] and B[j, k1] for j=j0,j1
                # B index: j*p + k = j*8 + k
                for j in range(2):
                    jj = j_pair*2 + j  # actual j
                    # k0 corresponds to local k=0, k1 to local k=1
                    V_full[jj*8 + k0, col] = V_s[j*2 + 0, r]  # local k=0
                    V_full[jj*8 + k1, col] = V_s[j*2 + 1, r]  # local k=1
                # W: C[i, k0] and C[i, k1] for i=0,1
                # W index: k*n + i = 2*k + i
                for i in range(2):
                    W_full[k0*2 + i, col] = W_s[i*2 + 0, r]  # local k=0
                    W_full[k1*2 + i, col] = W_s[i*2 + 1, r]  # local k=1
                col += 1
    
    # Verify: check that sum_r U[:,r] * V[:,r] * W[:,r] == T
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Let's verify
    T_recon = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    error = np.max(np.abs(T_recon - T))
    
    if error > 1e-10:
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce further using linear algebra
    # Check if any column is redundant
    
    def try_reduce(U_mat, V_mat, W_mat, T_tensor, max_iter=100):
        """Try to reduce rank by removing redundant columns."""
        R = U_mat.shape[1]
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            reduced = False
            for c in range(R):
                if time.time() > deadline:
                    break
                # Check if column c is redundant
                # Try to remove it: see if the remaining R-1 columns can still represent T
                # This is expensive, so use a heuristic:
                # If U[:,c] is in span of other U columns, AND V[:,c] in span of other V,
                # AND W[:,c] in span of other W, it might be removable.
                
                if R <= 1:
                    break
                    
                # Build matrices without column c
                mask = [i for i in range(R) if i != c]
                U_rest = U_cur[:, mask]
                V_rest = V_cur[:, mask]
                W_rest = W_cur[:, mask]
                
                # Check if U[:,c] is in column space of U_rest
                # Use SVD
                try:
                    U_res = U_rest - U_rest @ np.linalg.lstsq(U_rest, U_cur[:, c], rcond=None)[0]
                    if np.linalg.norm(U_res) < 1e-8:
                        # U column is dependent, check others
                        V_res = V_rest - V_rest @ np.linalg.lstsq(V_rest, V_cur[:, c], rcond=None)[0]
                        if np.linalg.norm(V_res) < 1e-8:
                            W_res = W_rest - W_rest @ np.linalg.lstsq(W_rest, W_cur[:, c], rcond=None)[0]
                            if np.linalg.norm(W_res) < 1e-8:
                                # All three are dependent, try to remove
                                # Verify by reconstruction
                                T_test = np.einsum('ar,br,cr->abc', U_rest, V_rest, W_rest)
                                if np.max(np.abs(T_test - T_tensor)) < 1e-8:
                                    U_cur = U_rest
                                    V_cur = V_rest
                                    W_cur = W_rest
                                    R -= 1
                                    reduced = True
                                    break
                except:
                    pass
            if not reduced:
                break
        
        return U_cur, V_cur, W_cur
    
    U_reduced, V_reduced, W_reduced = try_reduce(U_full, V_full, W_full, T)
    
    # Final verification
    T_final = np.einsum('ar,br,cr->abc', U_reduced, V_reduced, W_reduced)
    if np.max(np.abs(T_final - T)) > 1e-8:
        # Use the verified 56-rank version
        U_out, V_out, W_out = U_full, V_full, W_full
    else:
        U_out, V_out, W_out = U_reduced, V_reduced, W_reduced
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
