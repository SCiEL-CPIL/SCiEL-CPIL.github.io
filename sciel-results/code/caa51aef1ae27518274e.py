"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Strassen-style recursive decomposition
    # Split middle dimension m=4 into two blocks of size 2
    # This creates a block matrix multiplication problem
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # Result C = A1*B1 + A2*B2
    # 
    # For each 2x2 x 2x7 block multiplication, we can use
    # Strassen's algorithm for the 2x2 part combined with schoolbook
    # for the 2x7 part.
    #
    # Actually, let's think more carefully. The tensor for 2x4 x 4x7 is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split j into j1 (j=0,1) and j2 (j=2,3)
    # C[i,k] = sum_{j1=0}^1 A[i,j1] B[j1,k] + sum_{j2=0}^1 A[i,j2+2] B[j2+2,k]
    #
    # The first term is a 2x2 x 2x7 multiplication
    # The second term is also a 2x2 x 2x7 multiplication
    #
    # For 2x2 x 2x7: C[i,k] = sum_{j=0}^1 A[i,j] B[j,k]
    # This can be decomposed using Strassen for the 2x2 structure.
    # Strassen gives rank 7 for 2x2x2. For 2x2x2x7, we need to
    # handle the 7 output columns.
    #
    # Let's use a different approach: direct Strassen-like decomposition
    # for the full 2x4x4x7 problem.
    
    def strassen_decomposition():
        """
        Strassen-style decomposition for 2x4 x 4x7 matrix multiplication.
        
        Split the middle index j (size 4) into two blocks of size 2.
        A = [A_left | A_right] where A_left, A_right are 2x2
        B = [B_top; B_bottom] where B_top, B_bottom are 2x7
        
        C = A_left @ B_top + A_right @ B_bottom
        
        For each 2x2 @ 2x7 product, use Strassen decomposition:
        M1 = (A11 + A22) @ (B11 + B22)  -- but B is 2x7, so we need to be careful
        """
        # Let's think of it differently.
        # For 2x2 @ 2x7:
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # 
        # We can use the 7-term Strassen decomposition for each "column group"
        # but since B has 7 columns, we need to apply Strassen column-wise
        # or find a unified decomposition.
        #
        # Actually, the tensor for 2x2 @ 2x7 has rank at most 2*2*7 = 28 (schoolbook)
        # but potentially lower.
        #
        # Let's try a direct approach: use the fact that 2x2 @ 2x7 can be seen as
        # 7 independent 2x2 @ 2x1 multiplications, each of rank 7 (Strassen).
        # Total rank: 7*7 = 49. Not better than schoolbook 28.
        #
        # Better approach: The 2x2 @ 2x7 tensor can be decomposed as:
        # For each of the 7 columns of B, we have a 2x2 @ 2x1 problem.
        # But we can share the Strassen factors across columns.
        #
        # Let's use a structured decomposition:
        # M1 = (A11 + A22) @ (B11 + B22)
        # M2 = (A21 + A22) @ B11
        # M3 = A11 @ (B21 - B11)
        # M4 = A22 @ (B12 - B22)
        # M5 = (A11 + A12) @ B22
        # M6 = (A21 - A11) @ (B11 + B12)
        # M7 = (A12 - A22) @ (B21 + B22)
        #
        # For 2x2 @ 2x7, B has 2 rows and 7 columns.
        # B11, B22, etc. would be 1x7 vectors.
        #
        # Let me index: A is 2x2, B is 2x7
        # A[0,0], A[0,1], A[1,0], A[1,1]
        # B[0,0..6], B[1,0..6]
        #
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        #
        # Strassen-like:
        # M1 = (A[0,0] + A[1,1]) * (B[0,k] + B[1,k]) for all k
        # M2 = (A[1,0] + A[1,1]) * B[0,k] for all k
        # M3 = A[0,0] * (B[1,k] - B[0,k]) for all k
        # M4 = A[1,1] * (B[0,k] - B[1,k]) for all k
        # M5 = (A[0,0] + A[0,1]) * B[1,k] for all k
        # M6 = (A[1,0] - A[0,0]) * (B[0,k] + B[1,k]) for all k  -- wait, this doesn't work for 2x7
        #
        # Actually, standard Strassen for 2x2@2x2 uses:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        #
        # For 2x2 @ 2x7, we can apply this per-column but share the A-side factors.
        # The B-side factors depend on k, so we need 7 separate B-side computations.
        # Total: 7 * 7 = 49 terms. Still not better.
        #
        # Let's try a different splitting. Split the output dimension instead.
        # Or use a more clever algebraic decomposition.
        #
        # Alternative: Use the fact that rank(2x4x4x7) <= rank(2x4x4x4) + rank(2x4x4x3)
        # by splitting the output dimension p=7 into 4+3.
        #
        # rank(2x4x4x4): This is the tensor for 2x4 @ 4x4.
        # Can we decompose this better?
        #
        # Let's try yet another approach: use bilinear decomposition with
        # structured factor matrices.
        
        # Direct construction using known results:
        # The rank of matrix multiplication tensor M_{n,m,p} is at most
        # n*m*p (schoolbook) but can be lower.
        #
        # For 2x4x4x7, let's try to construct a decomposition of rank ~35.
        
        # Split m=4 into j1 (0,1) and j2 (2,3)
        # C[i,k] = sum_{j=0}^1 A[i,j] B[j,k] + sum_{j=0}^1 A[i,j+2] B[j+2,k]
        #
        # First sum: 2x2 @ 2x7 tensor
        # Second sum: 2x2 @ 2x7 tensor
        #
        # For each 2x2 @ 2x7, let's use a rank-14 decomposition (half of schoolbook 28)
        # if possible, or use Strassen-like rank-21 (7 per column group of 2).
        
        # Actually, let's just try to find a good decomposition using
        # the alternating minimization but with a better initialization
        # based on the Strassen structure.
        
        return None  # Will be handled by alternating minimization below
    
    # Precompute T in different orientations for efficiency
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # For optimizing W
    T_vec_u = T.reshape(nm, mp * pn)  # For optimizing U
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # For optimizing V
    
    # Try alternating minimization for various ranks
    def try_rank_alt(r, n_trials=10, max_iter=300):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization with structured patterns
            if trial % 3 == 0:
                # Standard random
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 3 == 1:
                # Sparse-like initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    idx_u = np.random.choice(nm, size=max(1, nm//4), replace=False)
                    idx_v = np.random.choice(mp, size=max(1, mp//4), replace=False)
                    idx_w = np.random.choice(pn, size=max(1, pn//4), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            else:
                # Strassen-inspired structured initialization
                # Split middle index into two blocks of 2
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                
                # For each rank term, assign it to one of the two blocks
                # and use Strassen-like coefficients
                for d in range(r):
                    # Determine which block this term belongs to
                    block = d % 2  # 0 for j=0,1; 1 for j=2,3
                    
                    # Strassen coefficients for 2x2 multiplication
                    # 7 terms per block
                    s = d % 7
                    
                    # U side (A matrix, 2x2 per block)
                    if block == 0:
                        # A rows: i=0,1; cols: j=0,1
                        if s == 0:
                            U[0*4+0, d] = 1; U[1*4+1, d] = 1  # A11+A22
                        elif s == 1:
                            U[1*4+0, d] = 1; U[1*4+1, d] = 1  # A21+A22
                        elif s == 2:
                            U[0*4+0, d] = 1  # A11
                        elif s == 3:
                            U[1*4+1, d] = 1  # A22
                        elif s == 4:
                            U[0*4+0, d] = 1; U[0*4+1, d] = 1  # A11+A12
                        elif s == 5:
                            U[1*4+0, d] = 1; U[0*4+0, d] = -1  # A21-A11
                        elif s == 6:
                            U[0*4+1, d] = 1; U[1*4+1, d] = -1  # A12-A22
                    else:
                        # A rows: i=0,1; cols: j=2,3
                        if s == 0:
                            U[0*4+2, d] = 1; U[1*4+3, d] = 1  # A11+A22
                        elif s == 1:
                            U[1*4+2, d] = 1; U[1*4+3, d] = 1  # A21+A22
                        elif s == 2:
                            U[0*4+2, d] = 1  # A11
                        elif s == 3:
                            U[1*4+3, d] = 1  # A22
                        elif s == 4:
                            U[0*4+2, d] = 1; U[0*4+3, d] = 1  # A11+A12
                        elif s == 5:
                            U[1*4+2, d] = 1; U[0*4+2, d] = -1  # A21-A11
                        elif s == 6:
                            U[0*4+3, d] = 1; U[1*4+3, d] = -1  # A12-A22
                    
                    # V side (B matrix, 2x7 per block)
                    # For block 0: B rows j=0,1; for block 1: B rows j=2,3
                    # Strassen coefficients depend on the column k
                    # We need to distribute across the 7 columns
                    for k in range(p):
                        if block == 0:
                            # B[0,k] and B[1,k]
                            if s == 0:
                                V[0*7+k, d] = 1; V[1*7+k, d] = 1  # B11+B22
                            elif s == 1:
                                V[0*7+k, d] = 1  # B11
                            elif s == 2:
                                V[1*7+k, d] = 1; V[0*7+k, d] = -1  # B21-B11
                            elif s == 3:
                                V[0*7+k, d] = 1; V[1*7+k, d] = -1  # B12-B22
                            elif s == 4:
                                V[1*7+k, d] = 1  # B22
                            elif s == 5:
                                V[0*7+k, d] = 1; V[1*7+k, d] = 1  # B11+B12
                            elif s == 6:
                                V[1*7+k, d] = 1; V[0*7+k, d] = 1  # B21+B22
                        else:
                            # B[2,k] and B[3,k]
                            if s == 0:
                                V[2*7+k, d] = 1; V[3*7+k, d] = 1  # B11+B22
                            elif s == 1:
                                V[2*7+k, d] = 1  # B11
                            elif s == 2:
                                V[3*7+k, d] = 1; V[2*7+k, d] = -1  # B21-B11
                            elif s == 3:
                                V[2*7+k, d] = 1; V[3*7+k, d] = -1  # B12-B22
                            elif s == 4:
                                V[3*7+k, d] = 1  # B22
                            elif s == 5:
                                V[2*7+k, d] = 1; V[3*7+k, d] = 1  # B11+B12
                            elif s == 6:
                                V[3*7+k, d] = 1; V[2*7+k, d] = 1  # B21+B22
                    
                    # W side (C matrix, 2x7)
                    # C[i,k] = combination of M1..M7
                    for i in range(n):
                        for k in range(p):
                            coeff = 0
                            if i == 0:
                                if s == 0: coeff += 1
                                elif s == 3: coeff += 1
                                elif s == 4: coeff -= 1
                                elif s == 6: coeff -= 1
                                # C11 = M1 + M4 - M5 - M7 (for k=0)
                                # But we're distributing across all k...
                            else:
                                if s == 1: coeff += 1
                                elif s == 3: coeff += 1
                                elif s == 0: coeff -= 1
                                elif s == 2: coeff += 1
                                # C21 = M2 + M4, C22 = M1 - M2 + M3 + M7
                            
                            # This is getting complex. Let's just use random W for now
                            # and let alternating minimization fix it.
                            W[k*n+i, d] = np.random.randn()
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Build X for optimizing W: X[a*mp+b, d] = U[a,d]*V[b,d]
                X = np.zeros((nm * mp, r))
                for d in range(r):
                    X[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                
                # Solve for W
                XTX = X.T @ X + 1e-10 * np.eye(r)
                XTT = X.T @ T_vec_w.T
                try:
                    W_new = np.linalg.solve(XTX, XTT).T
                except np.linalg.LinAlgError:
                    W_new = W
                
                # Build Y for optimizing U: Y[b*pn+c, d] = V[b,d]*W[c,d]
                Y = np.zeros((mp * pn, r))
                for d in range(r):
                    Y[:, d] = (V[:, d][:, None] * W_new[:, d][None, :]).flatten()
                
                # Solve for U
                YTY = Y.T @ Y + 1e-10 * np.eye(r)
                YTT = Y.T @ T_vec_u.T
                try:
                    U_new = np.linalg.solve(YTY, YTT).T
                except np.linalg.LinAlgError:
                    U_new = U
                
                # Build Z for optimizing V: Z[a*pn+c, d] = U[a,d]*W[c,d]
                Z = np.zeros((nm * pn, r))
                for d in range(r):
                    Z[:, d] = (U_new[:, d][:, None] * W_new[:, d][None, :]).flatten()
                
                # Solve for V
                ZTZ = Z.T @ Z + 1e-10 * np.eye(r)
                ZTT = Z.T @ T_vec_v.T
                try:
                    V_new = np.linalg.solve(ZTZ, ZTT).T
                except np.linalg.LinAlgError:
                    V_new = V
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_rec = tensor_from_factors(U, V, W)
                err = np.linalg.norm(T_rec - T)
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            
            # Final check
            T_rec = tensor_from_factors(U, V, W)
            if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec_r = tensor_from_factors(U_r, V_r, W_r)
                if np.allclose(T_rec_r, T, atol=1e-8, rtol=1e-8):
                    best_U, best_V, best_W = U_r, V_r, W_r
                else:
                    best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        return False
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But known results suggest lower ranks may exist
    lower_bound = max(n*m, m*p, p*n)
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        success = try_rank_alt(r, n_trials=8)
        if success:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
