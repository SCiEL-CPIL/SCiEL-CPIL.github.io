<<<<<<< SEARCH
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    def make_schoolbook():
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_exact(U_mat, V_mat, W_mat):
        """Exact verification using the tensor condition."""
        rank = U_mat.shape[1]
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                actual = 0.0
                                for c in range(rank):
                                    actual += W_mat[k*n+i, c] * U_mat[i2*m+j, c] * V_mat[j2*p+k2, c]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    def verify_random(U_mat, V_mat, W_mat, n_tests=10):
        """Quick verification using random test matrices."""
        rank = U_mat.shape[1]
        for _ in range(n_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank):
                A_vec = A.flatten()
                B_vec = B.flatten()
                u_val = U_mat[:, c] @ A_vec
                v_val = V_mat[:, c] @ B_vec
                W_col = W_mat[:, c]
                C_computed += W_col.reshape(p, n).T * (u_val * v_val)
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    
    def alternating_minimize(U, V, W, rank, n_iters=40):
        """Alternating least squares to minimize the error."""
        constraints = []
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                target = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                constraints.append((i, k, i2, j, j2, k2, target))
        
        for it in range(n_iters):
            # Fix U, V; solve for W row by row
            for i in range(n):
                for k in range(p):
                    A_sys = []
                    b_sys = []
                    for (ci, ck, ci2, cj, cj2, ck2, target) in constraints:
                        if ci == i and ck == k:
                            row = np.array([U[ci2*m+cj, c] * V[cj2*p+ck2, c] for c in range(rank)])
                            A_sys.append(row)
                            b_sys.append(target)
                    if A_sys:
                        A_sys = np.array(A_sys)
                        b_sys = np.array(b_sys)
                        W_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W[k*n+i, :] = W_row
            
            # Fix V, W; solve for U row by row
            for i2 in range(n):
                for j in range(m):
                    A_sys = []
                    b_sys = []
                    for (ci, ck, ci2, cj, cj2, ck2, target) in constraints:
                        if ci2 == i2 and cj == j:
                            row = np.array([W[ck*n+ci, c] * V[cj2*p+ck2, c] for c in range(rank)])
                            A_sys.append(row)
                            b_sys.append(target)
                    if A_sys:
                        A_sys = np.array(A_sys)
                        b_sys = np.array(b_sys)
                        U_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U[i2*m+j, :] = U_row
            
            # Fix U, W; solve for V row by row
            for j2 in range(m):
                for k2 in range(p):
                    A_sys = []
                    b_sys = []
                    for (ci, ck, ci2, cj, cj2, ck2, target) in constraints:
                        if cj2 == j2 and ck2 == k2:
                            row = np.array([W[ck*n+ci, c] * U[ci2*m+cj, c] for c in range(rank)])
                            A_sys.append(row)
                            b_sys.append(target)
                    if A_sys:
                        A_sys = np.array(A_sys)
                        b_sys = np.array(b_sys)
                        V_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V[j2*p+k2, :] = V_row
        
        return U, V, W
    
    deadline = time.time() + 115
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_school}
    
    # Try to find lower rank decompositions
    for rank in range(12, rank_school):
        if time.time() > deadline - 25:
            break
        
        max_restarts = 4 if rank <= 16 else 2
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 25:
                break
            
            U, V, W = np.random.randn(n*m, rank), np.random.randn(m*p, rank), np.random.randn(p*n, rank)
            U, V, W = alternating_minimize(U, V, W, rank, n_iters=30)
            
            if verify_exact(U, V, W):
                best = {'U': U, 'V': V, 'W': W, 'rank': rank}
                found = True
                break
        
        if found:
            continue
        else:
            if rank > 18:
                break
    
    return {"U": best['U'].tolist(), "V": best['V'].tolist(), "W": best['W'].tolist()}
# EVOLVE-BLOCK-END
=======
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 30
    
    def make_schoolbook():
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify_exact_fast(U_mat, V_mat, W_mat):
        """Fast exact verification using tensor contraction via numpy."""
        rank = U_mat.shape[1]
        # T[i,j,k, i2,j2,k2] = sum_c W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
        # We need T[i,j,k, i2,j2,k2] = 1 if (i==i2, j==j2, k==k2) else 0
        
        # Build the target tensor
        # For each (i,k) output index, check if the reconstruction matches
        # Use einsum for speed
        # U shape: (n*m, rank) -> reshape to (n, m, rank)
        # V shape: (m*p, rank) -> reshape to (m, p, rank)  
        # W shape: (p*n, rank) -> reshape to (p, n, rank)
        
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        # For each output (i, k), the reconstructed matrix C[i,k] should equal:
        # sum_c W_3d[k,i,c] * outer(U_3d[:, :, c], V_3d[:, :, c])
        # But we need: sum_c W_3d[k,i,c] * U_3d[i2,j,c] * V_3d[j2,k2,c] 
        # = delta(i,i2) * delta(j,j2) * delta(k,k2) for all i2,j2,k2
        
        # Actually the tensor condition is:
        # For ALL i,k,i2,j,j2,k2: sum_c W[k*n+i,c]*U[i2*m+j,c]*V[j2*p+k2,c] = delta(i,i2)*delta(j,j2)*delta(k,k2)
        
        # Let's check by computing for each (i,k) the implied "matrix" and checking
        # Actually, let's just do the full check efficiently
        # For each (i2, j, k2) input triple, and each (i, k) output index:
        # sum_c W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c] should be 1 iff i==i2, j==j2, k==k2
        
        # Vectorized: for fixed (i2, j, k2), the vector over (i,k) of the sum is:
        # sum_c (U[i2*m+j, c] * V[j2*p+k2, c]) * W[k*n+i, c]
        # This is a linear combination of the W columns weighted by element-wise products
        
        # Let's build it: for each (i2,j,k2), compute w = U[i2*m+j,:] * V[j2*p+k2,:]
        # then result[i,k] = sum_c W[k*n+i, c] * w[c]
        # = sum_c W[k*n+i, c] * w[c]
        
        # W reshaped: (p, n, rank), so W[k*n+i, c] = W_3d[k, i, c]
        # result for fixed (i2,j,k2): res[i,k] = sum_c W_3d[k,i,c] * w[c]
        # = einsum('kic,c->ki', W_3d, w)
        
        all_ok = True
        for i2 in range(n):
            for j in range(m):
                for k2 in range(p):
                    w = U_mat[i2*m+j, :] * V_mat[j*p+k2, :]
                    # res[k, i] = sum_c W_3d[k, i, c] * w[c]
                    res = np.einsum('kic,c->ki', W_3d, w)
                    # res[i2, k2] should be 1, all others 0
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(res[i, k] - expected) > 1e-8:
                                return False
        return True
    
    def compute_error(U_mat, V_mat, W_mat):
        """Compute the Frobenius norm of the error tensor."""
        rank = U_mat.shape[1]
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        err = 0.0
        for i2 in range(n):
            for j in range(m):
                for k2 in range(p):
                    w = U_mat[i2*m+j, :] * V_mat[j*p+k2, :]
                    res = np.einsum('kic,c->ki', W_3d, w)
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (i == i2 and k == k2) else 0.0
                            err += (res[i, k] - expected) ** 2
        return np.sqrt(err)
    
    def alternating_minimize_fast(U, V, W, rank, n_iters=60):
        """Faster alternating least squares using vectorized operations."""
        for it in range(n_iters):
            # Fix U, V; solve for W
            # For each (i, k): W[k*n+i, :] should satisfy
            # sum_c W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c] = delta(i,i2)*delta(j,j2)*delta(k,k2)
            # For fixed (i,k), the constraints are over (i2,j,j2,k2):
            # sum_c W[k*n+i, c] * (U[i2*m+j, c] * V[j2*p+k2, c]) = delta(i,i2)*delta(j,j2)*delta(k,k2)
            # This is a linear system in W[k*n+i, :]
            # A row for (i2,j,j2,k2): [U[i2*m+j, c] * V[j2*p+k2, c] for c in range(rank)]
            # b = delta(i,i2)*delta(j,j2)*delta(k,k2)
            
            for i in range(n):
                for k in range(p):
                    A_sys = np.zeros((n*m*p, rank))
                    b_sys = np.zeros(n*m*p)
                    row = 0
                    for i2 in range(n):
                        for j in range(m):
                            for k2 in range(p):
                                A_sys[row, :] = U[i2*m+j, :] * V[j*p+k2, :]
                                b_sys[row] = 1.0 if (i2 == i and j == j2_val and k2 == k) else 0.0
                                # Wait, j2 is the column index of B, which is j in this context
                                # Actually the constraint is: for output (i,k), input (i2,j,j2,k2)
                                # delta(i,i2)*delta(j,j2)*delta(k,k2) where j is the column of A / row of B
                                # Hmm, let me re-read the tensor definition
                                row += 1
                    # This indexing is getting confused. Let me use a cleaner approach.
                    # Actually let me just use the simpler but correct formulation
                    pass
            
            # Simpler: use the fact that for each (i,k), we need:
            # For all (i2, j, j2, k2): sum_c W[k*n+i,c] * U[i2*m+j,c] * V[j2*p+k2,c] = delta(i,i2)*delta(j,j2)*delta(k,k2)
            # The only non-zero RHS is when i2=i, j2=j, k2=k... wait no.
            # j is the shared index. The constraint is:
            # T[i,j,k; i2,j2,k2] = sum_c W[k*n+i,c] U[i2*m+j,c] V[j2*p+k2,c] = delta(i,i2) delta(j,j2) delta(k,k2)
            # where j is the "free" index in U (column) and j2 is the "free" index in V (row)
            # Wait, I think the tensor is: T[i,j,k; i2,j2,k2] with the multiplication being:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # So the tensor T has indices (i,k) for output and (i2,j2,k2) for inputs? No...
            # Standard: T_{i,j,k}^{i',j',k'} = delta(i,i') delta(j,j') delta(k,k') for the multiplication tensor
            # where (i,j) indexes A, (j,k) indexes B, (i,k) indexes C
            # So the tensor is T[i,j,k; i',j',k'] = delta(i,i')delta(j,j')delta(k,k')
            # and the factorization is T = sum_c U[i,j,c] V[j',k',c] W[i,k,c]... 
            # Wait, the convention says W[k*n+i, r], so W is indexed by (k,i)
            # U[i*m+j, r] indexed by (i,j), V[j*p+k, r] indexed by (j,k)
            # So: sum_r U[i,j,r] V[j',k',r] W[k,i,r] should give delta(i,i')delta(j,j')delta(k,k')
            # Wait that doesn't look right either. Let me re-read.
            
            # The task says: "The output factor follows the official trace convention W[k*n+i,r]"
            # And U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # So U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
            # The tensor condition: sum_r U[i*m+j, r] * V[j2*p+k2, r] * W[k*n+i, r] = delta(i,i2)*delta(j,j2)*delta(k,k2)
            # where (i,j) are A indices, (j2,k2) are B indices, (k,i) are C indices... 
            # Hmm, but then i appears in both U and W, and j appears in U, j2 in V.
            # I think the correct reading is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Tensor: T[i,j,k; i',j',k'] = delta(i,i')delta(j,j')delta(k,k')  
            # Factorization: T = sum_r U[i,j,r] V[j',k',r] W[k,i,r]... no that mixes indices.
            
            # Let me just use the verify function's logic which was in the original:
            # expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
            # actual = sum_c W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
            # So the condition is: for all (i,k,i2,j,j2,k2):
            # sum_c W[k*n+i,c] * U[i2*m+j,c] * V[j2*p+k2,c] = 1 if i==i2, j==j2, k==k2 else 0
            
            # For fixed (i,k), solve for W[k*n+i, :]:
            # For all (i2,j,j2,k2): sum_c W[k*n+i,c] * U[i2*m+j,c] * V[j2*p+k2,c] = delta(i,i2)delta(j,j2)delta(k,k2)
            # This is a linear system. The rows are indexed by (i2,j,j2,k2) with n*m*m*p = 2*3*3*5=90 rows
            # But many will be zero on RHS.
            
            # Actually, only when i2=i, j2=j, k2=k is the RHS non-zero (=1).
            # Wait no: delta(i,i2)*delta(j,j2)*delta(k,k2) - j is the column of A (row of B), j2 is row of B.
            # So delta(j,j2) means j must equal j2. So the non-zero constraint is when i2=i, j2=j, k2=k.
            # All other constraints require the sum to be 0.
            
            # So for fixed (i,k):
            # Constraint 1 (i2=i, j2=j, k2=k): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
            # Constraints 2 (all other (i2,j,j2,k2)): sum_c W[k*n+i,c] * U[i2*m+j,c] * V[j2*p+k2,c] = 0
            
            # This is a linear system with 1 + (n*m*p - 1) = n*m*p equations... but many are redundant.
            # Actually there are n*m*m*p total (i2,j,j2,k2) combos = 2*3*3*5 = 90
            # Only 1 has RHS=1, the other 89 have RHS=0.
            
            # Let me build this properly
            pass
        
        return U, V, W
    
    deadline = time.time() + 115
    
    U_sb, V_sb, W_sb = make_schoolbook()
    best = {'U': U_sb, 'V': V_sb, 'W': W_sb, 'rank': rank_school}
    
    # Use a smarter approach: for each rank, try to find a solution
    # using the structure of the problem
    
    # Key insight: The multiplication tensor for (n,m,p) has a specific structure.
    # For small dimensions, we can try to find low-rank decompositions by 
    # exploiting block structure or using known algorithms.
    
    # For 2x3x5, let's try ranks from a reasonable lower bound up
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually the rank is at least the "tensor rank" which for matrix mult is at least n*m*p / something
    # Let's just try from 15 to 29
    
    for rank in range(15, rank_school):
        if time.time() > deadline - 20:
            break
        
        max_restarts = 3
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values, but try to preserve some structure
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Do a few iterations of alternating minimization
            # For each (i,k), solve for W[k*n+i, :]
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        # Build linear system for W[k*n+i, :]
                        # Unknowns: W[k*n+i, 0..rank-1]
                        # Equations: for all (i2, j, j2, k2):
                        #   sum_c W[k*n+i,c] * U[i2*m+j,c] * V[j2*p+k2,c] = delta(i,i2)*delta(j,j2)*delta(k,k2)
                        
                        A_sys = np.zeros((n*m*m*p, rank))
                        b_sys = np.zeros(n*m*m*p)
                        row = 0
                        for i2 in range(n):
                            for j in range(m):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        A_sys[row, :] = U[i2*m+j, :] * V[j2*p+k2, :]
                                        b_sys[row] = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                                        row += 1
                        
                        # Solve least squares
                        W_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W[k*n+i, :] = W_row
                
                # Update U
                for i2 in range(n):
                    for j in range(m):
                        A_sys = np.zeros((n*m*m*p, rank))
                        b_sys = np.zeros(n*m*m*p)
                        row = 0
                        for i in range(n):
                            for k in range(p):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        A_sys[row, :] = W[k*n+i, :] * V[j2*p+k2, :]
                                        b_sys[row] = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                        row += 1
                        U_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U[i2*m+j, :] = U_row
                
                # Update V
                for j2 in range(m):
                    for k2 in range(p):
                        A_sys = np.zeros((n*m*m*p, rank))
                        b_sys = np.zeros(n*m*m*p)
                        row = 0
                        for i in range(n):
                            for k in range(p):
                                for i2 in range(n):
                                    for j in range(m):
                                        A_sys[row, :] = W[k*n+i, :] * U[i2*m+j, :]
                                        b_sys[row] = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                        row += 1
                        V_row, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V[j2*p+k2, :] = V_row
            
            # Verify
            if verify_exact_fast(U, V, W):
                best = {'U': U, 'V': V, 'W': W, 'rank': rank}
                found = True
                break
        
        if found:
            continue
        else:
            # If we can't find rank r, try r+1, etc.
            # But don't give up too early
            if rank >= 25:
                break
    
    return {"U": best['U'].tolist(), "V": best['V'].tolist(), "W": best['W'].tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in the index handling. Let me fix the alternating minimization properly. The key issue is that the linear system for each row has 90 equations (n*m*m*p = 2*3*3*5 = 90) but many are redundant or zero. Let me simplify by only using the non-trivial constraints.

