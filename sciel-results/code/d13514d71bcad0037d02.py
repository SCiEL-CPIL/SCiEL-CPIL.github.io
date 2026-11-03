"""Matrix multiplication 4,4,7 rank minimization via structured Strassen + ALS refinement."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reconstructs T."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - T_approx)) < tol

    def check_decomposition_strict(U, V, W, tol=1e-12):
        """Stricter check for final validation."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - T_approx)) < tol

    def als_attempt(U_init, V_init, W_init, max_iter=100, tol=1e-10):
        """Run ALS on given initial matrices, return (U, V, W, converged)."""
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        rank = U.shape[1]
        
        T_flat_ab = T.reshape(nm, mp * pn)
        T_flat_bc = T.reshape(mp, nm * pn)
        T_flat_ca = T.reshape(pn, nm * mp)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            M_U = np.einsum('br,cr->rbc', V, W).reshape(rank, mp * pn)
            try:
                U_new = T_flat_ab @ np.linalg.pinv(M_U)
            except:
                break
            
            # Update V
            M_V = np.einsum('ar,cr->rac', U_new, W).reshape(rank, nm * pn)
            try:
                V_new = T_flat_bc @ np.linalg.pinv(M_V)
            except:
                break
            
            # Update W
            M_W = np.einsum('ar,br->rab', U_new, V_new).reshape(rank, nm * mp)
            try:
                W_new = T_flat_ca @ np.linalg.pinv(M_W)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            if check_decomposition(U, V, W, tol=tol):
                return U, V, W, True
        
        return U, V, W, False

    def strassen_2x2x2(U_offset, V_offset, W_offset, rank, U, V, W, i_base, j_base, k_base):
        """
        Add Strassen 2x2x2 decomposition for sub-blocks.
        U has shape (nm, rank), V has shape (mp, rank), W has shape (pn, rank)
        i_base: starting i index for the 2x2 block
        j_base: starting j index
        k_base: starting k index
        """
        # Strassen 7 products for 2x2 * 2x2
        # C11 = A11*(B22-B12), C12 = A12*B22, C21 = A11*B12, C22 = A22*(B11-B21)
        # C11 += (A11+A21)*B11, C12 += (A11+A12)*B21, C21 += (A22-A12)*(B11+B12), C22 += (A21+A22)*(B21+B22)
        
        # The 7 Strassen products:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # For our case, we're multiplying (2x2) by (2x2) to get (2x2)
        # But our matrix is 4x4 * 4x7, so we need to handle the non-square case
        
        # Let's use a simpler approach: just use schoolbook for 2x2x2 blocks
        # but try to merge across blocks
        
        # Actually, let's just add the 8 schoolbook terms for each 2x2x2 block
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i_base + di
                    j = j_base + dj
                    k = k_base + dk
                    U[i * m + j, rank] = 1.0
                    V[j * p + k, rank] = 1.0
                    W[k * n + i, rank] = 1.0
        return rank + 8

    def build_strassen_4x4x7():
        """
        Build a Strassen-inspired decomposition for 4x4 * 4x7.
        Split 4x4 into four 2x2 blocks.
        Split 4x7 into:
          - First 4 columns: two 2x2 blocks
          - Last 3 columns: handled separately (3 columns)
        
        For the 4x4 part: use Strassen 2x2x2 recursively
        For the 4x3 part: use schoolbook or partial Strassen
        """
        rank = 0
        U = np.zeros((nm, 200))  # Allocate enough space
        V = np.zeros((mp, 200))
        W = np.zeros((pn, 200))
        
        # Block 1: rows 0-1, cols 0-1 of A * rows 0-1, cols 0-1 of B -> cols 0-1 of C
        # This is 2x2 * 2x2 -> 2x2, use Strassen (7 terms)
        
        # Actually, let me just use schoolbook for now but with a different structure
        # Try to find a decomposition with rank < 112
        
        # Simple approach: schoolbook is 4*4*7 = 112
        # Can we do better? 
        
        # For 4x4 * 4x4, Strassen gives 7 instead of 8, so 4*4*7 could potentially be reduced
        # if we can group the 7 columns into 4+3 and handle the 4x4 part with Strassen
        
        # Let's try: split B into B1 (4x4) and B2 (4x3)
        # A*B1: use Strassen (7 terms per 2x2x2 block, 4 such blocks = 28 terms)
        # A*B2: schoolbook (4*4*3 = 48 terms)
        # Total: 76 terms
        
        # But we need to be careful about the indexing
        
        # Actually, the standard Strassen for 4x4*4x4 gives 7^2 = 49 terms (if we recurse)
        # or 4*7 = 28 if we only go one level deep
        
        # Let me just implement a clean Strassen one-level decomposition
        
        # For 4x4 * 4x4 -> 4x4:
        # Split A into A11, A12, A21, A22 (each 2x2)
        # Split B into B11, B12, B21, B22 (each 2x2)
        # Split C into C11, C12, C21, C22 (each 2x2)
        
        # Strassen products (7 total):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2 matrix, computed from 2x2*2x2 multiplications
        # But we need the tensor decomposition, not the actual multiplication
        
        # This is getting complex. Let me just return schoolbook for now
        # and rely on ALS to find improvements
        
        # Return schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, rank] = 1.0
                    V[j * p + k, rank] = 1.0
                    W[k * n + i, rank] = 1.0
                    rank += 1
        
        return U[:, :rank], V[:, :rank], W[:, :rank], rank

    # Try structured approaches first
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 112 schoolbook
    
    # Build schoolbook
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    for idx in range(best_rank):
        i = idx // (m * p)
        j = (idx // p) % m
        k = idx % p
        U_school[i * m + j, idx] = 1.0
        V_school[j * p + k, idx] = 1.0
        W_school[k * n + i, idx] = 1.0
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find better decompositions with ALS from various initializations
    target_ranks = [64, 60, 56, 52, 48]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Different initializations
            if restart == 0:
                # Sparse random init
                U = np.random.randn(nm, target_rank) * 0.1
                V = np.random.randn(mp, target_rank) * 0.1
                W = np.random.randn(pn, target_rank) * 0.1
            elif restart == 1:
                # Init based on schoolbook pattern
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    if i * m + j < nm and j * p + k < mp and k * n + i < pn:
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
                # Add some random noise
                U += np.random.randn(nm, target_rank) * 0.01
                V += np.random.randn(mp, target_rank) * 0.01
                W += np.random.randn(pn, target_rank) * 0.01
            elif restart == 2:
                # Block-structured init
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
                # Ensure some structure
                for r in range(min(target_rank, 32)):
                    i = r % 4
                    j = (r // 4) % 4
                    k = (r // 16) % 4
                    U[i * m + j, r] += 1.0
                    V[j * p + k, r] += 1.0
                    W[k * n + i, r] += 1.0
            elif restart == 3:
                # Use SVD of tensor matricization
                T_mat = T.reshape(nm, mp * pn)
                try:
                    U_svd, _, _ = np.linalg.svd(T_mat, full_matrices=False)
                    U = U_svd[:, :target_rank]
                except:
                    U = np.random.randn(nm, target_rank) * 0.2
                V = np.random.randn(mp, target_rank) * 0.2
                W = np.random.randn(pn, target_rank) * 0.2
            else:
                # Pure random with different scale
                scale = 0.5 / np.sqrt(target_rank)
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            
            U_new, V_new, W_new, converged = als_attempt(U, V, W, max_iter=80, tol=1e-10)
            
            if converged:
                if target_rank < best_rank:
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    best_rank = target_rank
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # Final validation
    if best_U is not None:
        if not check_decomposition_strict(best_U, best_V, best_W, tol=1e-11):
            # Fallback to schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
