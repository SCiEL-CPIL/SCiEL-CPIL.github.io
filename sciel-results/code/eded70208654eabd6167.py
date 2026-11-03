"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Shapes
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    # Build the target tensor structure
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # We need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # List of (a, b, c) triples where T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    triple_set = set(triples)
    num_triples = len(triples)  # 144
    
    # Pre-compute target vectors for each slice
    # target_U[a] = vector of length V_rows*W_rows with 1s at positions (b,c) where (a,b,c) in triples
    target_U = np.zeros((U_rows, V_rows * W_rows))
    target_V = np.zeros((V_rows, U_rows * W_rows))
    target_W = np.zeros((W_rows, U_rows * V_rows))
    for (a, b, c) in triples:
        target_U[a, b * W_rows + c] = 1.0
        target_V[b, a * W_rows + c] = 1.0
        target_W[c, a * V_rows + b] = 1.0
    
    # Schoolbook solution (rank = n*m*p = 144)
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact for all entries."""
        r = U.shape[1]
        # Check all triple entries = 1
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check all non-triple entries = 0 using vectorized approach
        # For each a, compute T_a = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # and check that non-triple entries are ~0
        for a in range(U_rows):
            # Compute all (b,c) values for this a
            vals = np.zeros(V_rows * W_rows)
            for r in range(r):
                vals += U[a, r] * np.outer(V[:, r], W[:, r]).flatten()
            # Check non-triple positions
            for (ta, tb, tc) in triples:
                if ta == a:
                    vals[tb * W_rows + tc] = 0  # skip triple positions
            if np.max(np.abs(vals)) > tol:
                return False
        return True
    
    def check_decomposition_fast(U, V, W, tol=1e-5):
        """Faster check using vectorized operations."""
        r = U.shape[1]
        # Check triples
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # For non-triples, check all entries vectorized per a
        for a in range(U_rows):
            vals = np.zeros(V_rows * W_rows)
            for r in range(r):
                vals += U[a, r] * np.outer(V[:, r], W[:, r]).flatten()
            # Zero out triple positions
            for (ta, tb, tc) in triples:
                if ta == a:
                    vals[tb * W_rows + tc] = 0
            if np.max(np.abs(vals)) > tol:
                return False
        return True
    
    # Try to find lower rank decompositions using various strategies
    # Lower bound is max(U_rows, V_rows, W_rows) = 36
    
    # Strategy 1: Try to use known structure
    # For matrix multiplication, there are known decompositions
    # The rank of 4x4x9 matrix multiplication is at most 4*4*9 = 144 (schoolbook)
    # But can be lower.
    
    # Let's try a structured approach:
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1
    # We can try to exploit the structure
    
    # Try ranks from a reasonable lower bound
    min_rank = max(U_rows, V_rows, W_rows)  # 36
    
    # Use alternating least squares (ALS)
    def als_decompose(target_rank, max_iters=100, tol=1e-8, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(U_rows, target_rank) * 0.1
        V = rng.randn(V_rows, target_rank) * 0.1
        W = rng.randn(W_rows, target_rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each a, minimize sum over (b,c) of (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
            # This is a linear system: U[a,:] @ (V^T W) ... 
            # Actually, for fixed a: 
            # T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
            # Let M_a = T[a,:,:] reshaped
            # We want to find U[a,:] such that sum_r U[a,r] * outer(V[:,r], W[:,r]) approx T[a,:,:]
            
            # Build the design matrix for U[a,:]
            # Each column r contributes outer(V[:,r], W[:,r]) to the sum
            # So we need: U[a,:] @ B = vec(T[a,:,:])
            # where B[r, (b,c)] = V[b,r] * W[c,r]
            
            B = np.zeros((target_rank, V_rows * W_rows))
            for r in range(target_rank):
                B[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            for a in range(U_rows):
                # Target: T[a, b, c] = 1 if (a,b,c) in triples, else 0
                target_vec = np.zeros(V_rows * W_rows)
                for (ta, tb, tc) in triples:
                    if ta == a:
                        target_vec[tb * W_rows + tc] = 1.0
                
                # Solve: U[a,:] @ B = target_vec
                # i.e., B^T @ U[a,:] = target_vec
                try:
                    U[a, :] = np.linalg.lstsq(B.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Similarly update V and W
            # For V[b,:]: T[:,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Fix b: for each a,c: sum_r U[a,r] * W[c,r] * V[b,r] = T[a,b,c]
            # Design matrix C for V[b,:]: C[r, (a,c)] = U[a,r] * W[c,r]
            
            C = np.zeros((target_rank, U_rows * W_rows))
            for r in range(target_rank):
                C[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            for b in range(V_rows):
                target_vec = np.zeros(U_rows * W_rows)
                for (ta, tb, tc) in triples:
                    if tb == b:
                        target_vec[ta * W_rows + tc] = 1.0
                
                try:
                    V[b, :] = np.linalg.lstsq(C.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # For W[c,:]: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Fix c: for each a,b: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Design matrix D for W[c,:]: D[r, (a,b)] = U[a,r] * V[b,r]
            
            D = np.zeros((target_rank, U_rows * V_rows))
            for r in range(target_rank):
                D[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            for c in range(W_rows):
                target_vec = np.zeros(U_rows * V_rows)
                for (ta, tb, tc) in triples:
                    if tc == c:
                        target_vec[ta * V_rows + tb] = 1.0
                
                try:
                    W[c, :] = np.linalg.lstsq(D.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            if it % 10 == 0:
                # Compute error
                err = 0
                for (a, b, c) in triples:
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += (val - 1.0) ** 2
                if err < tol:
                    break
        
        return U, V, W
    
    def polish_decomposition(U, V, W, max_iters=200, tol=1e-12):
        """High-precision polish pass using alternating least squares with tighter tolerance."""
        if time.time() > deadline - 5:
            return U, V, W
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            B = np.zeros((r, V_rows * W_rows))
            for rr in range(r):
                B[rr, :] = np.outer(V[:, rr], W[:, rr]).flatten()
            
            for a in range(U_rows):
                target_vec = np.zeros(V_rows * W_rows)
                for (ta, tb, tc) in triples:
                    if ta == a:
                        target_vec[tb * W_rows + tc] = 1.0
                try:
                    U[a, :] = np.linalg.lstsq(B.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            C = np.zeros((r, U_rows * W_rows))
            for rr in range(r):
                C[rr, :] = np.outer(U[:, rr], W[:, rr]).flatten()
            
            for b in range(V_rows):
                target_vec = np.zeros(U_rows * W_rows)
                for (ta, tb, tc) in triples:
                    if tb == b:
                        target_vec[ta * W_rows + tc] = 1.0
                try:
                    V[b, :] = np.linalg.lstsq(C.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            D = np.zeros((r, U_rows * V_rows))
            for rr in range(r):
                D[rr, :] = np.outer(U[:, rr], V[:, rr]).flatten()
            
            for c in range(W_rows):
                target_vec = np.zeros(U_rows * V_rows)
                for (ta, tb, tc) in triples:
                    if tc == c:
                        target_vec[ta * V_rows + tb] = 1.0
                try:
                    W[c, :] = np.linalg.lstsq(D.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence with tight tolerance
            if it % 20 == 0:
                err = 0
                for (a, b, c) in triples:
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += (val - 1.0) ** 2
                if err < tol:
                    break
        
        return U, V, W
    
    # Try different ranks - start lower and use more seeds
    for target_rank in range(36, min(100, rank_school + 1)):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_seeds = 5 if target_rank < 50 else 3
        for seed in range(num_seeds):
            if time.time() > deadline - 15:
                break
            
            result = als_decompose(target_rank, max_iters=80, seed=seed)
            if result is None:
                continue
            
            U, V, W = result
            
            # Check if this is a valid decomposition (fast check)
            if check_decomposition_fast(U, V, W, tol=1e-5):
                # Apply polish pass for high precision
                U_polished, V_polished, W_polished = polish_decomposition(U, V, W, max_iters=100, tol=1e-12)
                
                # Check polished version with full verification
                if check_decomposition(U_polished, V_polished, W_polished, tol=1e-9):
                    # Try to round to half-integers
                    U_rounded = np.round(U_polished * 2) / 2
                    V_rounded = np.round(V_polished * 2) / 2
                    W_rounded = np.round(W_polished * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        best_U, best_V, best_W = U_polished, V_polished, W_polished
                        best_rank = target_rank
                        found = True
                        break
                elif check_decomposition(U, V, W, tol=1e-9):
                    # Use unpolished if valid
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_school:
        # Ensure entries are half-integers if possible
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        
        if check_decomposition(U_out, V_out, W_out, tol=1e-9):
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        else:
            # Final verification before returning
            if check_decomposition(best_U, best_V, best_W, tol=1e-8):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
