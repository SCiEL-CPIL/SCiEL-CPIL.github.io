"""Program for matrix multiplication tensor decomposition (3x4x5)."""
import numpy as np
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    schoolbook_rank = n * m * p  # 60
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    # 1. Construct the initial schoolbook decomposition
    U0 = np.zeros((dim_u, schoolbook_rank))
    V0 = np.zeros((dim_v, schoolbook_rank))
    W0 = np.zeros((dim_w, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
                
    # Helper to verify decomposition using exact arithmetic
    def verify_exact(U, V, W):
        r = U.shape[1]
        # Convert to Fraction for exact arithmetic
        U_frac = [[Fraction(U[i, j]) for j in range(r)] for i in range(dim_u)]
        V_frac = [[Fraction(V[i, j]) for j in range(r)] for i in range(dim_v)]
        W_frac = [[Fraction(W[i, j]) for j in range(r)] for i in range(dim_w)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = Fraction(1)
                    val = Fraction(0)
                    for t in range(r):
                        val += U_frac[i*m+j][t] * V_frac[j*p+k][t] * W_frac[k*n+i][t]
                    if val != target:
                        return False
        return True

    def verify_numeric(U, V, W, tol=1e-10):
        """Fast numeric verification for screening"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - target) > tol:
                        return False
        return True

    if not verify_numeric(U0, V0, W0):
        raise ValueError("Schoolbook decomposition failed verification")

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = schoolbook_rank

    # 2. Search for lower rank using local search from schoolbook
    # Strategy: Try to merge pairs of columns by finding linear combinations
    # that preserve the tensor contribution
    
    def try_merge_pair(U, V, W, t1, t2):
        """Try to merge columns t1 and t2 into a single column.
        
        We look for coefficients a, b such that:
        a * (U[:,t1] ⊗ V[:,t1] ⊗ W[:,t1]) + b * (U[:,t2] ⊗ V[:,t2] ⊗ W[:,t2])
        can be replaced by a single rank-1 tensor that has the same contribution
        to all (i,j,k) entries.
        
        This is generally not possible, but we can check if the two rank-1 tensors
        are linearly dependent in a specific way.
        """
        if time.time() > deadline:
            return None
        
        # The contribution of column t is: U[:,t] * V[:,t] * W[:,t] (outer product)
        # For the merge to work, we need the sum of two outer products to be
        # another outer product. This is a strong condition.
        
        # Check if the two rank-1 tensors are "compatible" for merging
        # A necessary condition: the supports must be related in a specific way
        
        u1, v1, w1 = U[:, t1], V[:, t1], W[:, t1]
        u2, v2, w2 = U[:, t2], V[:, t2], W[:, t2]
        
        # For the sum a*u1⊗v1⊗w1 + b*u2⊗v2⊗w2 to be rank-1,
        # we need specific relationships. One case: if (u1,v1,w1) = c*(u2,v2,w2)
        # then they're already the same direction.
        
        # Check if u1 is proportional to u2
        if np.allclose(u1, 0) or np.allclose(u2, 0):
            return None
            
        # Try to find c such that u1 ≈ c * u2
        # Use least squares
        c_est = np.dot(u1, u2) / np.dot(u2, u2)
        if np.allclose(u1, c_est * u2, atol=1e-10):
            # Check v and w
            cv_est = np.dot(v1, v2) / np.dot(v2, v2) if np.any(v2 != 0) else 0
            cw_est = np.dot(w1, w2) / np.dot(w2, w2) if np.any(w2 != 0) else 0
            if np.allclose(v1, cv_est * v2, atol=1e-10) and np.allclose(w1, cw_est * w2, atol=1e-10):
                # They're proportional - can merge by taking one
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                # Replace t1 with the sum, remove t2
                new_U[:, t1] = u1 + u2
                new_V[:, t1] = v1 + v2
                new_W[:, t1] = w1 + w2
                # Remove column t2
                new_U = np.delete(new_U, t2, axis=1)
                new_V = np.delete(new_V, t2, axis=1)
                new_W = np.delete(new_W, t2, axis=1)
                return (new_U, new_V, new_W)
        
        return None

    def local_search(U, V, W, max_iterations=1000):
        """Local search: repeatedly try to merge pairs of columns."""
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = current_U.shape[1]
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            r = current_U.shape[1]
            
            # Try all pairs
            for t1 in range(r):
                if time.time() > deadline:
                    break
                for t2 in range(t1 + 1, r):
                    result = try_merge_pair(current_U, current_V, current_W, t1, t2)
                    if result is not None:
                        new_U, new_V, new_W = result
                        # Verify the merge
                        if verify_numeric(new_U, new_V, new_W, tol=1e-8):
                            current_U, current_V, current_W = new_U, new_V, new_W
                            current_rank -= 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_U, current_V, current_W, current_rank

    # Run local search from schoolbook
    if time.time() < deadline - 10:
        U_ls, V_ls, W_ls, rank_ls = local_search(U0, V0, W0, max_iterations=500)
        if rank_ls < best_rank:
            # Verify with exact arithmetic
            if verify_exact(U_ls, V_ls, W_ls):
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                best_rank = rank_ls
    
    # 3. Try random restarts with structured initialization
    # Use GF(2) parity to guide support patterns
    
    def gf2_parity_support(U, V, W):
        """Compute the GF(2) parity pattern of the decomposition.
        
        For each term t, compute the parity (mod 2) of the entries.
        The sum of parities should match the target tensor's parity (all 1s).
        """
        r = U.shape[1]
        # Target tensor has all entries = 1, so parity is 1 for all (i,j,k)
        # We want sum_t (U[i,j,t] mod 2) * (V[j,k,t] mod 2) * (W[k,i,t] mod 2) ≡ 1 (mod 2)
        
        # This is a system of equations over GF(2)
        # For screening, we just check if the parity pattern is "reasonable"
        parity_sum = np.zeros((n, m, p), dtype=int)
        for t in range(r):
            u_parity = (U[:, t] != 0).astype(int) % 2
            v_parity = (V[:, t] != 0).astype(int) % 2
            w_parity = (W[:, t] != 0).astype(int) % 2
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        parity_sum[i, j, k] = (parity_sum[i, j, k] + 
                                              u_parity[i*m+j] * v_parity[j*p+k] * w_parity[k*n+i]) % 2
        
        # Check if parity matches target (all 1s)
        if not np.all(parity_sum == 1):
            return False
        return True

    # Random restart search
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Try rank best_rank - 1 first, then lower
        for target_rank in range(min(best_rank - 1, 55), max(20, best_rank - 10), -1):
            if time.time() > deadline - 5:
                break
            
            # Initialize with sparse random patterns guided by structure
            # Use the fact that each (i,j) pair in U should connect to
            # specific (j,k) pairs in V and (k,i) pairs in W
            
            U_rand = np.zeros((dim_u, target_rank))
            V_rand = np.zeros((dim_v, target_rank))
            W_rand = np.zeros((dim_w, target_rank))
            
            # For each term, assign a random "path" i -> j -> k
            for t in range(target_rank):
                # Random i, j, k
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                
                # Set entries with small probability of additional connections
                U_rand[i*m+j, t] = random.choice([-1, 1])
                V_rand[j*p+k, t] = random.choice([-1, 1])
                W_rand[k*n+i, t] = random.choice([-1, 1])
                
                # With some probability, add extra connections
                if random.random() < 0.3:
                    i2 = random.randint(0, n-1)
                    j2 = random.randint(0, m-1)
                    k2 = random.randint(0, p-1)
                    U_rand[i2*m+j2, t] += random.choice([-1, 1])
                    V_rand[j2*p+k2, t] += random.choice([-1, 1])
                    W_rand[k2*n+i2, t] += random.choice([-1, 1])
            
            # Clip to {-1, 0, 1} for simplicity
            U_rand = np.sign(U_rand)
            V_rand = np.sign(V_rand)
            W_rand = np.sign(W_rand)
            
            # Check parity support
            if gf2_parity_support(U_rand, V_rand, W_rand):
                # Try to refine with local search
                U_ref, V_ref, W_ref, rank_ref = local_search(U_rand, V_rand, W_rand, max_iterations=100)
                if rank_ref < best_rank:
                    if verify_numeric(U_ref, V_ref, W_ref, tol=1e-8):
                        if verify_exact(U_ref, V_ref, W_ref):
                            best_U, best_V, best_W = U_ref, V_ref, W_ref
                            best_rank = rank_ref
                            break
            
            if time.time() > deadline - 5:
                break
    
    # Final verification
    if not verify_numeric(best_U, best_V, best_W, tol=1e-10):
        # Fall back to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = schoolbook_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
