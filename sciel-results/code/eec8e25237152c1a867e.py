"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T
    # T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # We will search for a rank R decomposition
    # Start with a lower bound and increase if not found
    # Lower bound for 2x3x8 is likely around 30-40
    
    # Define the set of non-zero indices in T
    # t_indices: list of (u, v, w) where T[u,v,w] = 1
    t_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                t_indices.append((u, v, w))
                
    # Total number of non-zero entries in T is n*m*p
    num_nonzero = len(t_indices)
    
    # We want to find U, V, W such that sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
    # This is equivalent to finding a low rank tensor approximation that is exact on these entries and 0 elsewhere.
    # Note: The "elsewhere" constraints are many (dim_u*dim_v*dim_w - num_nonzero).
    # However, if we ensure the solution is sparse or structured, it might work.
    # Actually, the problem implies we just need to match the tensor T.
    # The tensor T is fixed. We need to represent it as a sum of rank-1 tensors.
    
    # Strategy: Alternating Least Squares (ALS) on the full tensor space?
    # That's too big. 
    # Instead, we can treat this as a system of equations.
    # For each (u,v,w) in the support of T, sum_r U[u,r] V[v,r] W[w,r] = 1.
    # For all other (u,v,w), sum_r U[u,r] V[v,r] W[w,r] = 0.
    
    # This is a huge system.
    # Let's try a smaller rank first, e.g., 30.
    
    def try_rank(R, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize with random small integers
        U = np.random.randint(-2, 3, size=(dim_u, R))
        V = np.random.randint(-2, 3, size=(dim_v, R))
        W = np.random.randint(-2, 3, size=(dim_w, R))
        
        # We will use a local search / hill climbing approach
        # Objective: Minimize the number of errors in the tensor reconstruction
        
        # To make it efficient, we only check the non-zero entries of T?
        # No, we must also ensure zeros are zero.
        # But usually, if we fit the non-zeros with small integers, the zeros might be close to 0.
        # Let's check all entries? 
        # dim_u * dim_v * dim_w is 6 * 24 * 16 = 2304 entries.
        # This is small enough to check all.
        
        def calculate_error(U, V, W):
            # Compute tensor T_reconstructed
            # T_rec[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
            # We can compute this efficiently by iterating over r
            T_rec = np.zeros((dim_u, dim_v, dim_w), dtype=int)
            for r in range(R):
                # Outer product of U[:,r], V[:,r], W[:,r]
                # This is a rank-1 tensor
                # T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                # But einsum might be slow if called R times in a loop for many iterations.
                # Let's use broadcasting
                T_rec += np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
            
            # Target tensor T
            # T is 1 at t_indices, 0 elsewhere
            # We calculate the number of mismatches
            errors = 0
            # Check non-zero entries
            for u, v, w in t_indices:
                if T_rec[u, v, w] != 1:
                    errors += 1
            
            # Check zero entries (sample or all?)
            # With 2304 entries, checking all is fine.
            # Create a mask of zero entries
            # It's easier to just check all entries against a target array
            T_target = np.zeros((dim_u, dim_v, dim_w), dtype=int)
            for u, v, w in t_indices:
                T_target[u, v, w] = 1
                
            # Count mismatches
            # T_rec might have large values if we don't constrain them
            # But we are using integer entries.
            # We want T_rec == T_target
            # Mismatches = sum(T_rec != T_target)
            # This is expensive to compute from scratch every time.
            # Let's just return the sum of squared errors for optimization, 
            # but we need exactness.
            # For the search, let's minimize the Hamming distance.
            
            # Optimization: Only compute mismatch count
            # We can do this by iterating
            # But 2304 * R operations is small.
            
            # Let's just return the number of mismatches
            return np.sum(T_rec != T_target)

        best_error = calculate_error(U, V, W)
        
        # Local Search
        # Try to change one entry in U, V, or W to reduce error
        # This is slow.
        
        # Alternative: ALS
        # Fix V, W. Solve for U.
        # The equations are: sum_r U[u,r] (V[v,r] W[w,r]) = T_target[u,v,w]
        # For a fixed u, we have a linear system for U[u, :]?
        # No, U[u,r] is coupled with V[v,r] and W[w,r].
        
        # Let's stick to a simpler heuristic: 
        # 1. Start with schoolbook (Rank 48).
        # 2. Try to reduce rank by finding linear dependencies or merging terms.
        
        # Actually, let's just return a known good rank if we can find one via search.
        # If search fails, return schoolbook.
        
        # Given the time constraint, a robust fallback is better.
        # But the task asks for improvement.
        # Let's try a specific construction for 2x3x8.
        
        # We can use the fact that 2x3x8 can be decomposed using Strassen on blocks?
        # A = [A11 A12] (2x2, 2x1)
        # B = [B11; B21; B31] (2x8, 1x8, 1x8)
        
        # This is complex to code dynamically.
        
        # Let's try the ALS approach properly.
        # It is known that ALS can find low-rank decompositions.
        
        # We will run ALS for a few iterations.
        
        # Initialize
        U = np.random.rand(dim_u, R)
        V = np.random.rand(dim_v, R)
        W = np.random.rand(dim_w, R)
        
        # T_target as a flat vector?
        # T_flat[u * dim_v * dim_w + v * dim_w + w] = T_target[u,v,w]
        
        # For each step, we solve a linear least squares problem.
        # To solve for U:
        # We need to minimize || sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] - T ||_F
        # This is linear in U.
        # We can construct a large matrix A_U such that A_U @ U_flat = T_flat.
        # But A_U is huge (2304 x (dim_u * R)).
        # dim_u * R = 6 * 30 = 180.
        # 2304 x 180 is manageable.
        
        # Let's build the design matrix for U.
        # For each entry (u,v,w) in T, the coefficient of U[u', r] is:
        # Non-zero only if u' == u.
        # Then the term is U[u, r] * V[v, r] * W[w, r].
        # So for a fixed u, the variables are U[u, 0], ..., U[u, R-1].
        # The equation for all (v,w) given u is:
        # sum_r U[u, r] * V[v, r] * W[w, r] = T[u, v, w]
        
        # We can solve for each row of U independently!
        # For each u in range(dim_u):
        #   Construct matrix M_u of size (dim_v * dim_w) x R
        #   M_u[v * dim_w + w, r] = V[v, r] * W[w, r]
        #   Vector b_u of size (dim_v * dim_w)
        #   b_u[v * dim_w + w] = T_target[u, v, w]
        #   Solve M_u @ U[u, :] = b_u (least squares)
        
        # This is efficient!
        
        def solve_U(U, V, W):
            new_U = np.zeros_like(U)
            for u in range(dim_u):
                # Build M_u
                # M_u shape: (dim_v * dim_w, R)
                M_u = np.zeros((dim_v * dim_w, R))
                b_u = np.zeros(dim_v * dim_w)
                
                # Fill M_u and b_u
                # T_target is sparse, but we can just iterate
                # Actually, T_target[u, :, :] is mostly 0.
                # We only need to fill the non-zero parts for b_u?
                # No, least squares needs all constraints.
                # But the zero constraints are important.
                
                # Iterate over v, w
                for v in range(dim_v):
                    for w in range(dim_w):
                        idx = v * dim_w + w
                        b_u[idx] = T_target[u, v, w]
                        # M_u[idx, :] = V[v, :] * W[w, :]
                        M_u[idx, :] = V[v, :] * W[w, :]
                        
                # Solve M_u @ x = b_u
                # Use np.linalg.lstsq
                sol, residuals, rank, sv = np.linalg.lstsq(M_u, b_u, rcond=None)
                new_U[u, :] = sol
                
            return new_U
            
        def solve_V(U, V, W):
            new_V = np.zeros_like(V)
            for v in range(dim_v):
                M_v = np.zeros((dim_u * dim_w, R))
                b_v = np.zeros(dim_u * dim_w)
                
                for u in range(dim_u):
                    for w in range(dim_w):
                        idx = u * dim_w + w
                        b_v[idx] = T_target[u, v, w]
                        M_v[idx, :] = U[u, :] * W[w, :]
                        
                sol, residuals, rank, sv = np.linalg.lstsq(M_v, b_v, rcond=None)
                new_V[v, :] = sol
                
            return new_V
            
        def solve_W(U, V, W):
            new_W = np.zeros_like(W)
            for w in range(dim_w):
                M_w = np.zeros((dim_u * dim_v, R))
                b_w = np.zeros(dim_u * dim_v)
                
                for u in range(dim_u):
                    for v in range(dim_v):
                        idx = u * dim_v + v
                        b_w[idx] = T_target[u, v, w]
                        M_w[idx, :] = U[u, :] * V[v, :]
                        
                sol, residuals, rank, sv = np.linalg.lstsq(M_w, b_w, rcond=None)
                new_W[w, :] = sol
                
            return new_W

        # Run ALS
        for _ in range(10):
            U = solve_U(U, V, W)
            V = solve_V(U, V, W)
            W = solve_W(U, V, W)
            
        # After ALS, snap to half-integers
        # And check for exactness
        # We need integer/half-integer entries.
        # Multiply by 2, round, divide by 2.
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Check error
        T_rec = np.zeros((dim_u, dim_v, dim_w), dtype=int)
        for r in range(R):
            T_rec += np.multiply.outer(np.multiply.outer(U_int[:, r].astype(int), V_int[:, r].astype(int)), W_int[:, r].astype(int))
            
        # Note: U_int, V_int, W_int are floats (half-integers).
        # The product might be float.
        # We need exact integer arithmetic for the check?
        # The problem says "integer/half-integer entries".
        # If we use half-integers, the products are quarters.
        # The sum must be exactly 1 or 0.
        
        # Let's convert to Fractions for exact check
        U_frac = [[Fraction(x).limit_denominator(2) for x in U_int[i]] for i in range(dim_u)]
        V_frac = [[Fraction(x).limit_denominator(2) for x in V_int[i]] for i in range(dim_v)]
        W_frac = [[Fraction(x).limit_denominator(2) for x in W_int[i]] for i in range(dim_w)]
        
        # Check
        is_exact = True
        for u, v, w in t_indices:
            s = sum(U_frac[u][r] * V_frac[v][r] * W_frac[w][r] for r in range(R))
            if s != 1:
                is_exact = False
                break
        
        if is_exact:
            # Check zeros (sample a few or all)
            # We should check all to be safe
            for u in range(dim_u):
                for v in range(dim_v):
                    for w in range(dim_w):
                        if (u, v, w) in t_indices:
                            continue
                        s = sum(U_frac[u][r] * V_frac[v][r] * W_frac[w][r] for r in range(R))
                        if s != 0:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
                    
        if is_exact:
            return {
                "U": [[float(x) for x in row] for row in U_frac],
                "V": [[float(x) for x in row] for row in V_frac],
                "W": [[float(x) for x in row] for row in W_frac]
            }
            
        return None

    # Try ranks from low to high
    # Lower bound is roughly n*m*p / 2? No.
    # Let's try 30 to 40.
    for R in range(30, 49):
        # Try a few seeds
        for seed in range(5):
            res = try_rank(R, seed)
            if res:
                return res
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
