import random
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            for j in range(m):
                outer = np.outer(u_mat[:, j], v_mat[j, :])
                T[:, j, :] += outer * w_mat.T
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True
    
    def is_half_integer(x, tol=1e-9):
        """Check if x is an integer or half-integer."""
        if abs(x - round(x)) < tol:
            return True
        if abs(2*x - round(2*x)) < tol:
            return True
        return False
    
    def find_half_integer_combination(coeffs, target=1.0, max_terms=2, search_range=10):
        """Find a combination of coeffs with half-integer weights that sums to target."""
        nz = [(r, c) for r, c in enumerate(coeffs) if abs(c) > 1e-9]
        if not nz:
            return None
        
        # Try single term
        for r, c in nz:
            w = target / c
            if is_half_integer(w):
                result = {r: w}
                return result
        
        if max_terms >= 2:
            # Try two terms
            for idx1 in range(len(nz)):
                for idx2 in range(idx1 + 1, len(nz)):
                    r1, c1 = nz[idx1]
                    r2, c2 = nz[idx2]
                    # Find a, b half-integers such that a*c1 + b*c2 = target
                    # a*c1 + b*c2 = 1
                    # Try small half-integer values
                    for a_num in range(-search_range*2, search_range*2+1):
                        a = a_num / 2.0
                        if a == 0:
                            continue
                        rem = target - a * c1
                        if abs(rem) < 1e-9:
                            continue
                        b = rem / c2
                        if is_half_integer(b):
                            result = {r1: a, r2: b}
                            return result
        return None
    
    def try_rank_r_structured(rank, time_limit):
        """
        Try a structured decomposition using the subspace method.
        
        Key idea: Partition the j-index (m=4) into groups and use polynomial
        evaluation/interpolation to reduce the rank.
        
        For the subspace method with m=4:
        - We can write the j-sum as evaluation of degree-3 polynomials
        - Using 4 evaluation points, we get the full sum
        - But we can sometimes do better by exploiting structure in i and k
        
        Strategy: Use a block structure where we group (i,k) pairs.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Strategy 1: Direct structured approach using polynomial basis
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        # Use the fact that we can separate the j-dependence
        # Let U[i,j,r] = sum_a A[i,a,r] * B[j,a,r] (separable in i,j)
        # Let V[j,k,r] = sum_b C[j,b,r] * D[k,b,r] (separable in j,k)
        
        # Simpler: Try rank-r with specific structure
        # For each r, U[:,r] is in R^(n*m), V[:,r] in R^(m*p), W[:,r] in R^(p*n)
        
        # Approach: Use random sparse matrices and solve for W
        for trial in range(100):
            if time.time() - start_time > time_limit:
                return None
            
            # Generate sparse random U, V
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Make U sparse: each row has 1-2 nonzeros
            for idx in range(n*m):
                nz_count = random.choice([1, 1, 2])
                cols = random.sample(range(rank), min(nz_count, rank))
                for c in cols:
                    U[idx, c] = random.choice([-1.0, 1.0])
            
            for idx in range(m*p):
                nz_count = random.choice([1, 1, 2])
                cols = random.sample(range(rank), min(nz_count, rank))
                for c in cols:
                    V[idx, c] = random.choice([-1.0, 1.0])
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # For each (k,i), find W[k*n+i, :] such that M[k*n+i, :] @ W[k*n+i, :] = 1
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                coeffs = M[idx, :]
                result = find_half_integer_combination(coeffs, 1.0, max_terms=2, search_range=8)
                if result is None:
                    valid = False
                    break
                for r, w_val in result.items():
                    W[idx, r] = w_val
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_polynomial_interpolation(rank, time_limit):
        """
        Use polynomial interpolation in the j-index.
        
        For m=4, we have 4 values j=0,1,2,3.
        Any function of j can be represented by a degree-3 polynomial.
        
        Key insight: If we choose evaluation points cleverly, we can
        reduce the effective rank.
        
        Strategy: Use the fact that for fixed i,k, the sum over j of
        U[i,j,r]*V[j,k,r] can be computed via polynomial evaluation.
        """
        if time.time() - start_time > time_limit:
            return None
        
        # Use Lagrange interpolation basis
        # Points: x_0, x_1, x_2, x_3 for j = 0, 1, 2, 3
        # Let's use x_j = j for simplicity, or better: use symmetric points
        
        # Actually, let's use a different approach:
        # Write the j-sum as: sum_j f(j) * g(j) where f depends on i and g on k
        # If f and g are polynomials of degree <= 2, we only need 3 points
        
        # But f(j) = U[i,j,r] and g(j) = V[j,k,r] are arbitrary, not polynomials.
        
        # Alternative: Use the structure that we can have rank = n*p + something
        # For n=3, p=7, that's 21, which is still > 17
        
        # Let's try a more direct approach with structured matrices
        
        # Use circulant-like structure
        for offset in range(rank):
            if time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Fill U with a pattern
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        # U[i,j,r] = 1 if (j + i) % rank == r
                        if (j + i) % rank == r:
                            U[i*m + j, r] = 1.0
            
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        if (k + j) % rank == r:
                            V[j*p + k, r] = 1.0
            
            # Now solve for W
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                coeffs = M[idx, :]
                result = find_half_integer_combination(coeffs, 1.0, max_terms=2, search_range=6)
                if result is None:
                    valid = False
                    break
                for r, w_val in result.items():
                    W[idx, r] = w_val
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try decreasing ranks
    # Lower bound: rank >= max(n, m, p) = 7, but realistically much higher
    # For 3x4x7, known bounds suggest rank around 20-30
    
    for rank in range(15, 84):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try structured approach
        result = try_rank_r_structured(rank, DEADLINE - 5)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break
        
        # Try polynomial interpolation
        result = try_polynomial_interpolation(rank, DEADLINE - 5)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
