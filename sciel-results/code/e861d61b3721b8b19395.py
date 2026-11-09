"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit for construct()

    def make_seed():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check if U, V, W form a valid tensor decomposition."""
        r = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)

        # Check that entries are integer or half-integer
        Ua2 = Ua * 2
        Va2 = Va * 2
        Wa2 = Wa * 2
        if not np.allclose(Ua2, np.round(Ua2), atol=1e-6):
            return False
        if not np.allclose(Va2, np.round(Va2), atol=1e-6):
            return False
        if not np.allclose(Wa2, np.round(Wa2), atol=1e-6):
            return False

        Ur = Ua.reshape(n, m, r)
        Vr = Va.reshape(m, p, r)
        Wr = Wa.reshape(p, n, r)
        T = np.einsum('ijl,jkl,kin->ijk', Ur, Vr, Wr)

        if not np.allclose(T, np.ones((n, m, p)), atol=1e-4):
            return False
        return True

    def check_half_integer(a, tol=1e-6):
        a2 = a * 2
        return np.allclose(a2, np.round(a2), atol=tol)

    def compute_T(U, V, W):
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        return np.einsum('ijl,jkl,kin->ijk', Ur, Vr, Wr)

    def time_left():
        return deadline - time.time()

    # Start with the trivial rank-90 decomposition
    seed = make_seed()
    best_result = seed
    best_rank = n * m * p

    # Strategy: Use structured search with MILP-inspired approach
    # For each rank r, we need to find U, V, W such that:
    # sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1 for all (i,j,k)
    # with entries in {0, ±0.5, ±1, ±1.5, ...}
    
    # Key insight: Fix W to have simple structure (e.g., W[k*n+i,l] in {0,1}),
    # then the problem becomes: for each l, the product U[:,l] ⊗ V[:,l] 
    # contributes to the tensor. We can search over possible patterns.
    
    # Alternative: Use the fact that we can permute columns of U,V,W.
    # Try rank 89, 88, ... by removing one term and checking if we can fix it.
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    rng = np.random.default_rng(42)

    # For small ranks, do exhaustive search over sparse patterns
    for target_rank in [89, 88, 87, 86, 85]:
        if time_left() < 20:
            break

        found = False

        # Strategy: Start from rank-90, remove one term, try to compensate
        # by adjusting other terms with half-integer values
        
        # The trivial decomposition has term (i,j,k) at index i*m*p + j*p + k
        # Removing term (i0,j0,k0) means T[i0,j0,k0] loses 1.
        # We need other terms to contribute +1 to T[i0,j0,k0] without affecting others.
        
        # A term l contributes to T[i,j,k] iff U[i*m+j,l]≠0, V[j*p+k,l]≠0, W[k*n+i,l]≠0
        # For the trivial term (i1,j1,k1), it contributes to T[i1,j1,k1] only.
        # So if we modify term (i1,j1,k1) to also have U[i0*m+j0,l]≠0, etc.,
        # it will contribute to T[i0,j0,k0] as well.
        
        # Let's try: for each missing term (i0,j0,k0), try to "split" it across
        # multiple existing terms by adding half-integer contributions.
        
        for i0 in range(n):
            if time_left() < 15:
                break
            for j0 in range(m):
                if time_left() < 15:
                    break
                for k0 in range(p):
                    if time_left() < 15:
                        break

                    # Try to find a set of existing terms that can compensate
                    # For simplicity, try modifying 1-3 terms
                    
                    # Create base rank-89 matrix (remove term i0,j0,k0)
                    U_base = np.zeros((n * m, target_rank))
                    V_base = np.zeros((m * p, target_rank))
                    W_base = np.zeros((p * n, target_rank))
                    
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i == i0 and j == j0 and k == k0:
                                    continue
                                U_base[i * m + j, col] = 1
                                V_base[j * p + k, col] = 1
                                W_base[k * n + i, col] = 1
                                col += 1
                    
                    # Current T has T[i0,j0,k0] = 0, others = 1
                    # We need to add 1 to T[i0,j0,k0]
                    
                    # Try modifying up to 3 terms
                    # A term (i1,j1,k1) can be modified to contribute to (i0,j0,k0)
                    # if we set U[i0*m+j0, l] = a, V[j0*p+k0, l] = b, W[k0*n+i0, l] = c
                    # such that a*b*c = 1 (to add 1 to T[i0,j0,k0])
                    # But this also affects T[i1,j1,k1] and other entries.
                    
                    # This is complex. Let's try a different approach:
                    # Random local search around the trivial solution
                    
                    for attempt in range(200):
                        if time_left() < 10:
                            break
                        
                        # Start from base and perturb
                        U = U_base.copy()
                        V = V_base.copy()
                        W = W_base.copy()
                        
                        # Perturb a few random entries
                        num_perturb = rng.integers(1, 4)
                        for _ in range(num_perturb):
                            # Pick a random column
                            l = rng.integers(0, target_rank)
                            # Pick random rows to modify
                            u_row = rng.integers(0, n * m)
                            v_row = rng.integers(0, m * p)
                            w_row = rng.integers(0, p * n)
                            
                            # Try setting to half-integer values
                            U[u_row, l] = rng.choice(values)
                            V[v_row, l] = rng.choice(values)
                            W[w_row, l] = rng.choice(values)
                        
                        # Check if it works
                        T = compute_T(U, V, W)
                        if np.allclose(T, np.ones((n, m, p)), atol=1e-4):
                            if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                                U_list = [[float(x) for x in row] for row in U]
                                V_list = [[float(x) for x in row] for row in V]
                                W_list = [[float(x) for x in row] for row in W]
                                if validate(U_list, V_list, W_list):
                                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                                    best_rank = target_rank
                                    found = True
                                    break
                    
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break

    # If we haven't found anything better, try a completely different approach:
    # Use the structure that for small matrices, there might be known lower bounds.
    # For 3x5*5x6, the schoolbook rank is 90. The lower bound is max(3*5, 5*6, 3*6) = 30.
    # But finding an exact decomposition is hard.
    
    # Try a more systematic search: for rank 89, fix the structure of W to be binary
    # and search for U, V that satisfy the equations.
    
    if best_rank == 90 and time_left() > 30:
        # Try rank 89 with structured W
        target_rank = 89
        
        # W has shape (p*n, rank) = (18, 89)
        # Try W with entries in {0, 1}
        # For each column l, W[:, l] is a binary vector of length 18
        
        # The constraint is: sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1
        # If W[k*n+i,l] = 1, then U[i*m+j,l] * V[j*p+k,l] must contribute to the sum.
        
        # This is still complex. Let's try a heuristic: 
        # For each (i,j,k), we need sum over l where W[k*n+i,l]=1 of U[i*m+j,l]*V[j*p+k,l] = 1
        
        # Let's try to construct W such that for each (k,i), the set of l with W[k*n+i,l]=1
        # is small, and then solve for U and V.
        
        # Given time constraints, let's just return the best we have.
        pass

    return best_result
# EVOLVE-BLOCK-END
