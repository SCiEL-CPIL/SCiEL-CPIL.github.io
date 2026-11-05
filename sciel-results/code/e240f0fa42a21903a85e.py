"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    # Equation: sum_c U[i*m+j,c] * V[l*p+k,c] * W[k*n+i,c] = delta_{j,l}
    # Multiply by 2: sum_c U[i*m+j,c] * V[l*p+k,c] * (2*W[k*n+i,c]) = 2 * delta_{j,l}
    
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_2arr = np.zeros((p * n, rank), dtype=np.int64)
    for k in range(p):
        for i in range(n):
            row = W[k * n + i]
            for c in range(rank):
                val = row[c]
                if isinstance(val, Frac):
                    # val is half-integer, so val * 2 is integer
                    W_2arr[k * n + i, c] = int(val * 2)
                else:
                    W_2arr[k * n + i, c] = int(val) * 2

    for i in range(n):
        for k in range(p):
            # U_sub: (m, rank)
            U_sub = U_arr[i * m : (i + 1) * m, :]
            # V_sub: (m, rank) - rows l*p+k for l in 0..m-1
            V_sub = V_arr[np.arange(m) * p + k, :]
            # W_vec: (rank,)
            W_vec = W_2arr[k * n + i, :]
            
            # Compute sum_c U_sub[j,c] * V_sub[l,c] * W_vec[c] for all j,l
            # Let A[j,c] = U_sub[j,c] * W_vec[c]. Then result[j,l] = sum_c A[j,c] * V_sub[l,c] = A @ V_sub.T
            A = U_sub * W_vec
            result = A @ V_sub.T
            
            # Expected: 2 * I_m
            expected = 2 * np.eye(m, dtype=np.int64)
            
            if not np.array_equal(result, expected):
                return False
    return True

def _schoolbook(n, m, p):
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

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    # Use a wider range for integers to increase chance of small integer solutions
    range_val = 3 
    
    while time.time() < deadline and it < max_iter:
        it += 1
        # Random integer U,V with structured initialization
        U = rng.randint(-range_val, range_val, (nm, rank))
        V = rng.randint(-range_val, range_val, (mp, rank))
        
        # Quick check: ensure no row of U or V is all zeros
        if np.any(np.all(U == 0, axis=1)):
            continue
        if np.any(np.all(V == 0, axis=1)):
            continue
        
        # For each (i,k), check consistency and solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Extract sub-matrices
                U_sub = U[i * m : (i + 1) * m, :]
                V_sub = V[np.arange(m) * p + k, :]
                
                # Build M: M[j*m+l, c] = U_sub[j, c] * V_sub[l, c]
                # M = (U_sub[:, None, :] * V_sub[None, :, :]).reshape(m*m, rank)
                M = (U_sub[:, np.newaxis, :] * V_sub[np.newaxis, :, :]).reshape(m * m, rank)
                
                # RHS
                b = np.zeros(m * m, dtype=float)
                for j in range(m):
                    b[j * m + j] = 1.0
                
                # Check consistency and solve
                # We want to check if solution is half-integer.
                
                # Calculate rank of M and M_aug
                M_aug = np.column_stack([M, b])
                
                # Use np.linalg.solve first. If it fails, check consistency with lstsq.
                try:
                    W_float = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    # Singular. Check consistency.
                    # If consistent, there are infinite solutions. 
                    # We pick the minimum norm solution? Or just check if *any* half-integer solution exists?
                    # This is hard. 
                    # Heuristic: If singular, it's likely a bad candidate for a "nice" decomposition 
                    # unless the null space contains half-integers. 
                    # Given the rarity, we skip singular systems in the fast path.
                    ok = False
                    break
                
                # Check if W_float is close to half-integers
                W_half_int_float = np.round(W_float * 2) / 2
                if not np.allclose(W_float, W_half_int_float, atol=1e-9):
                    ok = False
                    break
                
                # Construct candidate W_vec as Fractions
                W_vec = []
                for val in W_half_int_float:
                    # val is close to k/2
                    k_int = int(round(val * 2))
                    W_vec.append(Frac(k_int, 2))
                
                # Verify exactly
                # We need to check: sum_c U_sub[j,c] * V_sub[l,c] * W_vec[c] == delta_{j,l}
                # Using the integer trick: 2 * sum ... == 2 * delta
                W_2_vec = np.array([int(w * 2) for w in W_vec], dtype=np.int64)
                
                # A = U_sub * W_2_vec (element-wise)
                A = U_sub * W_2_vec
                result = A @ V_sub.T
                expected = 2 * np.eye(m, dtype=np.int64)
                
                if not np.array_equal(result, expected):
                    # Candidate failed exact check. 
                    ok = False
                    break
                
                W[k * n + i] = W_vec
            if not ok:
                break
        
        if not ok:
            continue
        
        # If we passed all (i,k) checks, we have a candidate W.
        # Do a full validation just in case (though the loop above checked all constraints).
        
        if _validate(U, V, W, n, m, p, rank):
            # Convert U, V to lists of lists of ints
            U_list = U.tolist()
            V_list = V.tolist()
            return U_list, V_list, W
    
    return None

def _try_structured_search(rank, n, m, p, deadline, rng):
    """Try structured search based on schoolbook perturbation."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start from schoolbook and perturb
    school = _schoolbook(n, m, p)
    U_init = np.array(school['U'], dtype=np.int64)
    V_init = np.array(school['V'], dtype=np.int64)
    
    # For rank < nm*mp, we need to merge some columns
    # Try random linear combinations of schoolbook columns
    while time.time() < deadline:
        # Randomly select rank columns from schoolbook
        if rank <= nm * mp:
            col_indices = rng.sample(range(nm * mp), rank)
        else:
            col_indices = list(range(nm * mp)) + [0] * (rank - nm * mp)
        
        U = U_init[:, col_indices]
        V = V_init[:, col_indices]
        
        # Add small random perturbations
        if rng.random() < 0.5:
            perturbation = rng.randint(-1, 1, (nm, rank))
            U = U + perturbation
        
        if rng.random() < 0.5:
            perturbation = rng.randint(-1, 1, (mp, rank))
            V = V + perturbation
        
        # Check if valid
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                U_sub = U[i * m : (i + 1) * m, :]
                V_sub = V[np.arange(m) * p + k, :]
                
                M = (U_sub[:, np.newaxis, :] * V_sub[np.newaxis, :, :]).reshape(m * m, rank)
                b = np.zeros(m * m, dtype=float)
                for j in range(m):
                    b[j * m + j] = 1.0
                
                try:
                    W_float = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                
                W_half_int_float = np.round(W_float * 2) / 2
                if not np.allclose(W_float, W_half_int_float, atol=1e-9):
                    ok = False
                    break
                
                W_vec = [Frac(int(round(val * 2)), 2) for val in W_half_int_float]
                
                W_2_vec = np.array([int(w * 2) for w in W_vec], dtype=np.int64)
                A = U_sub * W_2_vec
                result = A @ V_sub.T
                expected = 2 * np.eye(m, dtype=np.int64)
                
                if not np.array_equal(result, expected):
                    ok = False
                    break
                
                W[k * n + i] = W_vec
            
            if not ok:
                break
        
        if not ok:
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            U_list = U.tolist()
            V_list = V.tolist()
            return U_list, V_list, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First try to find a better decomposition
    best_result = _schoolbook(n, m, p)
    
    # Try ranks in increasing order
    # Known bounds for 3x3x4 suggest rank 26 is optimal.
    # We will try 26, 27, 28... up to 35.
    
    target_ranks = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for rank in target_ranks:
        if time.time() > deadline - 5: # Stop if less than 5s left
            break
            
        rng = random.Random(rank * 12345 + 678)
        
        start_rank_time = time.time()
        # Give more time to lower ranks? 
        # Let's give 10 seconds per rank, but if we find a solution, break.
        # If we don't find it, we move on.
        rank_deadline = min(deadline - 2, start_rank_time + 10)
        
        try:
            # First try structured search
            result = _try_structured_search(rank, n, m, p, rank_deadline, rng)
            if result is None:
                # Then try random search
                result = _search_rank(rank, n, m, p, rank_deadline, rng, max_iter=1000000)
            
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                # If we found a rank significantly lower than schoolbook, we can stop.
                if rank < 30:
                    break
        except Exception:
            pass

    return best_result
