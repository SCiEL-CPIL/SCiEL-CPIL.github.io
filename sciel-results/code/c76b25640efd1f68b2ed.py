"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct the target tensor T
    # T[a, b, k] = 1 if a=i*m+j, b=j*p+k, k=k*n+i for some i,j,k
    # Otherwise 0.
    # We represent T as a list of slices T_k, each of size nm x mp.
    T_slices = []
    for k in range(pn):
        slice_mat = np.zeros((nm, mp), dtype=int)
        # Iterate over all i, j, k_orig to populate
        for i in range(n):
            for j in range(m):
                for k_orig in range(p):
                    a = i * m + j
                    b = j * p + k_orig
                    k_idx = k_orig * n + i
                    if k_idx == k:
                        slice_mat[a, b] = 1
        T_slices.append(slice_mat)
        
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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

    def find_rank_decomposition(target_rank):
        """
        Try to find a decomposition of rank target_rank.
        Strategy:
        1. Generate random U and V matrices with small integer entries.
        2. Check if all T_slices lie in the span of the outer products U_c ⊗ V_c.
        3. If yes, solve for W.
        """
        # We need to check if T_slices are in the span of M_c = u_c ⊗ v_c
        # This is equivalent to checking if the matrix formed by stacking the vectorized
        # T_slices has rank <= target_rank and is consistent with the span.
        
        # More simply:
        # Construct a matrix S of size (pn * nm * mp) x target_rank? No.
        # Let's build the matrix of basis vectors M_c.
        # Each M_c is nm x mp. Flatten it to a vector of length nm*mp.
        # We want to check if each flattened T_slice is in the column space of the matrix B
        # where columns of B are flattened M_c.
        
        max_attempts = 50
        for attempt in range(max_attempts):
            # Generate random U and V with small integers
            # Using -1, 0, 1 to keep numbers small
            U_rand = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)])
            V_rand = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)])
            
            # Construct B: columns are vec(U_rand[:, c] ⊗ V_rand[:, c])
            # B is (nm * mp) x target_rank
            B = np.zeros((nm * mp, target_rank), dtype=int)
            for c in range(target_rank):
                outer = np.outer(U_rand[:, c], V_rand[:, c])
                B[:, c] = outer.flatten()
            
            # Check if all T_slices are in the column space of B
            # We can do this by checking if B^+ B T_slice == T_slice
            # Or by forming an augmented matrix and checking rank.
            
            # To be efficient, we can check if the rank of [B | T_slice_vec] is equal to rank(B)
            # for all slices.
            
            valid = True
            W_coeffs = []
            
            # Calculate rank of B using SVD (over reals for speed, then verify exactly)
            # Since we need exactness, we should use rational arithmetic if possible,
            # but for small integers, float SVD might be sufficient for checking,
            # followed by exact verification.
            
            # Let's use exact rational arithmetic for the linear system solution.
            # We need to solve B W_slice = T_slice_vec for each slice.
            # This requires B to have full column rank.
            
            # Check rank of B
            # We can try to solve the system for the first slice to get W_coeffs
            # and then verify for all slices.
            
            T_vecs = [s.flatten() for s in T_slices]
            
            # Try to solve B x = T_vecs[0]
            # Using fractions
            B_frac = [[Fraction(x) for x in row] for row in B]
            T_frac = [Fraction(x) for x in T_vecs[0]]
            
            # Gaussian elimination to solve B x = T
            # If B is not full rank, this might fail or have multiple solutions.
            # We assume generic B has full rank.
            
            try:
                # Simple Gaussian elimination for B (nm*mp x target_rank)
                # We want to find x such that B x = T
                # This is an overdetermined system if nm*mp > target_rank.
                # We can pick target_rank rows to form a square matrix.
                
                # Pick first target_rank rows
                B_sub = B_frac[:target_rank]
                T_sub = T_frac[:target_rank]
                
                # Solve B_sub x = T_sub
                # Invert B_sub?
                # We can use Cramer's rule or Gaussian elimination.
                
                # Let's use a helper to solve linear system with Fractions
                def solve_linear_system(A, b):
                    n = len(A)
                    # Augmented matrix
                    M = [row[:] + [b[i]] for i, row in enumerate(A)]
                    # Gaussian elimination
                    for col in range(n):
                        # Find pivot
                        pivot_row = -1
                        for row in range(col, n):
                            if M[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row == -1:
                            return None # Singular
                        # Swap
                        M[col], M[pivot_row] = M[pivot_row], M[col]
                        # Normalize
                        pivot_val = M[col][col]
                        M[col] = [x / pivot_val for x in M[col]]
                        # Eliminate
                        for row in range(n):
                            if row != col:
                                factor = M[row][col]
                                M[row] = [M[row][j] - factor * M[col][j] for j in range(n + 1)]
                    # Solution is in last column
                    return [M[i][n] for i in range(n)]

                x_sol = solve_linear_system(B_sub, T_sub)
                if x_sol is None:
                    continue
                
                # Verify x_sol against all rows of B and all slices
                valid = True
                W_candidate = [[0] * target_rank for _ in range(pn)]
                
                # First, verify that B x_sol matches T_vecs[0] completely
                for i in range(nm * mp):
                    dot = sum(B_frac[i][c] * x_sol[c] for c in range(target_rank))
                    if dot != T_frac[i]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Now check other slices
                for k in range(1, pn):
                    T_k_frac = [Fraction(x) for x in T_vecs[k]]
                    for i in range(nm * mp):
                        dot = sum(B_frac[i][c] * x_sol[c] for c in range(target_rank))
                        # Wait, x_sol is for the first slice.
                        # The coefficients W[k, c] might be different for each slice k.
                        # The condition is: T_k = sum_c W[k, c] M_c
                        # So we need to solve B W_k = T_k for each k.
                        
                        # My previous logic was flawed. x_sol depends on k.
                        # We need to solve for each k.
                        pass
                
                # Corrected logic:
                # For each slice k, solve B W_k = T_k.
                # We already solved for k=0.
                # Let's solve for all k.
                
                W_solution = []
                all_valid = True
                for k in range(pn):
                    T_k_frac = [Fraction(x) for x in T_vecs[k]]
                    x_k = solve_linear_system(B_sub, T_k_frac)
                    if x_k is None:
                        all_valid = False
                        break
                    # Verify against full B
                    for i in range(nm * mp):
                        dot = sum(B_frac[i][c] * x_k[c] for c in range(target_rank))
                        if dot != T_k_frac[i]:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                    W_solution.append(x_k)
                
                if all_valid:
                    # We found a solution!
                    # Convert to lists of integers or fractions
                    U_out = [[int(U_rand[i][c]) if U_rand[i][c] == int(U_rand[i][c]) else Fraction(U_rand[i][c]) for c in range(target_rank)] for i in range(nm)]
                    V_out = [[int(V_rand[i][c]) if V_rand[i][c] == int(V_rand[i][c]) else Fraction(V_rand[i][c]) for c in range(target_rank)] for i in range(mp)]
                    W_out = [[int(x) if x == int(x) else x for x in row] for row in W_solution]
                    
                    return {"U": U_out, "V": V_out, "W": W_out}
                    
            except Exception:
                continue
                
        return None

    # Try ranks from lower bound up to schoolbook
    lower_bound = max(nm, mp, pn)
    schoolbook_rank = nm * mp * pn // (mp) # n*m*p? No, n*m*p
    schoolbook_rank = n * m * p
    
    # We know rank <= 42.
    # Try 41, 40, ... down to lower_bound.
    # To save time, we can start with 41.
    
    for r in range(41, lower_bound - 1, -1):
        res = find_rank_decomposition(r)
        if res is not None:
            return res
            
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
