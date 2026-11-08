"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    valid_set = set(valid_triples)
    all_triples = list(product(range(nm), range(mp), range(pn)))
    invalid_triples = [t for t in all_triples if t not in valid_set]
    
    def verify(U, V, W, R):
        """Check if the decomposition is exact."""
        for (a, b, c) in valid_triples:
            s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(R))
            if abs(s - 1) > 1e-9:
                return False
        for (a, b, c) in invalid_triples:
            s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(R))
            if abs(s) > 1e-9:
                return False
        return True
    
    def try_rank_linear(R, max_attempts=20):
        """Try to find a decomposition of rank R using linear solve."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random U and V
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            
            # Build the linear system for W
            # For each (a,b,c), we have: sum_r U[a,r]*V[b,r]*W[c,r] = target
            # This is linear in W if we treat W[c,r] as unknowns
            # But it's not quite linear because of the product U[a,r]*V[b,r]
            
            # Actually, for fixed U and V, the equation is:
            # For valid (a,b,c): sum_r (U[a,r]*V[b,r]) * W[c,r] = 1
            # For invalid (a,b,c): sum_r (U[a,r]*V[b,r]) * W[c,r] = 0
            
            # Let M[c, r] = W[c, r]. Then for each (a,b,c):
            # sum_r M[c, r] * U[a,r] * V[b,r] = target(a,b,c)
            
            # This is linear in M! So we can set up a linear system.
            # But the number of equations is nm*mp*pn which is huge.
            # However, many are redundant.
            
            # Let's just solve using the valid and a subset of invalid equations.
            # Actually, let's use all equations but solve least squares.
            
            # Build the system: for each (a,b,c), we have an equation.
            # The unknowns are W[c, r] for c in range(pn), r in range(R).
            # Total unknowns: pn * R
            
            # For each (a,b,c): sum_r W[c,r] * U[a,r] * V[b,r] = target
            # This is: for fixed c, the equation involves only W[c, :].
            # So for each c, we have a separate system!
            
            # For a given c, the equations involving W[c, :] are:
            # For all (a, b) such that (a,b,c) is valid: sum_r W[c,r] * U[a,r] * V[b,r] = 1
            # For all (a, b) such that (a,b,c) is invalid: sum_r W[c,r] * U[a,r] * V[b,r] = 0
            
            # So for each c, we have a system: A_c * W[c, :]^T = b_c
            # where A_c[a_idx, r] = U[a, r] * V[b, r] for the (a,b) pairs that give this c.
            
            # Wait, but c determines k and i (c = k*n + i), and a determines i and j (a = i*m + j),
            # and b determines j and k (b = j*p + k). So for a given c, the valid (a,b) pairs are
            # those where i = c % n, k = c // n, and j can be anything (0 to m-1).
            # a = i*m + j, b = j*p + k.
            
            # So for a given c (with i = c%n, k = c//n):
            # Valid (a,b): for j in range(m): a = i*m+j, b = j*p+k. These give target=1.
            # Invalid (a,b): all other (a,b) pairs. These give target=0.
            
            # But we don't need to include ALL invalid pairs. We just need enough to determine W[c,:].
            # The number of valid pairs for a given c is m = 4.
            # The number of unknowns is R.
            # So we need at least R equations. We have 4 valid equations and many invalid ones.
            
            # Let's use the 4 valid equations plus (R-4) invalid equations (if R>4).
            
            success = True
            W = np.zeros((pn, R))
            
            for c in range(pn):
                i_c = c % n
                k_c = c // n
                
                # Build the system for this c
                # Valid equations
                valid_eqs = []
                for j in range(m):
                    a = i_c * m + j
                    b = j * p + k_c
                    row = np.array([U[a, r] * V[b, r] for r in range(R)])
                    valid_eqs.append((row, 1.0))
                
                # Invalid equations: pick some
                invalid_eqs = []
                for a in range(nm):
                    i_a = a // m
                    j_a = a % m
                    if i_a != i_c:
                        for b in range(mp):
                            j_b = b // p
                            k_b = b % p
                            if k_b != k_c or j_b != j_a:
                                row = np.array([U[a, r] * V[b, r] for r in range(R)])
                                invalid_eqs.append((row, 0.0))
                                if len(invalid_eqs) >= R:
                                    break
                    if len(invalid_eqs) >= R:
                        break
                
                # Combine equations
                all_eqs = valid_eqs + invalid_eqs[:max(0, R - len(valid_eqs))]
                
                if len(all_eqs) < R:
                    # Not enough equations, add more invalid
                    for a in range(nm):
                        for b in range(mp):
                            if (a, b, c) not in valid_set:
                                row = np.array([U[a, r] * V[b, r] for r in range(R)])
                                all_eqs.append((row, 0.0))
                                if len(all_eqs) >= R:
                                    break
                        if len(all_eqs) >= R:
                            break
                
                if len(all_eqs) < R:
                    success = False
                    break
                
                A = np.array([eq[0] for eq in all_eqs[:R]])
                b_vec = np.array([eq[1] for eq in all_eqs[:R]])
                
                # Solve
                try:
                    W_c = np.linalg.solve(A, b_vec)
                    W[c, :] = W_c
                except np.linalg.LinAlgError:
                    success = False
                    break
            
            if not success:
                continue
            
            # Verify
            if verify(W.tolist(), U.tolist(), V.tolist(), R):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from low to high
    for R in range(24, 73):
        if time.time() > deadline:
            break
        result = try_rank_linear(R)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
