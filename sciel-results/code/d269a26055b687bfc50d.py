import numpy as np
import time
import random
from itertools import product, combinations

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # 5s safety margin

    # Schoolbook decomposition (rank 60) as baseline
    base_rank = n * m * p
    U0 = [[0] * base_rank for _ in range(n * m)]
    V0 = [[0] * base_rank for _ in range(m * p)]
    W0 = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U0, V0, W0

    # --- Strategy: Linear algebra search for lower rank ---
    # The matrix multiplication tensor M_{n,m,p} satisfies:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    #
    # We can fix W (or fix two of the three) and solve for the others.
    # 
    # A key insight: if we fix W to have specific values, the constraints become
    # bilinear in U and V. But if we fix W AND fix the products U*V appropriately,
    # we get a linear system.
    #
    # Alternative approach: Fix W to be the identity-like structure and solve
    # for U and V as a linear system.
    #
    # Let's try: for a given rank r, choose W entries randomly (or structurally),
    # then the constraint becomes:
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    #
    # This is still bilinear in U and V. But if we consider it as:
    #   For each (i,j,k): sum_t (U[i*m+j, t] * W[k*n+i, t]) * V[j*p+k, t] = 1
    #
    # Define A_{(i,j), t} = U[i*m+j, t] and B_{(i,k), t} = W[k*n+i, t]
    # Then for each (i,j,k): sum_t A_{(i,j),t} * B_{(i,k),t} * V_{(j,k),t} = 1
    #
    # This is still hard. Let's try a different approach.
    #
    # --- Structured search: Fix W to be a permutation/selection structure ---
    # If W[k*n+i, t] is 1 for exactly one t per (k,i) pair (or has a simple pattern),
    # then the problem simplifies.
    
    # Try ranks from 30 to 59 (known lower bound is around 30)
    for r in range(30, 60):
        if time.time() > deadline:
            break
        
        found = False
        
        # Attempt 1: Structured W with W[k*n+i, t] = 1 if t == some function
        # Try: W[k*n+i, t] = 1 if (k*n+i) % r == t
        for w_pattern in range(min(r, 10)):  # Try a few W patterns
            if time.time() > deadline:
                break
            
            # Construct W based on pattern
            W_r = [[0] * r for _ in range(p * n)]
            for idx in range(p * n):
                # Different patterns
                if w_pattern == 0:
                    W_r[idx][idx % r] = 1
                elif w_pattern == 1:
                    W_r[idx][(idx // n) % r] = 1
                elif w_pattern == 2:
                    W_r[idx][(idx % n) % r] = 1
                elif w_pattern == 3:
                    W_r[idx][((idx // p) + (idx % p)) % r] = 1
                else:
                    continue
            
            # Now we need to find U and V such that:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
            #
            # For each (i,j,k), this is:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            #
            # Let's collect the non-zero W entries. For a given (i,k), W[k*n+i, t] 
            # is non-zero for some set of t's.
            
            # Build the system: unknowns are U[i*m+j, t] for all i,j,t and V[j*p+k, t] for all j,k,t
            # Total unknowns: n*m*r + m*p*r
            # Constraints: n*m*p = 60
            
            # This is a bilinear system. Let's try to solve it by fixing one set.
            
            # Fix V to random values and solve for U? No, it's bilinear.
            
            # Alternative: use iterative refinement
            # Initialize U, V randomly
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                U_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
                V_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
                
                # Check if valid
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for t in range(r):
                                s += U_r[i*m+j][t] * V_r[j*p+k][t] * W_r[k*n+i][t]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            break
    
    # --- Strategy 2: Direct linear system approach ---
    # Fix W to be the schoolbook-like structure but with rank r
    # For each (i,k), assign W[k*n+i, t] = 1 for t in some set S_{i,k}
    # Then the constraint becomes:
    #   For each (i,j,k): sum_{t in S_{i,k}} U[i*m+j, t] * V[j*p+k, t] = 1
    #
    # If we set S_{i,k} = {t : t % p == k} (so each column k gets r/p entries),
    # then for each (i,j,k): sum_{t: t%p==k} U[i*m+j, t] * V[j*p+k, t] = 1
    
    if time.time() < deadline:
        for r in range(max(30, best_rank - 5), min(best_rank, 55)):
            if time.time() > deadline:
                break
            
            # Try assigning W columns to (i,k) pairs
            # p*n = 15 entries of W, r columns
            # Each W[k*n+i, t] can be 0 or 1 (or -1)
            
            # Simple assignment: W[k*n+i, t] = 1 if t == (k*n+i) % r
            W_r = [[0] * r for _ in range(p * n)]
            for idx in range(p * n):
                W_r[idx][idx % r] = 1
            
            # Now solve for U and V
            # Constraint: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            # Since W[k*n+i, t] = 1 only when t = (k*n+i) % r:
            #   U[i*m+j, (k*n+i)%r] * V[j*p+k, (k*n+i)%r] = 1
            #
            # This means: for each i,k, let t_ik = (k*n+i) % r
            # Then U[i*m+j, t_ik] * V[j*p+k, t_ik] = 1 for all j
            #
            # This implies U[i*m+j, t_ik] = 1 and V[j*p+k, t_ik] = 1 for all j,
            # OR U[i*m+j, t_ik] = -1 and V[j*p+k, t_ik] = -1 for all j.
            #
            # But t_ik must be consistent: for fixed i, different k give different t_ik.
            # U[i*m+j, t_ik] is the same U entry for different j, so that's fine.
            # But for fixed j, different i,k give constraints on V[j*p+k, t_ik].
            #
            # Let's check if this assignment works:
            # For each (i,k), t_ik = (k*n+i) % r
            # Set U[i*m+j, t_ik] = 1 for all j
            # Set V[j*p+k, t_ik] = 1 for all j
            #
            # But U[i*m+j, t] might be set multiple times if different (i',k') give same t.
            # Check for conflicts:
            
            valid_assign = True
            U_test = [[0] * r for _ in range(n * m)]
            V_test = [[0] * r for _ in range(m * p)]
            
            for i in range(n):
                for k in range(p):
                    t = (k * n + i) % r
                    for j in range(m):
                        if U_test[i*m+j][t] != 0 and U_test[i*m+j][t] != 1:
                            valid_assign = False
                            break
                        U_test[i*m+j][t] = 1
                        if V_test[j*p+k][t] != 0 and V_test[j*p+k][t] != 1:
                            valid_assign = False
                            break
                        V_test[j*p+k][t] = 1
                    if not valid_assign:
                        break
                if not valid_assign:
                    break
            
            if valid_assign:
                # Verify the full decomposition
                W_test = [[0] * r for _ in range(p * n)]
                for idx in range(p * n):
                    W_test[idx][idx % r] = 1
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for t in range(r):
                                s += U_test[i*m+j][t] * V_test[j*p+k][t] * W_test[k*n+i][t]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = r
                    best_U, best_V, best_W = U_test, V_test, W_test
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
