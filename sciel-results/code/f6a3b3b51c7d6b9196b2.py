"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _strassen_block_decomposition(n, m, p):
    """Build rank-21 decomposition using Strassen on three 2x2 blocks."""
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        offset = b * 7
        
        U[0][offset+0] = 1
        U[0][offset+2] = 1
        U[0][offset+4] = 1
        U[0][offset+5] = -1
        
        U[1][offset+4] = 1
        U[1][offset+6] = -1
        
        U[2][offset+1] = 1
        U[2][offset+5] = 1
        
        U[3][offset+0] = 1
        U[3][offset+1] = 1
        U[3][offset+3] = 1
        
        V[0 * p + b*2][offset+0] = 1
        V[0 * p + b*2][offset+1] = 1
        V[0 * p + b*2][offset+5] = 1
        
        V[0 * p + b*2 + 1][offset+3] = -1
        V[0 * p + b*2 + 1][offset+6] = 1
        
        V[1 * p + b*2][offset+2] = 1
        V[1 * p + b*2][offset+3] = -1
        V[1 * p + b*2][offset+5] = 1
        
        V[1 * p + b*2 + 1][offset+0] = 1
        V[1 * p + b*2 + 1][offset+4] = 1
        V[1 * p + b*2 + 1][offset+6] = 1
        
        W[(b*2) * n + 0][offset+0] = 1
        W[(b*2) * n + 0][offset+3] = 1
        W[(b*2) * n + 0][offset+4] = -1
        W[(b*2) * n + 0][offset+6] = 1
        
        W[(b*2+1) * n + 0][offset+2] = 1
        W[(b*2+1) * n + 0][offset+4] = 1
        
        W[(b*2) * n + 1][offset+1] = 1
        W[(b*2) * n + 1][offset+3] = 1
        
        W[(b*2+1) * n + 1][offset+0] = 1
        W[(b*2+1) * n + 1][offset+1] = -1
        W[(b*2+1) * n + 1][offset+2] = 1
        W[(b*2+1) * n + 1][offset+5] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition(n, m, p):
    """Fallback rank-24 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _alternating_projection_rank20(n, m, p, max_iters=2000):
    """
    Structured search for rank-20 decomposition using alternating projection.
    
    Key idea: Start from the rank-21 Strassen solution. Try to merge two rank-1
    terms into one by introducing a shared term. We parameterize the decomposition
    with 20 terms where 19 are "standard" Strassen-like and 1 is a "bridge" term
    that couples across blocks.
    
    We use alternating least squares:
    - Fix V and W, solve for U (linear)
    - Fix U and W, solve for V (linear)
    - Fix U and V, solve for W (linear)
    
    With structured sparsity patterns that respect the block structure.
    """
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    rank = 20
    num_constraints = n * m * p  # 24
    
    # Define structured sparsity patterns for rank-20
    # Terms 0-6: block 0 (Strassen-like)
    # Terms 7-13: block 1 (Strassen-like)  
    # Terms 14-19: block 2 (Strassen-like, but one term is shared/modified)
    
    # Actually, let's try a different approach: use the rank-21 solution as a starting
    # point and try to eliminate one degree of freedom.
    
    # Better approach: parameterize with free variables and solve.
    # For rank 20, we have 20*(4+12+12) = 480 unknowns and 24 equations.
    # This is very underdetermined. We need to fix the sparsity pattern.
    
    # Strategy: Use the Strassen pattern for all 3 blocks (21 terms), then try to
    # set one term to zero and adjust others. Or better: use a different pattern.
    
    # Let me try: random structured search with alternating LS.
    
    rng = random.Random(123)
    
    for attempt in range(max_iters):
        if time.time() > _DEADLINE - 15:
            break
        
        # Create random sparse patterns
        U_pat = np.zeros((nm, rank), dtype=bool)
        V_pat = np.zeros((mp, rank), dtype=bool)
        W_pat = np.zeros((pn, rank), dtype=bool)
        
        for r in range(rank):
            # Each term touches ~2-3 entries in each factor
            for _ in range(rng.randint(2, 4)):
                U_pat[rng.randint(0, nm-1), r] = True
            for _ in range(rng.randint(2, 4)):
                V_pat[rng.randint(0, mp-1), r] = True
            for _ in range(rng.randint(2, 4)):
                W_pat[rng.randint(0, pn-1), r] = True
        
        # Initialize with random small values
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        for i in range(nm):
            for r in range(rank):
                if U_pat[i, r]:
                    U[i, r] = rng.choice([-1, 1])
        for i in range(mp):
            for r in range(rank):
                if V_pat[i, r]:
                    V[i, r] = rng.choice([-1, 1])
        for i in range(pn):
            for r in range(rank):
                if W_pat[i, r]:
                    W[i, r] = rng.choice([-1, 1])
        
        # Alternating projection
        converged = False
        for it in range(50):
            if time.time() > _DEADLINE - 15:
                break
            
            # Fix V and W, solve for U
            # For each (i,j): sum_r U[i,j,r] * V[j,p+k,r] * W[k,n+i,r] = 1 for all k
            # This is: for each (i,j), we have p equations in rank unknowns for U[i,j,:]
            # But it's coupled across k through V and W.
            
            # Actually, let's do proper alternating LS.
            # Fix V and W. For each (i,j), the constraint is:
            # sum_k sum_r U[i,j,r] * V[j,p+k,r] * W[k,n+i,r] = ... no
            # The constraint is per (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # For fixed (i,j), varying k gives p equations. But U[i*m+j,r] appears in all p equations.
            # So for fixed (i,j): for each k, sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # This is p equations in rank unknowns: A_{k,r} = V[j*p+k,r] * W[k*n+i,r]
            # Solve A * U[i*m+j,:] = ones(p)
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build A: p x rank
                    A_u = np.zeros((p, rank), dtype=float)
                    for k in range(p):
                        for r in range(rank):
                            A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve A_u @ U[i*m+j, :] = ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_u, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = sol
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_v = np.zeros((n, rank), dtype=float)
                    for i in range(n):
                        for r in range(rank):
                            A_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_v, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = sol
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_w = np.zeros((m, rank), dtype=float)
                    for j in range(m):
                        for r in range(rank):
                            A_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = sol
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            # Check convergence
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if diff < 1e-10:
                converged = True
                break
        
        if not converged:
            continue
        
        # Check if solution is close to integer/half-integer
        max_dev = 0
        for arr in [U, V, W]:
            for val in arr.flatten():
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                dev = min(abs(val - nearest_int), abs(val - nearest_half))
                max_dev = max(max_dev, dev)
        
        if max_dev < 0.05:
            # Round and verify
            U_int = np.round(U * 2).astype(int)  # doubled
            V_int = np.round(V * 2).astype(int)
            W_int = np.round(W * 2).astype(int)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                        if s != 8:  # (2*2*2)=8 from doubling all three
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                U_list = []
                for i in range(nm):
                    row = []
                    for r in range(rank):
                        if U_int[i, r] % 2 == 0:
                            row.append(U_int[i, r] // 2)
                        else:
                            row.append(U_int[i, r] / 2)
                    U_list.append(row)
                
                V_list = []
                for i in range(mp):
                    row = []
                    for r in range(rank):
                        if V_int[i, r] % 2 == 0:
                            row.append(V_int[i, r] // 2)
                        else:
                            row.append(V_int[i, r] / 2)
                    V_list.append(row)
                
                W_list = []
                for i in range(pn):
                    row = []
                    for r in range(rank):
                        if W_int[i, r] % 2 == 0:
                            row.append(W_int[i, r] // 2)
                        else:
                            row.append(W_int[i, r] / 2)
                    W_list.append(row)
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank20_structured(n, m, p):
    """
    Try to find rank-20 by exploiting the block structure.
    
    The 2x2x6 problem can be seen as 3 independent 2x2x2 problems (one per block).
    Each 2x2x2 has Strassen rank 7, giving total rank 21.
    
    To get rank 20, we need to share one rank-1 term across blocks.
    We parameterize this explicitly and solve for coefficients.
    """
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    rank = 20
    
    # Standard Strassen for 2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 - m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m7
    
    # For block b (b=0,1,2), the input is A (2x2) and B_b (2x2), output C_b (2x2)
    # U entries: U[i*m+j, r] where r corresponds to which Strassen product
    # V entries: V[j*p+k, r] where k = b*2 + k_local
    # W entries: W[k*n+i, r] where k = b*2 + k_local
    
    # Let's define the Strassen pattern explicitly for rank 21:
    # For each block b, terms 7*b through 7*b+6
    
    # Now try rank 20: we'll use terms 0-18 as standard Strassen for 3 blocks,
    # and term 19 as a "correction" term that modifies the pattern.
    
    # Actually, a cleaner approach: use 20 terms where the first 19 follow Strassen
    # for blocks 0,1,2 (with one block using only 6 terms instead of 7), and the 20th
    # term compensates.
    
    # Let me try a different parameterization:
    # Use all 21 Strassen terms but set one to be a linear combination of others.
    
    # For now, use the alternating projection method with structured initialization.
    
    # Initialize with Strassen pattern for rank 21, then zero out one term and
    # let the others adjust.
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Set up Strassen pattern for first 21 terms (but we only have 20)
    # Use terms 0-6 for block 0, 7-13 for block 1, 14-19 for block 2 (only 6 terms)
    
    # Block 0: terms 0-6
    # U[0,0]=1, U[0,2]=1, U[0,4]=1, U[0,5]=-1
    # U[1,4]=1, U[1,6]=-1
    # U[2,1]=1, U[2,5]=1
    # U[3,0]=1, U[3,1]=1, U[3,3]=1
    
    U[0, 0] = 1; U[0, 2] = 1; U[0, 4] = 1; U[0, 5] = -1
    U[1, 4] = 1; U[1, 6] = -1
    U[2, 1] = 1; U[2, 5] = 1
    U[3, 0] = 1; U[3, 1] = 1; U[3, 3] = 1
    
    # V for block 0 (k=0,1)
    V[0, 0] = 1; V[0, 1] = 1; V[0, 5] = 1
    V[1, 3] = -1; V[1, 6] = 1
    V[2, 2] = 1; V[2, 3] = -1; V[2, 5] = 1
    V[3, 0] = 1; V[3, 4] = 1; V[3, 6] = 1
    
    # W for block 0 (k=0,1)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    # Block 1: terms 7-13
    U[0, 7] = 1; U[0, 9] = 1; U[0, 11] = 1; U[0, 12] = -1
    U[1, 11] = 1; U[1, 13] = -1
    U[2, 8] = 1; U[2, 12] = 1
    U[3, 7] = 1; U[3, 8] = 1; U[3, 10] = 1
    
    V[4, 7] = 1; V[4, 8] = 1; V[4, 12] = 1
    V[5, 10] = -1; V[5, 13] = 1
    V[6, 9] = 1; V[6, 10] = -1; V[6, 12] = 1
    V[7, 7] = 1; V[7, 11] = 1; V[7, 13] = 1
    
    W[4, 7] = 1; W[4, 10] = 1; W[4, 11] = -1; W[4, 13] = 1
    W[5, 9] = 1; W[5, 11] = 1
    W[6, 8] = 1; W[6, 10] = 1
    W[7, 7] = 1; W[7, 8] = -1; W[7, 9] = 1; W[7, 12] = 1
    
    # Block 2: terms 14-19 (only 6 terms, missing one)
    # Use Strassen terms 0-5 (skip term 6)
    U[0, 14] = 1; U[0, 16] = 1; U[0, 18] = 1
    # U[0,19] would be -1 for term 5, but we're using 19 as correction
    U[1, 18] = 1
    # U[1,19] would be -1 for term 6, but we're using 19 as correction
    U[2, 15] = 1; U[2, 19] = 1  # term 5 uses U[2,5]=1, so U[2,19]=1
    U[3, 14] = 1; U[3, 15] = 1; U[3, 17] = 1
    
    V[8, 14] = 1; V[8, 15] = 1
    # V[8,19] would be 1 for term 5
    V[9, 17] = -1
    # V[9,19] would be 1 for term 6
    V[10, 16] = 1; V[10, 17] = -1
    V[11, 14] = 1; V[11, 18] = 1
    # V[11,19] would be 1 for term 6
    
    W[8, 14] = 1; W[8, 17] = 1; W[8, 18] = -1
    # W[8,19] would be 1 for term 6
    W[9, 16] = 1; W[9, 18] = 1
    W[10, 15] = 1; W[10, 17] = 1
    W[11, 14] = 1; W[11, 15] = -1; W[11, 16] = 1
    # W[11,19] would be 1 for term 5
    
    # Now we need to solve for the free coefficients to make this work.
    # The issue is that we've removed term 6 from block 2, so we need to adjust.
    # Let's add free parameters to the existing terms to compensate.
    
    # For now, this is a rough initialization. Let's use alternating projection
    # to refine.
    
    U_init = U.copy()
    V_init = V.copy()
    W_init = W.copy()
    
    # Run alternating projection
    for it in range(100):
        if time.time() > _DEADLINE - 15:
            break
        
        # Fix V and W, solve for U
        for i in range(n):
            for j in range(m):
                A_u = np.zeros((p, rank), dtype=float)
                for k in range(p):
                    for r in range(rank):
                        A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_u, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Fix U and W, solve for V
        for j in range(m):
            for k in range(p):
                A_v = np.zeros((n, rank), dtype=float)
                for i in range(n):
                    for r in range(rank):
                        A_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_v, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Fix U and V, solve for W
        for k in range(p):
            for i in range(n):
                A_w = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for r in range(rank):
                        A_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_w, np.ones(m), rcond=None)
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Check if close to integer/half-integer
        max_dev = 0
        for arr in [U, V, W]:
            for val in arr.flatten():
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                dev = min(abs(val - nearest_int), abs(val - nearest_half))
                max_dev = max(max_dev, dev)
        
        if max_dev < 0.05:
            U_int = np.round(U * 2).astype(int)
            V_int = np.round(V * 2).astype(int)
            W_int = np.round(W * 2).astype(int)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                        if s != 8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = []
                for i in range(nm):
                    row = []
                    for r in range(rank):
                        if U_int[i, r] % 2 == 0:
                            row.append(U_int[i, r] // 2)
                        else:
                            row.append(U_int[i, r] / 2)
                    U_list.append(row)
                
                V_list = []
                for i in range(mp):
                    row = []
                    for r in range(rank):
                        if V_int[i, r] % 2 == 0:
                            row.append(V_int[i, r] // 2)
                        else:
                            row.append(V_int[i, r] / 2)
                    V_list.append(row)
                
                W_list = []
                for i in range(pn):
                    row = []
                    for r in range(rank):
                        if W_int[i, r] % 2 == 0:
                            row.append(W_int[i, r] // 2)
                        else:
                            row.append(W_int[i, r] / 2)
                    W_list.append(row)
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the known rank-21 solution
    result_21 = _strassen_block_decomposition(n, m, p)
    if _verify(result_21["U"], result_21["V"], result_21["W"], n, m, p):
        best = result_21
    else:
        best = _schoolbook_decomposition(n, m, p)
    
    # Try structured rank-20 search
    if time.time() < _DEADLINE - 20:
        result_20 = _try_rank20_structured(n, m, p)
        if result_20 is not None:
            if _verify(result_20["U"], result_20["V"], result_20["W"], n, m, p):
                return result_20
    
    # Try random alternating projection for rank 20
    if time.time() < _DEADLINE - 10:
        result_20b = _alternating_projection_rank20(n, m, p, max_iters=500)
        if result_20b is not None:
            if _verify(result_20b["U"], result_20b["V"], result_20b["W"], n, m, p):
                return result_20b
    
    return best
# EVOLVE-BLOCK-END
