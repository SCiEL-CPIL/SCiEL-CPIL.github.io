"""Matrix multiplication 2,3,3 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic for ALL entries."""
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute target values for all (u, v, w)
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                targets[(u, v, w)] = 1

    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                target = targets.get((u, v, w), 0)
                s = Fraction(0)
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[w][c]
                if s != target:
                    return False
    return True

def _try_rank_exact(rank, n, m, p, max_attempts=50, time_limit=None):
    """Try to find a rank-r decomposition using exact rational arithmetic.
    
    Strategy: Randomly choose U and V with small rational entries, then solve
    for W exactly. The system is:
    For each (u,v,w): sum_c U[u,c]*V[v,c]*W[w,c] = target(u,v,w)
    
    This is linear in W for fixed U,V. We solve this linear system exactly.
    """
    rng = np.random.default_rng(42 + rank)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute which (u,v,w) triples have target=1
    ones_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                ones_set.add((u, v, w))
    
    for attempt in range(max_attempts):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Generate random U and V with small integer/rational entries
        # Use values from {-1, 0, 1, 2} to keep denominators small
        U_frac = [[Fraction(rng.choice([-1, 0, 1, 2])) for _ in range(rank)] for _ in range(nm)]
        V_frac = [[Fraction(rng.choice([-1, 0, 1, 2])) for _ in range(rank)] for _ in range(mp)]
        
        # Build linear system for W
        # Unknowns: W[w][c] for w in range(pn), c in range(rank)
        # num_unk = pn * rank
        num_unk = pn * rank
        
        # Equations: for each (u,v,w), sum_c U[u,c]*V[v,c]*W[w,c] = target
        # This gives nm*mp equations, each involving pn*rank unknowns
        # But each equation only involves rank unknowns (for fixed w)
        
        # Actually, for fixed (u,v), the equation is:
        # sum_w [sum_c U[u,c]*V[v,c]*W[w,c]] = target(u,v,w) for each w
        # Wait, no. The tensor entry T[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
        # So for each (u,v,w), we have one equation.
        
        # Total equations: nm * mp * pn = 6*9*6 = 324
        # Total unknowns: pn * rank = 6 * rank
        # For rank=9: 54 unknowns, 324 equations -> overdetermined
        
        # Build the system A x = b where x = W flattened
        # A[eq_idx, unk_idx] where eq_idx = (u*mp+v)*pn+w, unk_idx = w*rank+c
        # A[eq_idx, unk_idx] = U[u,c]*V[v,c] if unk_idx corresponds to same w
        
        # This is a block-diagonal structure! For each w, the equations involving W[w,:]
        # are decoupled from other w values.
        
        # For fixed w, the equations are: for all (u,v) such that we have target(u,v,w):
        # sum_c U[u,c]*V[v,c]*W[w,c] = target(u,v,w)
        
        # So we can solve for each w independently!
        
        W_frac = [[Fraction(0) for _ in range(rank)] for _ in range(pn)]
        
        valid = True
        for w in range(pn):
            # Collect all equations for this w
            # Equation (u,v,w): sum_c U[u,c]*V[v,c]*W[w,c] = target(u,v,w)
            # Number of such equations: nm * mp = 54
            # Unknowns: rank
            
            # Build A_w (54 x rank) and b_w (54,)
            A_w = [[Fraction(0) for _ in range(rank)] for _ in range(nm * mp)]
            b_w = [Fraction(0) for _ in range(nm * mp)]
            
            eq_idx = 0
            for u in range(nm):
                for v in range(mp):
                    target = 1 if (u, v, w) in ones_set else 0
                    for c in range(rank):
                        A_w[eq_idx][c] = U_frac[u][c] * V_frac[v][c]
                    b_w[eq_idx] = target
                    eq_idx += 1
            
            # Solve A_w * W[w,:] = b_w
            # This is a linear system. Use Gaussian elimination with rationals.
            # If the system is inconsistent, this attempt fails.
            
            # Convert to numpy for solving, then convert back
            A_w_np = np.array([[float(x) for x in row] for row in A_w], dtype=np.float64)
            b_w_np = np.array([float(x) for x in b_w], dtype=np.float64)
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A_w_np, b_w_np, rcond=None)
            except np.linalg.LinAlgError:
                valid = False
                break
            
            # Check if solution is valid (residual close to 0)
            if len(residuals) > 0:
                if residuals[0] > 1e-10:
                    valid = False
                    break
            else:
                diff = np.linalg.norm(A_w_np @ sol - b_w_np)
                if diff > 1e-10:
                    valid = False
                    break
            
            # Convert solution to rationals
            for c in range(rank):
                if abs(sol[c]) < 1e-12:
                    W_frac[w][c] = Fraction(0)
                else:
                    # Try to find a nice rational
                    frac = Fraction(sol[c]).limit_denominator(100)
                    W_frac[w][c] = frac
        
        if not valid:
            continue
        
        # Verify the full decomposition exactly
        if _verify_decomposition_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 second budget with safety margin
    
    # Try ranks from lower bound to schoolbook
    # Lower bound is max(nm, mp, pn) = 9
    schoolbook_rank = n * m * p
    
    for rank in range(9, min(schoolbook_rank + 1, 12)):  # Try up to rank 11
        if time.time() > deadline - 20:  # Keep 20s margin
            break
        
        result = _try_rank_exact(rank, n, m, p, max_attempts=100, time_limit=deadline - 10)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
