"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor with exact rational arithmetic."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # With convention W[k*n+i, r], the tensor is:
    # T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # where e_{(i,j)} is unit vector at position i*m+j in R^{n*m}
    # e_{(j,k)} is unit vector at position j*p+k in R^{m*p}
    # e_{(k,i)} is unit vector at position k*n+i in R^{p*n}
    
    # Build the tensor using exact fractions
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = Fraction(1)
    
    # Try to find a decomposition with rank < 30 using systematic rank-1 peeling
    # with exact rational arithmetic
    
    # We'll try ranks from 24 up to 30
    for target_rank in range(24, 31):
        result = try_peel(T, n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fall back to schoolbook rank 30
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def try_peel(T, n, m, p, target_rank):
    """Try to decompose T into exactly target_rank rank-1 tensors with rational entries."""
    import random
    from fractions import Fraction
    
    # Use deterministic random for reproducibility
    rng = random.Random(42)
    
    # We'll try multiple random initializations and peeling strategies
    for attempt in range(200):
        result = peel_once(T, n, m, p, target_rank, rng)
        if result is not None:
            return result
    
    return None

def peel_once(T, n, m, p, target_rank, rng):
    """Single peeling attempt with exact rational arithmetic."""
    from fractions import Fraction
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Represent residual as dict of (i,j,k) -> Fraction
    residual = dict(T)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(target_rank):
        # Find a rank-1 tensor u ⊗ v ⊗ w that is in the span of residual
        # Strategy: pick random support and solve for coefficients
        
        # Pick random small supports for u, v, w
        sup_u = set(rng.sample(range(dim_u), min(4, dim_u)))
        sup_v = set(rng.sample(range(dim_v), min(4, dim_v)))
        sup_w = set(rng.sample(range(dim_w), min(4, dim_w)))
        
        # Build system: for each (i,j,k) in residual support,
        # sum_{c} U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
        # We want to find one rank-1 term u,v,w such that
        # u[i]*v[j]*w[k] matches residual at some points
        
        # Simpler: try to find u,v,w with small rational values
        # such that u⊗v⊗w is "close" to residual
        
        # Try various small rational values
        found = False
        for scale_u in [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2)]:
            for scale_v in [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2)]:
                for scale_w in [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2)]:
                    u = {i: scale_u for i in sup_u}
                    v = {j: scale_v for j in sup_v}
                    w = {k: scale_w for k in sup_w}
                    
                    # Check if this rank-1 tensor is consistent with residual
                    # i.e., for all (i,j,k) where u[i]*v[j]*w[k] != 0,
                    # residual should have a corresponding entry
                    consistent = True
                    for i in sup_u:
                        for j in sup_v:
                            for k in sup_w:
                                val = u[i] * v[j] * w[k]
                                if val != 0:
                                    if (i,j,k) not in residual or residual[(i,j,k)] == 0:
                                        consistent = False
                                        break
                            if not consistent:
                                break
                        if not consistent:
                            break
                    if not consistent:
                        continue
                    
                    # Subtract this rank-1 tensor from residual
                    new_residual = dict(residual)
                    for i in sup_u:
                        for j in sup_v:
                            for k in sup_w:
                                val = u[i] * v[j] * w[k]
                                if val != 0:
                                    new_residual[(i,j,k)] = residual.get((i,j,k), Fraction(0)) - val
                                    if new_residual[(i,j,k)] == 0:
                                        del new_residual[(i,j,k)]
                    
                    # Check if residual is getting simpler
                    if len(new_residual) < len(residual) or (len(new_residual) == len(residual) and step > 0):
                        # Accept this step
                        residual = new_residual
                        
                        # Build column vectors
                        u_col = [Fraction(0)] * dim_u
                        v_col = [Fraction(0)] * dim_v
                        w_col = [Fraction(0)] * dim_w
                        for i in sup_u:
                            u_col[i] = u[i]
                        for j in sup_v:
                            v_col[j] = v[j]
                        for k in sup_w:
                            w_col[k] = w[k]
                        
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            # Try a different approach: use linear algebra on the tensor slices
            # Flatten residual and try to find rank-1 components
            break
        
        # Check if residual is zero
        if len(residual) == 0:
            break
    
    if len(residual) != 0:
        return None
    
    if len(U_cols) != target_rank:
        return None
    
    # Build the factor matrices
    U = [[float(col[i]) for col in U_cols] for i in range(dim_u)]
    V = [[float(col[i]) for col in V_cols] for i in range(dim_v)]
    W = [[float(col[i]) for col in W_cols] for i in range(dim_w)]
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(target_rank))
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    # Check non-diagonal entries are zero
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 != i2 or j1 != j2 or k1 != k2:
                                val = sum(U[i1*m+j1][c] * V[j2*p+k2][c] * W[k2*n+i2][c] for c in range(target_rank))
                                err = abs(val)
                                max_err = max(max_err, err)
    
    if max_err > 1e-9:
        return None
    
    return {"U": U, "V": V, "W": W}
