import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank0 = n * m * p
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # We'll try ranks from 25 down to some minimum
    best = None
    deadline = time.time() + 140  # 10s safety margin
    
    for target_rank in range(25, 19, -1):  # Try 25, 24, 23, 22, 21, 20
        if time.time() > deadline:
            break
        # Try random search for this rank
        found = _search_rank(n, m, p, target_rank, deadline)
        if found:
            best = found
            break
    
    if best:
        return best
    
    # Fall back to schoolbook
    return {"U": U0, "V": V0, "W": W0}

def _search_rank(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition."""
    import random
    import numpy as np
    
    # Initialize with random small integer/half-integer values
    max_iters = 5000
    iters_per_rank = max(1000, max_iters // 7)
    
    for attempt in range(3):
        if time.time() > deadline:
            return None
        
        # Random initialization with small values
        U = []
        for i in range(n*m):
            row = []
            for r in range(target_rank):
                row.append(random.choice([-1, 0, 0, 1]))
            U.append(row)
        
        V = []
        for j in range(m*p):
            row = []
            for r in range(target_rank):
                row.append(random.choice([-1, 0, 0, 1]))
            V.append(row)
        
        W = []
        for k in range(p*n):
            row = []
            for r in range(target_rank):
                row.append(random.choice([-1, 0, 0, 1]))
            W.append(row)
        
        # Local search: try to fix errors
        for it in range(iters_per_rank):
            if time.time() > deadline:
                return None
            
            # Find which (i,j,k) are wrong
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            errors.append((i, j, k, s))
            
            if not errors:
                return {"U": U, "V": V, "W": W}
            
            # Pick a random error and try to fix it
            i, j, k, s = random.choice(errors)
            # Try changing one entry
            choice = random.random()
            if choice < 0.33:
                # Change U
                r = random.randint(0, target_rank-1)
                old = U[i*m+j][r]
                new = random.choice([-1, 0, 1])
                if new != old:
                    U[i*m+j][r] = new
                    # Check if it improved
                    new_s = 0
                    for rr in range(target_rank):
                        new_s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(new_s - 1) > abs(s - 1):
                        U[i*m+j][r] = old
            elif choice < 0.66:
                # Change V
                r = random.randint(0, target_rank-1)
                old = V[j*p+k][r]
                new = random.choice([-1, 0, 1])
                if new != old:
                    V[j*p+k][r] = new
                    new_s = 0
                    for rr in range(target_rank):
                        new_s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(new_s - 1) > abs(s - 1):
                        V[j*p+k][r] = old
            else:
                # Change W
                r = random.randint(0, target_rank-1)
                old = W[k*n+i][r]
                new = random.choice([-1, 0, 1])
                if new != old:
                    W[k*n+i][r] = new
                    new_s = 0
                    for rr in range(target_rank):
                        new_s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(new_s - 1) > abs(s - 1):
                        W[k*n+i][r] = old
    
    return None
