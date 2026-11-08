import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # The multiplication tensor: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    def verify(U, V, W, rank):
        """Check if decomposition is valid. Returns True if valid."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find a decomposition of rank < schoolbook_rank
    # Use random initialization with structure
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_sb = [[0.0] * schoolbook_rank for _ in range(n*m)]
    V_sb = [[0.0] * schoolbook_rank for _ in range(m*p)]
    W_sb = [[0.0] * schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1.0
                V_sb[j*p+k][col] = 1.0
                W_sb[k*n+i][col] = 1.0
                col += 1
    
    best_U = [row[:] for row in U_sb]
    best_V = [row[:] for row in V_sb]
    best_W = [row[:] for row in W_sb]
    
    # Try lower ranks with random search
    # For each candidate rank, try random initializations
    for target_rank in range(max(n*m, m*p, p*n), schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with some structure
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Scale to help
            # We want sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # Try iterative refinement
            
            # Use a simple gradient-free approach: for each (i,j,k), adjust
            # Actually, let's use a different strategy: structured random
            # Set some entries to 1 and others to 0 randomly
            U_bin = np.where(np.random.rand(n*m, target_rank) < 0.3, 1.0, 0.0)
            V_bin = np.where(np.random.rand(m*p, target_rank) < 0.3, 1.0, 0.0)
            W_bin = np.where(np.random.rand(p*n, target_rank) < 0.3, 1.0, 0.0)
            
            # Check if this binary decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U_bin[i*m+j, r] * V_bin[j*p+k, r] * W_bin[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_bin.tolist()
                best_V = V_bin.tolist()
                best_W = W_bin.tolist()
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to list format
    if best_U is None:
        best_U = [[0.0] * best_rank for _ in range(n*m)]
        best_V = [[0.0] * best_rank for _ in range(m*p)]
        best_W = [[0.0] * best_rank for _ in range(p*n)]
    
    # Ensure we have the right number of columns
    if best_rank < schoolbook_rank:
        # Pad if needed (shouldn't happen)
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
