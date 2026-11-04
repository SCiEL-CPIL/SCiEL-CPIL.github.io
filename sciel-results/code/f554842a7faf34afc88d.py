"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_sparse_rank(rank, num_trials=100, time_limit=None):
        """
        Try to find a rank-r exact decomposition using sparse U, V with entries in {-1, 0, 1}.
        Key insight: if U and V are very sparse (few non-zeros per column), then M rows will
        have few non-zeros, making it easier to find simple W.
        """
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            # Generate sparse U, V - each column has ~2-3 non-zeros
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            for r in range(rank):
                # For U column r, pick 2-3 random (i,j) positions to be non-zero
                num_nz_u = random.choice([2, 3])
                indices_u = random.sample(range(n*m), num_nz_u)
                for idx in indices_u:
                    U[idx, r] = random.choice([-1, 1])
                
                # For V column r, pick 2-3 random (j,k) positions to be non-zero
                num_nz_v = random.choice([2, 3])
                indices_v = random.sample(range(m*p), num_nz_v)
                for idx in indices_v:
                    V[idx, r] = random.choice([-1, 1])
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # For each (k,i), we need sum_r M[k*n+i, r] * W[k*n+i, r] = 1
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                found = False
                # Try single non-zero W entry
                for r in range(rank):
                    if abs(M[idx, r]) > 1e-12:
                        w_val = 1.0 / M[idx, r]
                        # Check if w_val is integer or half-integer
                        if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                            W[idx, r] = w_val
                            found = True
                            break
                
                if not found:
                    # Try two non-zero W entries
                    for r1 in range(rank):
                        if abs(M[idx, r1]) < 1e-12:
                            continue
                        for r2 in range(r1+1, rank):
                            if abs(M[idx, r2]) < 1e-12:
                                continue
                            for a in range(-4, 5):
                                for b in range(-4, 5):
                                    if a == 0 and b == 0:
                                        continue
                                    if abs(a*M[idx, r1] + b*M[idx, r2] - 1.0) < 1e-9:
                                        W[idx, r1] = float(a)
                                        W[idx, r2] = float(b)
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                
                if not found:
                    valid = False
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_systematic_rank(rank, time_limit=None):
        """
        Systematic approach: try structured U, V patterns.
        """
        # Pattern 1: U has 1s on diagonal-like structure
        for trial in range(20):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Random permutation-based structure
            perm_u = np.random.permutation(n*m)
            perm_v = np.random.permutation(m*p)
            
            for r in range(rank):
                # Assign non-zeros based on pattern
                if r < n*m:
                    U[perm_u[r % n*m], r] = 1.0
                if r < m*p:
                    V[perm_v[r % m*p], r] = 1.0
            
            # Compute M
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                found = False
                for r in range(rank):
                    if abs(M[idx, r]) > 1e-12:
                        w_val = 1.0 / M[idx, r]
                        if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                            W[idx, r] = w_val
                            found = True
                            break
                
                if not found:
                    for r1 in range(rank):
                        if abs(M[idx, r1]) < 1e-12:
                            continue
                        for r2 in range(r1+1, rank):
                            if abs(M[idx, r2]) < 1e-12:
                                continue
                            for a in range(-3, 4):
                                for b in range(-3, 4):
                                    if a == 0 and b == 0:
                                        continue
                                    if abs(a*M[idx, r1] + b*M[idx, r2] - 1.0) < 1e-9:
                                        W[idx, r1] = float(a)
                                        W[idx, r2] = float(b)
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                
                if not found:
                    valid = False
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bounds: max(n, m, p) = 7, but tensor rank is typically higher
    # For 3x4x7, known bounds suggest rank around 20-30
    ranks_to_try = list(range(15, min(n*m*p, 84)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try sparse random search
        result = try_sparse_rank(rank, num_trials=50, time_limit=DEADLINE - 5)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            break
        
        # Try systematic search if time permits
        if time.time() - start_time < DEADLINE - 10:
            result = try_systematic_rank(rank, time_limit=DEADLINE - 5)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
