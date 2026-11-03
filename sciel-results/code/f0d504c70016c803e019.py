import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate(U, V, W):
    """Check exact tensor decomposition. Returns (rank, U, V, W) if valid, else None."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wk[c]
                if s != 1:
                    return None
    return (r, U, V, W)

def _schoolbook():
    """Schoolbook rank-45 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return (rank, U, V, W)

def _try_rank15_ilp():
    """
    Try to find a rank-15 decomposition by fixing U and V to small integers
    and solving for W, checking if W is half-integer.
    """
    n, m, p = DIMENSIONS
    rank = 15
    rng = random.Random(12345)
    
    # We search for U and V in {-1, 0, 1}
    # Then solve for W
    
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(100):
        if _time_left() < 5.0:
            break
            
        # Random U, V
        U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        
        # Convert to numpy for linear algebra
        U_np = np.array(U)
        V_np = np.array(V)
        
        W_list = []
        valid = True
        score = 0.0
        
        for i in range(n):
            for k in range(p):
                # Construct M_ki (3 x rank)
                # M_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M_ki = np.zeros((m, rank))
                for j in range(m):
                    M_ki[j] = U_np[i * m + j] * V_np[j * p + k]
                
                # Solve M_ki @ W_ki = ones(3)
                # W_ki is (rank, 1)
                # Use least squares
                try:
                    W_ki, residuals, rank_m, s = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if W_ki entries are close to half-integers
                # 2 * W_ki should be close to integer
                W_ki_doubled = 2.0 * W_ki.flatten()
                W_ki_rounded = np.round(W_ki_doubled)
                error = np.sum((W_ki_doubled - W_ki_rounded) ** 2)
                score += error
                
                if error > 1e-6:
                    valid = False
                    break
                
                W_list.append(W_ki_rounded / 2.0)
        
        if valid and score < best_score:
            best_score = score
            best_U = U
            best_V = V
            # Reconstruct W in correct order: W[k*n+i]
            W = [[0] * rank for _ in range(p * n)]
            idx = 0
            for k in range(p):
                for i in range(n):
                    W[k * n + i] = W_list[idx]
                    idx += 1
            
            # Verify
            if _validate(best_U, best_V, W) is not None:
                return (rank, best_U, best_V, W)
    
    return None

def construct():
    """Main entry point. Returns {'U', 'V', 'W'} with the best valid decomposition found."""
    # Start with schoolbook as fallback
    best_rank, best_U, best_V, best_W = _schoolbook()
    
    # Try rank 15 ILP-like construction
    if _time_left() > 10.0:
        result = _try_rank15_ilp()
        if result is not None:
            r, U, V, W = result
            if _validate(U, V, W) is not None:
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
