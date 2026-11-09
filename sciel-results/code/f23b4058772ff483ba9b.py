"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _validate_tensor(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=np.int64) if isinstance(U, list) else U
    Va = np.array(V, dtype=np.int64) if isinstance(V, list) else V
    Wa = np.array(W, dtype=np.int64) if isinstance(W, list) else W
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = int(np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]))
                if s != 1:
                    return False
    return True

def _make_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _search_rank9_2x3x3(time_limit=10.0):
    """Search for rank-9 decomposition of 2x3 * 3x3 multiplication."""
    n, m, p = 2, 3, 3
    start_time = time.time()
    target_rank = 9
    
    # Try structured patterns based on known low-rank decompositions
    # For 2x3 * 3x3, we can try to find a rank-9 solution
    # The key is that each (i,j,k) triple must satisfy the constraint
    
    # Strategy: Use the fact that we can share computation between blocks
    # Try random sparse integer solutions
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(attempt * 7919)
        
        U = np.zeros((n*m, target_rank), dtype=np.int64)
        V = np.zeros((m*p, target_rank), dtype=np.int64)
        W = np.zeros((p*n, target_rank), dtype=np.int64)
        
        # Initialize with random small integers
        for r in range(target_rank):
            # Each column should have some structure
            for idx in range(n*m):
                if random.random() < 0.4:
                    U[idx, r] = random.choice([-1, 1, 2])
            for idx in range(m*p):
                if random.random() < 0.4:
                    V[idx, r] = random.choice([-1, 1, 2])
            for idx in range(p*n):
                if random.random() < 0.4:
                    W[idx, r] = random.choice([-1, 1, 2])
        
        # Verify
        if _validate_tensor(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    # Try a more systematic approach: build from known patterns
    # For 2x3 * 3x3, one known approach uses the structure of matrix multiplication
    # Let's try to construct it manually based on the bilinear form
    
    # Alternative: Use the fact that rank 9 might be achievable with specific patterns
    # Try dense random with more attempts
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(attempt * 104729 + 31)
        
        U = np.zeros((n*m, target_rank), dtype=np.int64)
        V = np.zeros((m*p, target_rank), dtype=np.int64)
        W = np.zeros((p*n, target_rank), dtype=np.int64)
        
        # Random sparse with controlled density
        for r in range(target_rank):
            nzu_u = random.randint(2, min(4, n*m))
            nzu_v = random.randint(2, min(4, m*p))
            nzu_w = random.randint(2, min(4, p*n))
            
            uidx = random.sample(range(n*m), nzu_u)
            vidx = random.sample(range(m*p), nzu_v)
            widx = random.sample(range(p*n), nzu_w)
            
            for a in uidx:
                U[a, r] = random.choice([-1, 1, 2, -2])
            for a in vidx:
                V[a, r] = random.choice([-1, 1, 2, -2])
            for a in widx:
                W[a, r] = random.choice([-1, 1, 2, -2])
        
        if _validate_tensor(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Combine two 2x3 * 3xp1 and 2x3 * 3xp2 decompositions into 2x3 * 3x(p1+p2)."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*(p1+p2))]
    W = [[0]*total_rank for _ in range((p1+p2)*n)]
    
    # First block: columns 0 to rank1-1
    for i in range(n*m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    
    for j in range(m):
        for k in range(p1):
            for r in range(rank1):
                V[j*(p1+p2)+k][r] = V1[j*p1+k][r]
        for k in range(p2):
            for r in range(rank1):
                V[j*(p1+p2)+p1+k][r] = 0
    
    for k in range(p1):
        for i in range(n):
            for r in range(rank1):
                W[k*n+i][r] = W1[k*n+i][r]
    for k in range(p2):
        for i in range(n):
            for r in range(rank1):
                W[(p1+k)*n+i][r] = 0
    
    # Second block: columns rank1 to total_rank-1
    for i in range(n*m):
        for r in range(rank2):
            U[i][rank1+r] = U2[i][r]
    
    for j in range(m):
        for k in range(p1):
            for r in range(rank2):
                V[j*(p1+p2)+k][rank1+r] = 0
        for k in range(p2):
            for r in range(rank2):
                V[j*(p1+p2)+p1+k][rank1+r] = V2[j*p2+k][r]
    
    for k in range(p1):
        for i in range(n):
            for r in range(rank2):
                W[k*n+i][rank1+r] = 0
    for k in range(p2):
        for i in range(n):
            for r in range(rank2):
                W[(p1+k)*n+i][rank1+r] = W2[k*n+i][r]
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, try to find a good decomposition for 2x3 * 3x3
    # If we can find rank-9 for each 3x3 block, total rank would be 18
    # Otherwise, fall back to schoolbook rank 36
    
    # Start with schoolbook as baseline
    U_school, V_school, W_school, rank_school = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find rank-9 decomposition for 2x3 * 3x3
    rank9_result = _search_rank9_2x3x3(time_limit=30.0)
    
    if rank9_result is not None and time.time() < deadline - 20:
        U1, V1, W1 = rank9_result
        # Use the same decomposition for both blocks
        U2, V2, W2 = U1, V1, W1
        
        U_comb, V_comb, W_comb, rank_comb = _combine_blocks(U1, V1, W1, U2, V2, W2, n, m, 3, 3)
        
        # Verify the combined solution
        if _validate_tensor(U_comb, V_comb, W_comb, n, m, p):
            best_U, best_V, best_W = U_comb, V_comb, W_comb
            best_rank = rank_comb
            print(f"Found combined rank {rank_comb} solution")
    
    # If we didn't find rank 9, try to find any better than schoolbook
    if best_rank == rank_school and time.time() < deadline - 30:
        # Try random search for lower rank
        for target_rank in range(rank_school - 1, max(20, rank_school // 2), -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                random.seed(attempt * 7919 + target_rank)
                
                U = np.zeros((n*m, target_rank), dtype=np.int64)
                V = np.zeros((m*p, target_rank), dtype=np.int64)
                W = np.zeros((p*n, target_rank), dtype=np.int64)
                
                for r in range(target_rank):
                    nzu_u = random.randint(1, min(3, n*m))
                    nzu_v = random.randint(1, min(3, m*p))
                    nzu_w = random.randint(1, min(3, p*n))
                    
                    uidx = random.sample(range(n*m), nzu_u)
                    vidx = random.sample(range(m*p), nzu_v)
                    widx = random.sample(range(p*n), nzu_w)
                    
                    for a in uidx:
                        U[a, r] = random.choice([-1, 1, 2])
                    for a in vidx:
                        V[a, r] = random.choice([-1, 1, 2])
                    for a in widx:
                        W[a, r] = random.choice([-1, 1, 2])
                
                if _validate_tensor(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    # Final safety check
    if not _validate_tensor(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
