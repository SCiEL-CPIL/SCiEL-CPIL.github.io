import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp(n, m, p, U, V, W):
        """Verify that U, V, W form a valid decomposition of (n,m,p) matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                    
                    actual = 0
                    for r in range(rank):
                        actual += U[a][r] * V[b][r] * W[c][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Kronecker product approach: (2,3,9) = (2,3,3) ⊗ (1,1,3)
    # If we have rank-r decomposition for (2,3,3), we get rank 3r for (2,3,9)
    
    n2, m2, p2 = 2, 3, 3
    
    # Search for low-rank decomposition of (2,3,3)
    best_rank_233 = None
    best_decomp_233 = None
    
    # Try ranks from 9 upward (schoolbook for 2,3,3 is 18)
    for rank in range(9, 19):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.zeros((n2 * m2, rank))
            V = np.zeros((m2 * p2, rank))
            W = np.zeros((p2 * n2, rank))
            
            # Initialize with random values in {-1, 0, 1}
            for r in range(rank):
                for a in range(n2 * m2):
                    U[a, r] = np.random.choice([-1, 0, 1])
                for b in range(m2 * p2):
                    V[b, r] = np.random.choice([-1, 0, 1])
                for c in range(p2 * n2):
                    W[c, r] = np.random.choice([-1, 0, 1])
            
            # Convert to list of lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomp(n2, m2, p2, U_list, V_list, W_list):
                best_rank_233 = rank
                best_decomp_233 = (U_list, V_list, W_list)
                break
        
        if best_rank_233 is not None:
            break
    
    if best_rank_233 is not None:
        # Use Kronecker product to extend to (2,3,9)
        # (2,3,9) = (2,3,3) ⊗ (1,1,3)
        
        U233, V233, W233 = best_decomp_233
        r233 = best_rank_233
        total_rank = r233 * 3
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # For (1,1,3): n2=1, m2=1, p2=3
        U2 = [[0] * 3 for _ in range(1)]
        V2 = [[0] * 3 for _ in range(3)]
        W2 = [[0] * 3 for _ in range(3)]
        for r in range(3):
            U2[0][r] = 1
            V2[r][r] = 1
            W2[r][r] = 1
        
        n1, m1, p1 = n2, m2, p2  # (2,3,3)
        n2k, m2k, p2k = 1, 1, 3  # (1,1,3)
        
        for a1 in range(n1 * m1):
            for a2 in range(n2k * m2k):
                for r1 in range(r233):
                    for r2 in range(3):
                        r = r1 * 3 + r2
                        a = a1 * (n2k * m2k) + a2
                        U[a][r] = U233[a1][r1] * U2[a2][r2]
        
        for b1 in range(m1 * p1):
            for b2 in range(m2k * p2k):
                for r1 in range(r233):
                    for r2 in range(3):
                        r = r1 * 3 + r2
                        b = b1 * (m2k * p2k) + b2
                        V[b][r] = V233[b1][r1] * V2[b2][r2]
        
        for c1 in range(p1 * n1):
            for c2 in range(p2k * n2k):
                for r1 in range(r233):
                    for r2 in range(3):
                        r = r1 * 3 + r2
                        c = c1 * (p2k * n2k) + c2
                        W[c][r] = W233[c1][r1] * W2[c2][r2]
        
        # Verify
        if verify_decomp(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook()
