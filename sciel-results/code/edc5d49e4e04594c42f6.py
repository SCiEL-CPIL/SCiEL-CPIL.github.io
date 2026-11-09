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
    
    # Try Kronecker product approach: (2,3,9) = (2,3,3) ⊗ (1,1,3)
    # If we have rank-r decomposition for (2,3,3), we get rank 3r for (2,3,9)
    
    # First, try to find a low-rank decomposition for (2,3,3)
    n2, m2, p2 = 2, 3, 3
    
    def verify_decomp(n, m, p, U, V, W):
        """Verify that U, V, W form a valid decomposition of (n,m,p) matrix multiplication."""
        rank = len(U[0]) if U else 0
        # Build the tensor
        # T[i,j,k] = 1 if we're multiplying A[i,j] * B[j,k]
        # The decomposition should satisfy: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other combinations, the sum should be 0
        # Actually, the tensor for matrix multiplication is:
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # More precisely, T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        # And 0 otherwise
        
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # What should T[a,b,c] be?
                    # a = i*m + j, b = j'*p + k, c = k'*n + i'
                    # T[a,b,c] = 1 if i=i', j=j', k=k' (i.e., the indices match for a valid multiplication)
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
    
    # Search for low-rank decomposition of (2,3,3)
    best_rank_233 = None
    best_decomp_233 = None
    
    # Try ranks from 9 upward
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
        # The (1,1,3) decomposition has rank 3: U1[0,r]=1, V1[r,r]=1, W1[r*1+0,r]=1 for r in 0,1,2
        # Combined rank = best_rank_233 * 3
        
        U233, V233, W233 = best_decomp_233
        r233 = best_rank_233
        total_rank = r233 * 3
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Kronecker product structure
        # For (2,3,3) ⊗ (1,1,3):
        # U_combined[a1*1 + a2, r1*3 + r2] = U233[a1, r1] * U1[a2, r2]
        # where a1 in [0,6) for (2,3), a2 in [0,1) for (1,1)
        # V_combined[b1*1 + b2, r1*3 + r2] = V233[b1, r1] * V1[b2, r2]
        # where b1 in [0,9) for (3,3), b2 in [0,1) for (1,1)
        # W_combined[c1*1 + c2, r1*3 + r2] = W233[c1, r1] * W1[c2, r2]
        # where c1 in [0,6) for (3,2), c2 in [0,1) for (1,1)
        
        # But wait, the dimensions don't match up this way. Let me reconsider.
        
        # Actually, for (n1,m1,p1) ⊗ (n2,m2,p2), the combined dimensions are:
        # (n1*n2, m1*m2, p1*p2)
        # So (2,3,3) ⊗ (1,1,3) = (2*1, 3*1, 3*3) = (2,3,9). Correct!
        
        # The Kronecker product of decompositions:
        # If (U1,V1,W1) is a rank-r1 decomposition for (n1,m1,p1)
        # and (U2,V2,W2) is a rank-r2 decomposition for (n2,m2,p2)
        # then the combined decomposition has rank r1*r2 with:
        # U[a1*n2 + a2, r1*r2 + r2] = U1[a1, r1] * U2[a2, r2]
        # V[b1*m2 + b2, r1*r2 + r2] = V1[b1, r1] * V2[b2, r2]
        # W[c1*p2 + c2, r1*r2 + r2] = W1[c1, r1] * W2[c2, r2]
        
        # For (1,1,3): n2=1, m2=1, p2=3
        # U2 has shape (1*1, 3) = (1, 3): U2[0, r] for r in 0,1,2
        # V2 has shape (1*3, 3) = (3, 3): V2[r, r] = 1
        # W2 has shape (3*1, 3) = (3, 3): W2[r*1+0, r] = 1, i.e., W2[r, r] = 1
        
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
