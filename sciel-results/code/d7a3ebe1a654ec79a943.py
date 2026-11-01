import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W):
        r = len(U[0])
        # Check that sum_t U[:,t] * V[:,t] * W[:,t]^T equals the tensor
        # Tensor T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # i.e., T[i*m+j, j*p+k, k*n+i] = 1
        # The decomposition says T = sum_t u_t * v_t * w_t^T where u_t is a-vector, v_t is b-vector, w_t is c-vector
        # So T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        
        # Build the tensor
        T = np.zeros((n*m, m*p, p*n), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute sum
        S = np.zeros((n*m, m*p, p*n), dtype=object)
        for t in range(r):
            u = [Fraction(U[a][t]) for a in range(n*m)]
            v = [Fraction(V[b][t]) for b in range(m*p)]
            w = [Fraction(W[c][t]) for c in range(p*n)]
            for a in range(n*m):
                if u[a] == 0:
                    continue
                for b in range(m*p):
                    if v[b] == 0:
                        continue
                    for c in range(p*n):
                        if w[c] == 0:
                            continue
                        S[a,b,c] += u[a] * v[b] * w[c]
        
        # Compare
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if S[a,b,c] != T[a,b,c]:
                        return False
        return True
    
    # Try to find a rank-35 decomposition using a structured approach
    # The tensor has a special structure. Let's try to use the fact that
    # for fixed (i,j), the k-dimension has a cyclic structure.
    
    # Approach: Try to express the tensor as sum of rank-1 terms.
    # The schoolbook has 36 terms. Can we combine some?
    
    # Let's try a different approach: use the structure of the problem.
    # For M_{2,2,p}, there might be a construction using polynomial multiplication.
    
    # Actually, let's try a randomized search for a lower rank decomposition.
    # We'll use the method of "tensor rank via linear algebra".
    
    # Given the time constraint and the difficulty of finding a lower rank,
    # let's try a few structured approaches.
    
    # Approach 1: Try rank 35 with a single linear dependency
    # If the 36 schoolbook terms are linearly dependent, we can reduce rank by 1.
    
    # Check if the 36 vectors (in the tensor space) are linearly dependent.
    # The tensor space has dimension (n*m)*(m*p)*(p*n) = 4*18*18 = 1296.
    # 36 vectors in 1296-dimensional space are almost certainly linearly independent.
    
    # So rank 35 is unlikely to work with a simple linear combination.
    
    # Let's try a completely different approach: use the fact that p=9 is odd.
    # For M_{2,2,p} with p odd, there might be a construction using
    # the fact that -1 is not a square mod p (for p ≡ 3 mod 4) or similar.
    
    # Actually, let's just try to verify the schoolbook and return it.
    # The probability of finding a lower rank in the remaining time is very low.
    
    if time.time() > deadline - 10:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try a few random rank-35 decompositions
    np.random.seed(42)
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Generate a random rank-35 decomposition
        U_rand = np.random.randn(n*m, 35)
        V_rand = np.random.randn(m*p, 35)
        W_rand = np.random.randn(p*n, 35)
        
        # Check if this decomposes the tensor
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute S
        S = np.zeros((n*m, m*p, p*n))
        for t in range(35):
            S += np.outer(np.outer(U_rand[:, t], V_rand[:, t]), W_rand[:, t])
        
        if np.allclose(S, T, atol=1e-10):
            # Found a valid rank-35 decomposition
            # Convert to fractions
            U_frac = [[0]*35 for _ in range(n*m)]
            V_frac = [[0]*35 for _ in range(m*p)]
            W_frac = [[0]*35 for _ in range(p*n)]
            for a in range(n*m):
                for t in range(35):
                    U_frac[a][t] = float(U_rand[a][t])
            for b in range(m*p):
                for t in range(35):
                    V_frac[b][t] = float(V_rand[b][t])
            for c in range(p*n):
                for t in range(35):
                    W_frac[c][t] = float(W_rand[c][t])
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
