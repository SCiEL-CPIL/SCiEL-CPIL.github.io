"""
Rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Use Strassen's rank-7 algorithm for each of the 5 column-pairs (blocks 0-4).
The tensor decomposition for 2x2 matrix multiplication has a minimal rank of 7.
Since the 5 blocks are independent in the general tensor sense, the total rank is 5 * 7 = 35.
Hybrid approaches with 'schoolbook' blocks (rank 8) or hypothetical lower ranks were analyzed and found to be suboptimal or invalid.
"""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 Strassen blocks (rank 7 each)
    rank = 35
    
    # U shape (n*m, rank) = (4, 35)
    # V shape (m*p, rank) = (20, 35)
    # W shape (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms explicitly for clarity and verification
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U entries are (i, j) for A[i,j]
    # V entries are (j, k) for B[j,k]
    # W entries are (k, i) for C[i,k] (since W index is k*n + i)
    
    # Term 0: M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[11][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (c+d)e -> C[1,0]+=1, C[1,1]-=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 2: M3 = a(f-h) -> C[0,1]+=1, C[1,1]+=1
    U[0][2] = 1
    V[1][2] = 1; V[11][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 3: M4 = d(g-e) -> C[0,0]+=1, C[1,0]+=1
    U[3][3] = 1
    V[10][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: M5 = (a+b)h -> C[0,0]-=1, C[0,1]+=1
    U[0][4] = 1; U[1][4] = 1
    V[11][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5: M6 = (c-a)(e+f) -> C[1,1]+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (b-d)(g+h) -> C[0,0]+=1
    U[1][6] = 1; U[3][6] = -1
    V[10][6] = 1; V[11][6] = 1
    W[0][6] = 1
    
    # Now replicate this pattern for the remaining 4 blocks
    # Block t (t=1..4) uses columns 2*t and 2*t+1 of B, and rows 2*t and 2*t+1 of C (in the p dimension)
    # V index for B[j, 2*t + k] is j*p + (2*t + k) = j*10 + 2*t + k
    # W index for C[2*t + k, i] is (2*t + k)*n + i = (2*t + k)*2 + i
    
    for t in range(1, 5):
        col_base = t * 7
        
        # Term 0: M1
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[10+2*t][col] = 1  # V[0, 2t] and V[1, 2t+1] -> indices 2t and 10+2t+1? No.
        # B[j, k] index is j*p + k.
        # For block t, columns are 2t and 2t+1.
        # e = B[0, 2t], h = B[1, 2t+1]
        # So V[0*p + 2t] = V[2t] and V[1*p + 2t+1] = V[10+2t+1]
        # Wait, my previous manual setup for t=0 was:
        # V[0][0] = 1 (e) -> B[0,0]
        # V[11][0] = 1 (h) -> B[1,1] -> index 1*10+1 = 11. Correct.
        # For t=1: e=B[0,2] -> V[2], h=B[1,3] -> V[13]
        
        # Let's re-verify the indices for general t.
        # e = B[0, 2t] -> V[2t]
        # f = B[0, 2t+1] -> V[2t+1]
        # g = B[1, 2t] -> V[10+2t]
        # h = B[1, 2t+1] -> V[10+2t+1]
        
        # W indices:
        # C[i, 2t] -> W[2t*n + i] = W[4t + i]
        # C[i, 2t+1] -> W[(2t+1)*n + i] = W[4t+2 + i]
        
        # Term 0: M1 = (a+d)(e+h)
        # U: a(U[0]), d(U[3])
        # V: e(V[2t]), h(V[10+2t+1])
        # W: C[0,2t](W[4t]), C[1,2t+1](W[4t+2+1]=W[4t+3])
        # My previous code for t=0: W[0][0]=1 (C[0,0]), W[3][0]=1 (C[1,1]).
        # For t=1: W[4][col]=1 (C[0,2]), W[7][col]=1 (C[1,3]).
        
        # Let's just use the loop from the inspiration which was cleaner.
        pass

    # Let's restart the loop logic properly based on the clean definition.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms defined as (u_indices, v_indices, w_indices)
    # u_indices: list of (flat_index, value)
    # v_indices: list of (flat_index, value) -- relative to block start? No, absolute is safer.
    # w_indices: list of (flat_index, value)
    
    # It is easier to define the terms for a single 2x2 block and then map them.
    # Block t uses B columns 2t, 2t+1 and C columns 2t, 2t+1.
    
    # A indices (fixed):
    # a: 0, b: 1, c: 2, d: 3
    
    # B indices for block t:
    # e (B[0,2t]): 2t
    # f (B[0,2t+1]): 2t+1
    # g (B[1,2t]): 10+2t
    # h (B[1,2t+1]): 10+2t+1
    
    # C indices for block t:
    # C[0,2t]: 4t
    # C[0,2t+1]: 4t+2
    # C[1,2t]: 4t+1
    # C[1,2t+1]: 4t+3
    
    # Term 0: M1 = (a+d)(e+h)
    # U: a(0), d(3)
    # V: e(2t), h(10+2t+1)
    # W: C[0,2t](4t), C[1,2t+1](4t+3)
    
    # Term 1: M2 = (c+d)e
    # U: c(2), d(3)
    # V: e(2t)
    # W: C[1,2t](4t+1), C[1,2t+1](4t+3) with -1
    
    # Term 2: M3 = a(f-h)
    # U: a(0)
    # V: f(2t+1), h(10+2t+1) with -1
    # W: C[0,2t+1](4t+2), C[1,2t+1](4t+3)
    
    # Term 3: M4 = d(g-e)
    # U: d(3)
    # V: g(10+2t), e(2t) with -1
    # W: C[0,2t](4t), C[1,2t](4t+1)
    
    # Term 4: M5 = (a+b)h
    # U: a(0), b(1)
    # V: h(10+2t+1)
    # W: C[0,2t](4t) with -1, C[0,2t+1](4t+2)
    
    # Term 5: M6 = (c-a)(e+f)
    # U: c(2), a(0) with -1
    # V: e(2t), f(2t+1)
    # W: C[1,2t+1](4t+3)
    
    # Term 6: M7 = (b-d)(g+h)
    # U: b(1), d(3) with -1
    # V: g(10+2t), h(10+2t+1)
    # W: C[0,2t](4t)
    
    for t in range(5):
        base = t * 7
        e_idx = 2*t
        f_idx = 2*t+1
        g_idx = 10+2*t
        h_idx = 10+2*t+1
        
        c00_idx = 4*t      # C[0, 2t]
        c01_idx = 4*t+2    # C[0, 2t+1]
        c10_idx = 4*t+1    # C[1, 2t]
        c11_idx = 4*t+3    # C[1, 2t+1]
        
        # Term 0
        col = base
        U[0][col] = 1; U[3][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[c00_idx][col] = 1; W[c11_idx][col] = 1
        
        # Term 1
        col = base + 1
        U[2][col] = 1; U[3][col] = 1
        V[e_idx][col] = 1
        W[c10_idx][col] = 1; W[c11_idx][col] = -1
        
        # Term 2
        col = base + 2
        U[0][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[c01_idx][col] = 1; W[c11_idx][col] = 1
        
        # Term 3
        col = base + 3
        U[3][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[c00_idx][col] = 1; W[c10_idx][col] = 1
        
        # Term 4
        col = base + 4
        U[0][col] = 1; U[1][col] = 1
        V[h_idx][col] = 1
        W[c00_idx][col] = -1; W[c01_idx][col] = 1
        
        # Term 5
        col = base + 5
        U[2][col] = 1; U[0][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[c11_idx][col] = 1
        
        # Term 6
        col = base + 6
        U[1][col] = 1; U[3][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[c00_idx][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
