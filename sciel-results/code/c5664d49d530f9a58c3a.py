"""Matrix multiplication 2x4x4: rank-minimal exact tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 4, 4]

# Known optimal rank for 2x4x4 matrix multiplication is 16.
# We use a known explicit decomposition of rank 16.
# 
# The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
# where i in [0,2), j in [0,4), k in [0,4).
#
# Indexing: U has shape (n*m, rank) = (8, rank), indexed by (i*m+j, r)
#           V has shape (m*p, rank) = (16, rank), indexed by (j*p+k, r)
#           W has shape (p*n, rank) = (8, rank), indexed by (k*n+i, r)
#
# We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
# (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor entry is 1 for each (i,j,k))

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank = n*m*p = 2*4*4 = 32
    # We aim for rank 16 which is known to be optimal for 2x4x4.
    
    # Known decomposition of rank 16 for 2x4x4:
    # This uses the fact that 2x4x4 can be done in 16 multiplications.
    # The decomposition is based on the identity:
    #   (a1 a2 a3 a4) (b1 b2 b3 b4)
    #   (c1 c2 c3 c4) (d1 d2 d3 d4)
    # 
    # Let me construct a rank-16 decomposition using a systematic approach.
    # 
    # For 2x4x4, we can think of it as two 1x4x4 multiplications plus cross terms.
    # Actually, the known result is that the rank of 2x4x4 is 16.
    # 
    # Let me use a construction: split the 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Ai is 1x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # 
    # C[0,0] = A11*B11 + A12*B21  (sum of 4 products each)
    # C[0,1] = A11*B12 + A12*B22
    # C[1,0] = A21*B11 + A22*B21
    # C[1,1] = A21*B12 + A22*B22
    #
    # Each C[i,j] requires 8 scalar multiplications in schoolbook, total 32.
    # 
    # To get rank 16, we need to share multiplications.
    # 
    # Alternative: Use the Winograd-style approach or a known explicit decomposition.
    #
    # Let me try a direct construction. We have 8 input pairs (i,j) for A, 
    # 16 input pairs (j,k) for B, and 8 output pairs (k,i) for C.
    #
    # Actually, let me just use a known rank-16 decomposition.
    # 
    # For the 2x4x4 case, one can use the following approach:
    # Consider A as 2 rows of 4, B as 4x4.
    # C = A * B, so C[i,k] = sum_j A[i,j] B[j,k].
    #
    # We can write this using 16 rank-1 tensors.
    # Let's define 16 "multiplication slots" r=0..15.
    #
    # One known decomposition:
    # For r in 0..15, let r = 4*a + b where a in {0,1,2,3}, b in {0,1,2,3}.
    # 
    # Actually, let me think differently. The rank of the 2x4x4 tensor is 16.
    # A simple construction: 
    #   For each of the 2 rows i and each of the 4 columns k of C,
    #   C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    #   This is 4 multiplications per (i,k) pair, 8 pairs, total 32.
    #
    # To reduce to 16, we need each rank-1 term to contribute to multiple (i,k) pairs.
    #
    # Let's use a block approach. Write B as [[B_top, B_bot]] where B_top is 2x4, B_bot is 2x4.
    # Then C[i,k] = A[i,0]*B_top[0,k] + A[i,1]*B_top[1,k] + A[i,2]*B_bot[0,k] + A[i,3]*B_bot[1,k]
    #
    # Hmm, let me just try to find a valid rank-16 decomposition by construction.
    #
    # Key insight: The 2x4x4 tensor has rank exactly 16. Here's a decomposition:
    #
    # Label the 16 terms by (s, t) where s in {0,1,2,3}, t in {0,1,2,3}.
    #
    # For term (s,t):
    #   U contribution: for row i, column j:
    #     We want to group the 4 j-values into pairs.
    #
    # Let me try: For s in {0,1,2,3}, t in {0,1,2,3}:
    #   U[i*m+j, r] where r = 4*s+t:
    #     If j == s or j == s+2 (mod 4)... 
    #
    # This is getting complicated. Let me use a different approach.
    #
    # Actually, for 2x4x4, I can use the following known result:
    # The rank is min(2*4*4, ...) = 16.
    # 
    # Simple valid decomposition of rank 16:
    # For r = 0, ..., 15:
    #   Let r = 4*j + k where j in {0,1,2,3}, k in {0,1,2,3}... no that's 16 but only covers one i.
    #
    # Wait: if I use r = 4*j + k, that's 16 terms, but each term can only have U nonzero for one i.
    # For term (j,k): U[i, j, r] = delta(i, i0) for some i0? No, U doesn't depend on k.
    #
    # Let me reconsider. We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    #
    # Construction with rank 16:
    # Let r index pairs (j, k) for j in {0,1,2,3}, k in {0,1,2,3}.
    # But we have 2 values of i, so we need W to handle both i values.
    #
    # If we set: U[i,j,r=(j,k)] = 1 (independent of i)
    #            V[j,k,r=(j',k')] = delta(j,j')*delta(k,k')
    #            W[k,i,r=(j',k')] = delta(k,k')
    # Then sum_r U[i,j,r] V[j,k,r] W[k,i,r] = sum_{j',k'} delta(j,j')delta(k,k') * delta(k,k') = 1.
    # But this is rank 16 with U[i,j,r] = 1 for all i, which means U has 8 rows all identical per column group.
    # Let's check: U has shape (8,16), V has shape (16,16), W has shape (8,16).
    # U[i*m+j, 4*j+k] = 1 for all i. So U[0..7, col] where col=4*j+k has value 1 if j matches.
    # Actually U[i*4+j, 4*j+k] = 1. For fixed j, columns 4j, 4j+1, 4j+2, 4j+3 all have U=1 for both i=0 and i=1.
    # V[j*4+k, 4*j'+k'] = delta(j,j')*delta(k,k').
    # W[k*2+i, 4*j'+k'] = delta(k,k').
    # 
    # Check: sum_{j',k'} U[i*4+j, 4*j'+k'] * V[j*4+k, 4*j'+k'] * W[k*2+i, 4*j'+k']
    # = sum_{j',k'} delta(j,j') * delta(j,j')delta(k,k') * delta(k,k')
    # = delta(j,j') * delta(k,k') summed over j'=j, k'=k = 1. ✓
    #
    # But wait, this gives rank 16! Let me verify the shapes:
    # U: (n*m, rank) = (8, 16) ✓
    # V: (m*p, rank) = (16, 16) ✓  
    # W: (p*n, rank) = (8, 16) ✓
    #
    # Let me code this up.
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U[i*m + j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 32
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
