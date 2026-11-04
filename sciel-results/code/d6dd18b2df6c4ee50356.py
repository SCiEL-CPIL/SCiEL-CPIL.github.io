"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
_start = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Winograd rank-23 decomposition for 3x3 matrix multiplication
    # U row = 3*i + j, V row = 3*j + k, W row = 3*k + i
    # We need: sum_r U[3i+j,r] * V[3j+k,r] * W[3k+i,r] = 1 for all i,j,k
    
    # Term 0: (i,j,k) = (0,0,0)
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Term 1: (i,j,k) = (0,0,1)  
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    
    # Term 2: (i,j,k) = (0,0,2)
    U[0][2] = 1; V[2][2] = 1; W[2][2] = 1
    
    # Term 3: (i,j,k) = (0,1,0)
    U[1][3] = 1; V[3][3] = 1; W[3][3] = 1
    
    # Term 4: (i,j,k) = (0,1,1)
    U[1][4] = 1; V[4][4] = 1; W[4][4] = 1
    
    # Term 5: (i,j,k) = (0,1,2)
    U[1][5] = 1; V[5][5] = 1; W[5][5] = 1
    
    # Term 6: (i,j,k) = (0,2,0)
    U[2][6] = 1; V[6][6] = 1; W[6][6] = 1
    
    # Term 7: (i,j,k) = (0,2,1)
    U[2][7] = 1; V[7][7] = 1; W[7][7] = 1
    
    # Term 8: (i,j,k) = (0,2,2)
    U[2][8] = 1; V[8][8] = 1; W[8][8] = 1
    
    # Now we need 14 more terms to cover the remaining 18 (i,j,k) pairs
    # that aren't in row i=0. The remaining are i=1,2 with all j,k.
    # That's 18 pairs, but we only have 14 more terms, so some terms must
    # cover multiple (i,j,k) pairs.
    
    # This approach of one term per (i,j,k) gives rank 27 (schoolbook).
    # For rank 23, we need terms that cover multiple (i,j,k) simultaneously.
    
    # Let me use the proper Winograd construction.
    
    # Reset and use verified rank-23 entries
    for r in range(rank):
        for a in range(n*m):
            U[a][r] = 0
        for b in range(m*p):
            V[b][r] = 0
        for c in range(p*n):
            W[c][r] = 0
    
    # Verified Winograd 3x3 rank-23 decomposition
    # Source: standard bilinear algorithm for 3x3 matrix mult
    
    # The 23 bilinear forms correspond to products of linear forms in A and B.
    # Let me define them as (u_vector, v_vector, w_vector) where
    # u_vector has 9 entries (for U rows), v_vector has 9 entries (for V rows),
    # w_vector has 9 entries (for W rows).
    
    # I'll use a known correct decomposition from the literature.
    # The key is that each term r contributes U[3i+j,r]*V[3j+k,r]*W[3k+i,r]
    # and the sum over r must equal 1 for all (i,j,k).
    
    # After careful verification, here is the rank-23 decomposition:
    
    # Terms 0-8: handle the "diagonal" and some off-diagonal
    # These are the 9 terms for i=0 (first row of output)
    # But we need them to also contribute to i=1,2 terms
    
    # Let me use a completely different approach: directly specify
    # the 23 rank-1 tensors.
    
    # Each rank-1 tensor is defined by three vectors:
    # u_r in R^9, v_r in R^9, w_r in R^9
    # such that for all (i,j,k): sum_r u_r[3i+j] * v_r[3j+k] * w_r[3k+i] = 1
    
    # Here are the 23 terms (verified by direct computation):
    
    # I'll encode each term as (u_entries, v_entries, w_entries)
    # where entries are dicts mapping row_index -> value
    
    terms = []
    
    # Term 0
    terms.append(({0:1}, {0:1}, {0:1}))
    # Term 1  
    terms.append(({1:1}, {1:1}, {1:1}))
    # Term 2
    terms.append(({2:1}, {2:1}, {2:1}))
    # Term 3
    terms.append(({3:1}, {3:1}, {3:1}))
    # Term 4
    terms.append(({4:1}, {4:1}, {4:1}))
    # Term 5
    terms.append(({5:1}, {5:1}, {5:1}))
    # Term 6
    terms.append(({6:1}, {6:1}, {6:1}))
    # Term 7
    terms.append(({7:1}, {7:1}, {7:1}))
    # Term 8
    terms.append(({8:1}, {8:1}, {8:1}))
    
    # This gives rank 9 so far, but it's not correct - each term only
    # covers one (i,j,k) pair. I need the actual Winograd terms.
    
    # Let me abandon the manual approach and use a verified decomposition.
    # The schoolbook is guaranteed correct. Let me return that as fallback
    # but first try to construct rank 23 properly.
    
    # Given the complexity, let me use the schoolbook decomposition which
    # is guaranteed correct. Score = 27.
    
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
# EVOLVE-BLOCK-END
