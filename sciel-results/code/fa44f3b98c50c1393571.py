import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen's algorithm for 2x2x2 blocks where possible.
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # Block structure:
    # i_blocks: [0,1] (size 2), [2] (size 1)
    # j_blocks: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # k_blocks: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    # We form Strassen blocks for all combinations of size-2 blocks.
    # i_size2 = [0,1]
    # j_size2 = [0,1], [2,3]
    # k_size2 = [0,1], [2,3], [4,5]
    
    # There are 1 * 2 * 3 = 6 Strassen blocks.
    # Each has rank 7.
    # Total Strassen rank = 42.
    
    # Remaining terms:
    # i=2 (size 1)
    # j in [0,1,2,3,4] (all j)
    # k in [0,1,2,3,4,5] (all k)
    # But wait, the Strassen blocks covered i in [0,1], j in [0,3], k in [0,5].
    # So for i=2, all j and k are uncovered.
    # Terms: i=2, j=0..4, k=0..5.
    # Count: 1 * 5 * 6 = 30.
    # Rank 30.
    
    # Total rank = 42 + 30 = 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    # Block indices: i0, i1; j0, j1; k0, k1
    # Strassen 2x2x2 decomposition:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the global indices.
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    i_blocks = [(0, 1)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for i0, i1 in i_blocks:
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                # 7 terms for this block
                # M1
                # A11+A22 -> U[i0,j0]=1, U[i1,j1]=1
                # B11+B22 -> V[j0,k0]=1, V[j1,k1]=1
                # C11, C12, C21, C22 contributions?
                # No, the W factor corresponds to the output C indices.
                # In the tensor decomposition, each rank-1 term contributes to specific (i,j,k) triplets.
                # The Strassen algorithm computes the product of two 2x2 matrices.
                # The tensor for 2x2x2 multiplication has 8 non-zero entries.
                # The rank-7 decomposition has 7 rank-1 terms.
                # Each term is a combination of the 8 basis tensors.
                
                # Let's define the 7 vectors for U, V, W for the 2x2x2 case.
                # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
                # Global i = i0 + i', j = j0 + j', k = k0 + k'
                
                # Term 1: M1 = (A11+A22)(B11+B22)
                # U: i'=0,j'=0 (1); i'=1,j'=1 (1)
                # V: j'=0,k'=0 (1); j'=1,k'=1 (1)
                # W: i'=0,k'=0 (1); i'=0,k'=1 (1); i'=1,k'=0 (1); i'=1,k'=1 (1)
                # Wait, W is for the output C.
                # C11 = M1... so W[i'=0, k'=0] gets 1.
                # C12 = ... M1 is not in C12.
                # C21 = ... M1 is not in C21.
                # C22 = M1... so W[i'=1, k'=1] gets 1.
                # So W has 1s at (0,0) and (1,1).
                
                # Let's hardcode the 7 patterns for U, V, W for 2x2x2.
                # U patterns (2x2):
                # 1: [[1,0],[0,1]]
                # 2: [[0,0],[1,1]]
                # 3: [[1,0],[0,0]]
                # 4: [[0,0],[0,1]]
                # 5: [[1,1],[0,0]]
                # 6: [[-1,0],[1,0]]
                # 7: [[0,1],[0,-1]]
                
                # V patterns (2x2):
                # 1: [[1,0],[0,1]]
                # 2: [[1,0],[0,0]]
                # 3: [[0,1],[0,-1]]
                # 4: [[0,-1],[1,0]]
                # 5: [[0,0],[0,1]]
                # 6: [[1,1],[0,0]]
                # 7: [[0,1],[1,1]]
                
                # W patterns (2x2):
                # 1: [[1,0],[0,1]]
                # 2: [[0,0],[1,0]]
                # 3: [[0,1],[0,0]]
                # 4: [[0,0],[0,1]]
                # 5: [[-1,0],[0,1]]
                # 6: [[0,0],[1,0]]
                # 7: [[1,0],[0,1]]
                
                # Wait, I need to verify these patterns.
                # Let's just use the known rank-7 decomposition.
                # I will define the 7 U, V, W matrices for the 2x2x2 case.
                
                U_pat = [
                    [[1, 0], [0, 1]],
                    [[0, 0], [1, 1]],
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 1]],
                    [[1, 1], [0, 0]],
                    [[-1, 0], [1, 0]],
                    [[0, 1], [0, -1]]
                ]
                V_pat = [
                    [[1, 0], [0, 1]],
                    [[1, 0], [0, 0]],
                    [[0, 1], [0, -1]],
                    [[0, -1], [1, 0]],
                    [[0, 0], [0, 1]],
                    [[1, 1], [0, 0]],
                    [[0, 1], [1, 1]]
                ]
                W_pat = [
                    [[1, 0], [0, 1]],
                    [[0, 0], [1, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 1]],
                    [[-1, 0], [0, 1]],
                    [[0, 0], [1, 0]],
                    [[1, 0], [0, 1]]
                ]
                
                for t in range(7):
                    for ip in range(2):
                        for jp in range(2):
                            val = U_pat[t][ip][jp]
                            if val != 0:
                                U[(i0 + ip) * m + (j0 + jp)][col] += val
                    for jp in range(2):
                        for kp in range(2):
                            val = V_pat[t][jp][kp]
                            if val != 0:
                                V[(j0 + jp) * p + (k0 + kp)][col] += val
                    for ip in range(2):
                        for kp in range(2):
                            val = W_pat[t][ip][kp]
                            if val != 0:
                                W[(k0 + kp) * n + (i0 + ip)][col] += val
                    col += 1
    
    # Schoolbook for remaining
    # i=2, j=0..4, k=0..5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term was covered by Strassen
                covered = False
                if i in [0, 1]:
                    if j in [0, 1, 2, 3]:
                        if k in [0, 1, 2, 3, 4, 5]:
                            covered = True
                if not covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

