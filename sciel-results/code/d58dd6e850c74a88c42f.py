<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 decomposition
    def add_strassen(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, col_offset):
        # Local 2x2 indices mapping to global
        # U rows: (i+j) -> u
        # V rows: (j*2+k) -> v
        # W rows: (k*2+i) -> w
        
        u_idxs = [u_offset_i * m + u_offset_j, 
                  u_offset_i * m + u_offset_j + 1,
                  (u_offset_i + 1) * m + u_offset_j, 
                  (u_offset_i + 1) * m + u_offset_j + 1]
        
        v_idxs = [v_offset_j * p + v_offset_k, 
                  v_offset_j * p + v_offset_k + 1,
                  (v_offset_j + 1) * p + v_offset_k, 
                  (v_offset_j + 1) * p + v_offset_k + 1]
        
        w_idxs = [w_offset_k * n + w_offset_i, 
                  w_offset_k * n + w_offset_i + 1,
                  (w_offset_k + 1) * n + w_offset_i, 
                  (w_offset_k + 1) * n + w_offset_i + 1]

        # Strassen U, V, W matrices (4x7, 4x7, 4x7)
        # Rows 0-3 correspond to A00, A01, A10, A11
        U_S = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_S = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        # W_S rows correspond to C00, C10, C01, C11 in order k*n+i?
        # C00: k=0, i=0 -> 0
        # C10: k=0, i=1 -> 1
        # C01: k=1, i=0 -> 2
        # C11: k=1, i=1 -> 3
        # Formulas:
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        W_S = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]

        for r in range(4):
            for c in range(7):
                val_u = U_S[r][c]
                if val_u != 0:
                    U[u_idxs[r]][col_offset + c] += val_u
                val_v = V_S[r][c]
                if val_v != 0:
                    V[v_idxs[r]][col_offset + c] += val_v
                val_w = W_S[r][c]
                if val_w != 0:
                    W[w_idxs[r]][col_offset + c] += val_w

    # Block 1: A(0:2, 0:2) x B(0:2, 0:2)
    # i=0, j=0 ; j=0, k=0 ; k=0, i=0
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A(0:2, 0:2) x B(0:2, 2:4)
    # i=0, j=0 ; j=0, k=2 ; k=2, i=0
    add_strassen(0, 0, 0, 2, 2, 0, 7)

    # Block 3: A(0:2, 0:2) x B(0:2, 4:5) -> 2x2x1
    # Schoolbook 4 terms
    # A(0,0) B(0,4) -> C(0,4)
    U[0][14] += 1
    V[4][14] += 1
    W[8][14] += 1  # k=4, i=0 -> 4*2+0 = 8
    
    # A(1,0) B(0,4) -> C(1,4)
    U[3][15] += 1
    V[4][15] += 1
    W[9][15] += 1  # k=4, i=1 -> 4*2+1 = 9
    
    # A(0,1) B(1,4) -> C(0,4)
    U[1][16] += 1
    V[9][16] += 1
    W[8][16] += 1
    
    # A(1,1) B(1,4) -> C(1,4)
    U[4][17] += 1
    V[9][17] += 1
    W[9][17] += 1

    # Block 4: A(0:2, 2:3) x B(2:3, 0:5) -> 2x1x5
    # 10 terms: A(i,2) * B(2,k) -> C(i,k)
    for i in range(2):
        for k in range(5):
            col = 18 + i * 5 + k
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2():
    """Return (U_S, V_S, W_S) for 2x2x2 Strassen decomposition."""
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U_S, V_S, W_S

def _build_base_rank28():
    """Build the base rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_S, V_S, W_S = _strassen_2x2x2()
    
    def add_strassen(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, col_offset):
        u_idxs = [u_offset_i * m + u_offset_j, 
                  u_offset_i * m + u_offset_j + 1,
                  (u_offset_i + 1) * m + u_offset_j, 
                  (u_offset_i + 1) * m + u_offset_j + 1]
        v_idxs = [v_offset_j * p + v_offset_k, 
                  v_offset_j * p + v_offset_k + 1,
                  (v_offset_j + 1) * p + v_offset_k, 
                  (v_offset_j + 1) * p + v_offset_k + 1]
        w_idxs = [w_offset_k * n + w_offset_i, 
                  w_offset_k * n + w_offset_i + 1,
                  (w_offset_k + 1) * n + w_offset_i, 
                  (w_offset_k + 1) * n + w_offset_i + 1]

        for r in range(4):
            for c in range(7):
                val_u = U_S[r][c]
                if val_u != 0:
                    U[u_idxs[r]][col_offset + c] += val_u
                val_v = V_S[r][c]
                if val_v != 0:
                    V[v_idxs[r]][col_offset + c] += val_v
                val_w = W_S[r][c]
                if val_w != 0:
                    W[w_idxs[r]][col_offset + c] += val_w

    add_strassen(0, 0, 0, 0, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 2, 0, 7)

    # Block 3: A(0:2, 0:2) x B(0:2, 4:5) -> 2x2x1, schoolbook
    U[0][14] += 1; V[4][14] += 1; W[8][14] += 1
    U[3][15] += 1; V[4][15] += 1; W[9][15] += 1
    U[1][16] += 1; V[9][16] += 1; W[8][16] += 1
    U[4][17] += 1; V[9][17] += 1; W[9][17] += 1

    # Block 4: A(0:2, 2:3) x B(2:3, 0:5) -> 2x1x5, schoolbook
    for i in range(2):
        for k in range(5):
            col = 18 + i * 5 + k
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1

    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the 2x3 by 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 if (i,j,k) is a valid triple, else 0
    # Actually the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank_27(deadline):
    """
    Try to find a rank-27 decomposition by modifying the schoolbook boundary terms.
    Strategy: The 2x2x1 block (cols 14-17) and 2x1x5 block (cols 18-27) use 14 rank-1 terms.
    Try to see if we can reduce by 1 by finding a linear dependency or using a smarter decomposition.
    
    Key idea: For the 2x1x5 block, we have terms A(i,2)*B(2,k) for i in {0,1}, k in {0..4}.
    This is a 2x1x5 tensor which has rank 2 (not 10!). We can factor it as:
      A(0,2)*B(2,k) for all k -> one rank-1 term with U[2,r]=1, V[2*p+k,r]=1 for all k, W[k*n,r]=1
      A(1,2)*B(2,k) for all k -> one rank-1 term with U[5,r]=1, V[2*p+k,r]=1 for all k, W[k*n+1,r]=1
    
    Wait, that's not right. Each rank-1 term is U[i,j]*V[j,k]*W[k,i]. 
    For a fixed (i,j), varying k: the term is A(i,j)*B(j,k) -> C(i,k).
    This contributes to W[k*n+i] for all k. So for j=2, i=0:
      U[0*m+2, r] = 1, V[2*p+k, r] = 1 for all k, W[k*n+0, r] = 1 for all k
    That's ONE rank-1 term! Similarly for i=1, j=2.
    
    So the 2x1x5 block can be done in rank 2, not 10!
    And the 2x2x1 block: A(i,j)*B(j,4) -> C(i,4) for i in {0,1}, j in {0,1}.
    For fixed (i,4): sum over j of A(i,j)*B(j,4). This is like a dot product.
    The tensor is T[i,j,4] = A(i,j)*B(j,4). 
    This is a 2x2x1 tensor. Its rank is at most 4 (schoolbook) but could be less.
    Actually for 2x2x1, the rank is 2: we can use
      Term 1: U[0*m+0]=1, U[0*m+1]=1, V[0*p+4]=1, V[1*p+4]=1, W[4*n+0]=1
      Term 2: U[1*m+0]=1, U[1*m+1]=1, V[0*p+4]=1, V[1*p+4]=1, W[4*n+1]=1
    Wait no, that's not how rank-1 works. Each rank-1 term is a single product.
    
    Let me reconsider. A rank-1 decomposition of the matmul tensor means:
    C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k, r] * W[k*n+i, r]
    
    No, that's not right either. The standard tensor decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where T[i,j,k] = 1 for the matmul tensor (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    For a rank-1 term with index r:
    U[i,j,r] * V[j,k,r] * W[k,i,r] must equal 1 when this term is the only one, or contribute appropriately.
    
    Actually, for the matmul tensor T[i,j,k]=1, a rank-1 term contributes:
    u_{i,j} * v_{j,k} * w_{k,i}
    
    For this to be valid, we need the sum over r of u_{i,j}^{(r)} * v_{j,k}^{(r)} * w_{k,i}^{(r)} = 1 for all i,j,k.
    
    For the 2x1x5 sub-tensor (i in {0,1}, j=2, k in {0..4}):
    We need: sum_r U[i*3+2, r] * V[2*5+k, r] * W[k*2+i, r] = 1 for all i in {0,1}, k in {0..4}.
    
    This is a 2x5 matrix of equations (well, 10 equations) with variables U[i*3+2,r], V[2*5+k,r], W[k*2+i,r].
    
    Key insight: If we set for a single rank r:
    U[2, r] = 1, U[5, r] = 1
    V[10+k, r] = 1 for all k
    W[2k, r] = 1 for all k (i=0)
    W[2k+1, r] = 1 for all k (i=1)
    
    Then for i=0: U[2,r]*V[10+k,r]*W[2k,r] = 1*1*1 = 1 ✓
    For i=1: U[5,r]*V[10+k,r]*W[2k+1,r] = 1*1*1 = 1 ✓
    
    So yes! The 2x1x5 block needs only rank 1, not 10!
    
    Similarly for the 2x2x1 block (i in {0,1}, j in {0,1}, k=4):
    We need: sum_r U[i*3+j, r] * V[j*5+4, r] * W[4*2+i, r] = 1 for all i in {0,1}, j in {0,1}.
    
    This is a 2x2 sub-tensor. The rank of the 2x2x1 matmul tensor is 2:
    Term r1: U[0,r1]=1, U[1,r1]=1, V[4,r1]=1, V[9,r1]=1, W[8,r1]=1, W[9,r1]=1
    Check: For (i=0,j=0): U[0,r1]*V[4,r1]*W[8,r1] = 1*1*1 = 1 ✓
    For (i=0,j=1): U[1,r1]*V[9,r1]*W[8,r1] = 1*1*1 = 1 ✓
    For (i=1,j=0): U[0,r1]*V[4,r1]*W[9,r1] = 1*1*1 = 1 ✓
    For (i=1,j=1): U[1,r1]*V[9,r1]*W[9,r1] = 1*1*1 = 1 ✓
    
    Wait, that's one term handling all 4 cases! So the 2x2x1 block needs rank 1!
    
    So total rank = 7 (Strassen block 1) + 7 (Strassen block 2) + 1 (2x2x1) + 1 (2x1x5) = 16!
    
    But wait, I need to be more careful. The Strassen blocks use specific columns, and the boundary blocks must not interfere. Let me re-examine.
    """
    n, m, p = DIMENSIONS
    # Try rank 16: 7 + 7 + 1 + 1
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_S, V_S, W_S = _strassen_2x2x2()
    
    def add_strassen(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, col_offset):
        u_idxs = [u_offset_i * m + u_offset_j, 
                  u_offset_i * m + u_offset_j + 1,
                  (u_offset_i + 1) * m + u_offset_j, 
                  (u_offset_i + 1) * m + u_offset_j + 1]
        v_idxs = [v_offset_j * p + v_offset_k, 
                  v_offset_j * p + v_offset_k + 1,
                  (v_offset_j + 1) * p + v_offset_k, 
                  (v_offset_j + 1) * p + v_offset_k + 1]
        w_idxs = [w_offset_k * n + w_offset_i, 
                  w_offset_k * n + w_offset_i + 1,
                  (w_offset_k + 1) * n + w_offset_i, 
                  (w_offset_k + 1) * n + w_offset_i + 1]
        for r in range(4):
            for c in range(7):
                if U_S[r][c] != 0:
                    U[u_idxs[r]][col_offset + c] += U_S[r][c]
                if V_S[r][c] != 0:
                    V[v_idxs[r]][col_offset + c] += V_S[r][c]
                if W_S[r][c] != 0:
                    W[w_idxs[r]][col_offset + c] += W_S[r][c]

    add_strassen(0, 0, 0, 0, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 2, 0, 7)
    
    # 2x2x1 block: A(0:2, 0:2) x B(0:2, 4:5) -> C(0:2, 4:5)
    # Rank 1: col 14
    # U[i*3+j, 14] = 1 for i in {0,1}, j in {0,1}
    # V[j*5+4, 14] = 1 for j in {0,1}
    # W[4*2+i, 14] = 1 for i in {0,1}
    for i in range(2):
        for j in range(2):
            U[i*m+j][14] += 1
    for j in range(2):
        V[j*p+4][14] += 1
    for i in range(2):
        W[4*n+i][14] += 1
    
    # 2x1x5 block: A(0:2, 2:3) x B(2:3, 0:5) -> C(0:2, 0:5)
    # Rank 1: col 15
    # U[i*3+2, 15] = 1 for i in {0,1}
    # V[2*5+k, 15] = 1 for k in {0..4}
    # W[k*2+i, 15] = 1 for k in {0..4}, i in {0,1}
    for i in range(2):
        U[i*m+2][15] += 1
    for k in range(5):
        V[2*p+k][15] += 1
    for k in range(5):
        for i in range(2):
            W[k*n+i][15] += 1
    
    # Verify
    if _verify(U, V, W):
        return U, V, W
    return None

def _try_rank_15(deadline):
    """Try to find rank 15 by further optimizing."""
    n, m, p = DIMENSIONS
    # The two Strassen blocks use columns 0-6 and 7-13.
    # The 2x2x1 uses col 14, the 2x1x5 uses col 15.
    # Can we share any structure? Probably not directly, but let's try rank 15.
    # Actually, let's first check if rank 16 works.
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # First try the optimized rank-16 decomposition
    result = _try_rank_27(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-28
    U, V, W = _build_base_rank28()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to reconsider. The rank-1 claim for the 2x2x1 and 2x1x5 blocks needs verification. Let me trace through more carefully.

For the 2x1x5 block with rank 1 (col 15):
- U[i*3+2, 15] = 1 for i=0,1 → U[2,15]=1, U[5,15]=1
- V[2*5+k, 15] = 1 for k=0..4 → V[10,15]=V[11,15]=V[12,15]=V[13,15]=V[14,15]=1
- W[k*2+i, 15] = 1 for all k,i → W[0,15]=W[1,15]=W[2,15]=W[3,15]=W[4,15]=W[5,15]=W[6,15]=W[7,15]=W[8,15]=W[9,15]=1

Check T[i,2,k] = sum_r U[i*3+2,r]*V[2*5+k,r]*W[k*2+i,r]:
- For r=15: U[i*3+2,15]*V[10+k,15]*W[2k+i,15] = 1*1*1 = 1 ✓
- For other r: need to make sure they don't contribute to (i,2,k) entries.

But wait! The Strassen blocks also use some of these indices. Let me check:
- Strassen block 1 uses U rows 0,1,3,4 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1) and V rows 0,1,5,6 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1) and W rows 0,1,2,3 (k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1).
- Strassen block 2 uses U rows 0,1,3,4 and V rows 10,11,15,16 (j=0,k=2; j=0,k=3; j=1,k=2; j=1,k=3) and W rows 4,5,6,7 (k=2,i=0; k=2,i=1; k=3,i=0; k=3,i=1).

So for the 2x1x5 block, V rows 10,11,15,16 are already used by Strassen block 2! And W rows 0-7 are used by both Strassen blocks. This means the rank-1 term for 2x1x5 would interfere with the Strassen blocks.

I need to be more careful. The rank-1 term for 2x1x5 adds to V[10+k, r] and W[2k+i, r] which overlap with Strassen. So I can't just use a single rank-1 term.

Let me reconsider. The correct approach is that each "block" in the partition uses disjoint sets of (i,j), (j,k), (k,i) triples, but the rank-1 terms can overlap in their U, V, W entries. The issue is that when we add a rank-1 term, it contributes to ALL (i,j,k) triples where U[i,j,r], V[j,k,r], and W[k,i,r] are all nonzero.

So for the 2x1x5 rank-1 term at col 15:
- It contributes to (i, j, k) where U[i*3+2, 15]≠0, V[j*5+k, 15]≠0, W[k*2+i, 15]≠0.
- U[i*3+2,15]≠0 only for i=0,1 → j must be 2.
- V[j*5+k,15]≠0 only for j=2 → consistent.
- W[k*2+i,15]≠0 for all k,i.
- So it contributes 1 to T[i,2,k] for all i∈{0,1}, k∈{0..4}. ✓
- Does it contribute to any other (i,j,k)? Only if U[i*3+j,15]≠0 for j≠2, but U[i*3+j,15]=0 for j≠2. So no interference! ✓

Similarly for the 2x2x1 rank-1 term at col 14:
- U[i*3+j,14]≠0 for i∈{0,1}, j∈{0,1}
- V[j*5+4,14]≠0 for j∈{0,1}
- W[4*2+i,14]≠0 for i∈{0,1}
- Contributes to (i,j,k) where j∈{0,1} (from U and V), k=4 (from V), i∈{0,1} (from U and W).
- So contributes 1 to T[i,j,4] for i∈{0,1}, j∈{0,1}. ✓
- No interference with other triples. ✓

And the Strassen blocks don't use cols 14,15, so no interference there either.

So rank 16 should work! Let me verify my code is correct.

