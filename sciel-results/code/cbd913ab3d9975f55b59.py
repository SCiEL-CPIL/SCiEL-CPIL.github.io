"""Program for 2x3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _decompose_sub(n, m, p, u_offset, v_offset, w_offset, U, V, W):
    """
    Recursively decomposes n x m x p multiplication.
    Returns the rank used.
    Appends to U, V, W at given offsets.
    """
    if n == 0 or m == 0 or p == 0:
        return 0
    
    # Base case: Schoolbook
    if n == 1 and m == 1 and p == 1:
        U[u_offset][0] = 1.0
        V[v_offset][0] = 1.0
        W[w_offset][0] = 1.0
        return 1
        
    # Strategy: Split P dimension if p > 1
    # This creates independent sub-problems for each part of B's columns
    # C = A * B = [A * B1, A * B2, ...]
    if p > 1:
        # Split p into p1 and p2
        p1 = p // 2
        p2 = p - p1
        
        rank1 = _decompose_sub(n, m, p1, u_offset, v_offset, w_offset, U, V, W)
        # For the second part, V indices shift by p1 (since V is m*p, but we are decomposing sub-problems)
        # Wait, the indices in U, V, W are global for the sub-problem.
        # U index: i*m + j. Unchanged by p split.
        # V index: j*p + k. For B2, k starts at p1.
        # W index: k*n + i. For C2, k starts at p1.
        
        # We need to pass the correct offsets.
        # U is shared? No, U is for A. A is the same.
        # But in tensor decomposition, we can reuse U columns?
        # No, each multiplication term is a rank-1 tensor.
        # If we split B, we are effectively doing:
        # T = T1 + T2
        # T1 uses A and B1. T2 uses A and B2.
        # The U matrix for T1 and T2 will be the same A part?
        # In the tensor sum $\sum r_k u^{(k)} \otimes v^{(k)} \otimes w^{(k)}$,
        # if we add two decompositions, we just concatenate the columns.
        # So U for part 2 is the same as U for part 1?
        # No, the coefficients might be different, but the structure is the same.
        # Actually, if we use the same decomposition for $A B_1$ and $A B_2$,
        # the U vectors are identical.
        # However, we cannot merge them unless we factor out.
        # Standard tensor rank addition: Rank(T1 + T2) <= Rank(T1) + Rank(T2).
        # We just append columns.
        
        # So for part 2:
        # U indices are the same relative to the sub-problem A.
        # But we are appending to the global U.
        # So we need to copy the U structure?
        # No, we can just call the function again, but it will overwrite U if we don't offset.
        # We must offset U as well?
        # Wait. If I decompose $A B_1$, I get $r_1$ columns in U.
        # If I decompose $A B_2$, I get $r_2$ columns in U.
        # The U columns for $A B_1$ correspond to A.
        # The U columns for $A B_2$ correspond to A.
        # They are the SAME U vectors!
        # Can we reuse them?
        # $C = \sum_{k=1}^{r_1} (u_k \cdot a) (v_{1,k} \cdot b_1) w_{1,k} + \sum_{k=1}^{r_2} (u_k \cdot a) (v_{2,k} \cdot b_2) w_{2,k}$
        # This is not a simple sum of rank-1 tensors unless we expand.
        # $\sum u_k \otimes v_{1,k} \otimes w_{1,k} + \sum u_k \otimes v_{2,k} \otimes w_{2,k}$.
        # This is valid. We just have $r_1 + r_2$ terms.
        # The U vector for term $j$ in the first sum is $u_j$.
        # The U vector for term $j$ in the second sum is $u_j$.
        # So yes, we append the same U columns again.
        
        # So:
        # Part 1: U[u_offset : u_offset + rank1], V[v_offset : v_offset + rank1], W[w_offset : w_offset + rank1]
        # Part 2: U[u_offset + rank1 : u_offset + rank1 + rank2], ...
        # But the U values are the same.
        
        # Let's refine the recursive step.
        # We need to know the "local" indices.
        
        # This recursion is getting complicated with offsets.
        # Let's just return the matrices for the sub-problem and merge them.
        pass

    return 0

def construct():
    n, m, p = 2, 3, 6
    
    # We will use the P-split strategy.
    # Split p=6 into 3 blocks of 2.
    # Each block is 2x3x2.
    # For 2x3x2:
    # Split m=3 into 1 and 2.
    # Block 1: 2x1x2 (Rank 4)
    # Block 2: 2x2x2 (Rank 7)
    # Total per 2x3x2 block: 11.
    # 3 blocks * 11 = 33.
    
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1
        
    # This is still hardcoded logic.
    # The "New Approach" should be more generic.
    # But for now, let's just implement the P-split logic cleanly.
    
    for block in range(3):
        # B block is columns 2*block, 2*block+1
        # C block is columns 2*block, 2*block+1
        
        # Part 1: A[:, 0] * B[0, block_cols]
        # A[:, 0] is 2x1. B[0, block_cols] is 1x2.
        # Rank 4.
        # Terms:
        # i=0, j=0, k=0 -> A00 B0k0 C0k0
        # i=1, j=0, k=0 -> A10 B0k0 C1k0
        # i=0, j=0, k=1 -> A00 B0k1 C0k1
        # i=1, j=0, k=1 -> A10 B0k1 C1k1
        
        for k_local in range(2):
            k_global = 2 * block + k_local
            for i in range(2):
                u_idx = i * m + 0
                v_idx = 0 * p + k_global
                w_idx = k_global * n + i
                add_term(u_idx, v_idx, w_idx, 1.0)
        
        # Part 2: A[:, 1:3] * B[1:3, block_cols]
        # A is 2x2, B is 2x2.
        # Strassen Rank 7.
        # A indices: (0,1), (0,2), (1,1), (1,2)
        # B indices: (1, 2*block), (1, 2*block+1), (2, 2*block), (2, 2*block+1)
        
        a00 = 0 * m + 1
        a01 = 0 * m + 2
        a10 = 1 * m + 1
        a11 = 1 * m + 2
        
        b00 = 1 * p + 2 * block
        b01 = 1 * p + 2 * block + 1
        b10 = 2 * p + 2 * block
        b11 = 2 * p + 2 * block + 1
        
        c00 = (2 * block) * n + 0
        c01 = (2 * block + 1) * n + 0
        c10 = (2 * block) * n + 1
        c11 = (2 * block + 1) * n + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[a00][col] = 1.0; U[a11][col] = 1.0
        V[b00][col] = 1.0; V[b11][col] = 1.0
        W[c00][col] = 1.0; W[c11][col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10][col] = 1.0; U[a11][col] = 1.0
        V[b00][col] = 1.0
        W[c10][col] = 1.0; W[c11][col] = -1.0
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00][col] = 1.0
        V[b01][col] = 1.0; V[b11][col] = -1.0
        W[c01][col] = 1.0; W[c11][col] = 1.0
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11][col] = 1.0
        V[b10][col] = 1.0; V[b00][col] = -1.0
        W[c00][col] = 1.0; W[c10][col] = 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00][col] = 1.0; U[a01][col] = 1.0
        V[b11][col] = 1.0
        W[c00][col] = -1.0; W[c01][col] = 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10][col] = 1.0; U[a00][col] = -1.0
        V[b00][col] = 1.0; V[b01][col] = 1.0
        W[c11][col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01][col] = 1.0; U[a11][col] = -1.0
        V[b10][col] = 1.0; V[b11][col] = 1.0
        W[c00][col] = 1.0
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
