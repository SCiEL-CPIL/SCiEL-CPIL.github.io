"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Decompose into two 2x2 * 2x8 blocks.
    # Each 2x2 * 2x8 block is 4 independent 2x2 * 2x2 Strassen blocks.
    # Total rank = 2 * 4 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # Mapping for one Strassen block (2x2 * 2x2):
    # x indices: (i, j) in {0,1}x{0,1}
    # y indices: (j, k) in {0,1}x{0,1}
    # z indices: (i, k) in {0,1}x{0,1}
    
    # We have two main blocks: Block A (j in 0,1) and Block B (j in 2,3)
    # For each main block, we have 4 sub-blocks for k in 0..7 (step 2)
    
    col = 0
    for block_id in range(2):
        # block 0: j offset 0, block 1: j offset 2
        j_off = block_id * 2
        
        for k_block in range(4):
            # k offset for this 2x2 block
            k_off = k_block * 2
            
            # Process 7 Strassen multiplications
            # M1
            # x: (0,0), (1,1) -> U indices: (0, j_off+0), (1, j_off+1)
            # y: (0,0), (1,1) -> V indices: (j_off+0, k_off+0), (j_off+1, k_off+1)
            # Wait, V index is j*p + k. j is local 0 or 1, so global j is j_off + local_j.
            # k is local 0 or 1, so global k is k_off + local_k.
            
            # M1: (x00+x11)(y00+y11)
            U[0 * m + (j_off + 0)][col] = 1
            U[1 * m + (j_off + 1)][col] = 1
            V[(j_off + 0) * p + (k_off + 0)][col] = 1
            V[(j_off + 1) * p + (k_off + 1)][col] = 1
            # z00 += M1, z11 += M1
            # z00: i=0, k=k_off+0. W index: (k_off+0)*n + 0
            # z11: i=1, k=k_off+1. W index: (k_off+1)*n + 1
            W[(k_off + 0) * n + 0][col] = 1
            W[(k_off + 1) * n + 1][col] = 1
            col += 1
            
            # M2: (x10+x11)y00
            U[1 * m + (j_off + 0)][col] = 1
            U[1 * m + (j_off + 1)][col] = 1
            V[(j_off + 0) * p + (k_off + 0)][col] = 1
            # z10 += M2
            # z10: i=1, k=k_off+0. W index: (k_off+0)*n + 1
            W[(k_off + 0) * n + 1][col] = 1
            col += 1
            
            # M3: x00(y01-y11)
            U[0 * m + (j_off + 0)][col] = 1
            V[(j_off + 0) * p + (k_off + 1)][col] = 1
            V[(j_off + 1) * p + (k_off + 1)][col] = -1
            # z01 += M3, z11 += M3
            # z01: i=0, k=k_off+1. W index: (k_off+1)*n + 0
            # z11: i=1, k=k_off+1. W index: (k_off+1)*n + 1
            W[(k_off + 1) * n + 0][col] = 1
            W[(k_off + 1) * n + 1][col] = 1
            col += 1
            
            # M4: x11(y10-y00)
            U[1 * m + (j_off + 1)][col] = 1
            V[(j_off + 1) * p + (k_off + 0)][col] = 1
            V[(j_off + 0) * p + (k_off + 0)][col] = -1
            # z00 += M4, z10 += M4
            # z00: i=0, k=k_off+0. W index: (k_off+0)*n + 0
            # z10: i=1, k=k_off+0. W index: (k_off+0)*n + 1
            W[(k_off + 0) * n + 0][col] = 1
            W[(k_off + 0) * n + 1][col] = 1
            col += 1
            
            # M5: (x00+x01)y11
            U[0 * m + (j_off + 0)][col] = 1
            U[0 * m + (j_off + 1)][col] = 1
            V[(j_off + 1) * p + (k_off + 1)][col] = 1
            # z00 -= M5, z01 += M5
            # z00: i=0, k=k_off+0. W index: (k_off+0)*n + 0
            # z01: i=0, k=k_off+1. W index: (k_off+1)*n + 0
            W[(k_off + 0) * n + 0][col] = -1
            W[(k_off + 1) * n + 0][col] = 1
            col += 1
            
            # M6: (x10-x00)(y00+y01)
            U[1 * m + (j_off + 0)][col] = 1
            U[0 * m + (j_off + 0)][col] = -1
            V[(j_off + 0) * p + (k_off + 0)][col] = 1
            V[(j_off + 0) * p + (k_off + 1)][col] = 1
            # z11 += M6
            # z11: i=1, k=k_off+1. W index: (k_off+1)*n + 1
            W[(k_off + 1) * n + 1][col] = 1
            col += 1
            
            # M7: (x01-x11)(y10+y11)
            U[0 * m + (j_off + 1)][col] = 1
            U[1 * m + (j_off + 1)][col] = -1
            V[(j_off + 1) * p + (k_off + 0)][col] = 1
            V[(j_off + 1) * p + (k_off + 1)][col] = 1
            # z00 += M7
            # z00: i=0, k=k_off+0. W index: (k_off+0)*n + 0
            W[(k_off + 0) * n + 0][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
