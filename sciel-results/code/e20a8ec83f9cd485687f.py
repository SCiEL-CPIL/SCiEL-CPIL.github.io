"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 53 decomposition using Strassen for 7 blocks of size 2 and schoolbook for 1 block of size 1
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_val(mat, idx, col, val):
        mat[idx][col] += val

    col = 0
    # 7 Strassen blocks
    for block in range(7):
        k_offset = block * 2
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        set_val(U, 0*2+0, col, 1)
        set_val(U, 1*2+1, col, 1)
        set_val(V, 0*p+k_offset+0, col, 1)
        set_val(V, 1*p+k_offset+1, col, 1)
        set_val(W, (k_offset+0)*n+0, col, 1)
        set_val(W, (k_offset+1)*n+1, col, 1)
        col += 1
        
        # M2: (A10+A11)B00 -> C10
        set_val(U, 1*2+0, col, 1)
        set_val(U, 1*2+1, col, 1)
        set_val(V, 0*p+k_offset+0, col, 1)
        set_val(W, (k_offset+0)*n+1, col, 1)
        col += 1
        
        # M3: A00(B01-B11) -> C01
        set_val(U, 0*2+0, col, 1)
        set_val(V, 0*p+k_offset+1, col, 1)
        set_val(V, 1*p+k_offset+1, col, -1)
        set_val(W, (k_offset+1)*n+0, col, 1)
        col += 1
        
        # M4: A11(B10-B00) -> C10
        set_val(U, 1*2+1, col, 1)
        set_val(V, 1*p+k_offset+0, col, 1)
        set_val(V, 0*p+k_offset+0, col, -1)
        set_val(W, (k_offset+0)*n+1, col, 1)
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        set_val(U, 0*2+0, col, 1)
        set_val(U, 0*2+1, col, 1)
        set_val(V, 1*p+k_offset+1, col, 1)
        set_val(W, (k_offset+0)*n+0, col, -1)
        set_val(W, (k_offset+1)*n+0, col, 1)
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        set_val(U, 1*2+0, col, 1)
        set_val(U, 0*2+0, col, -1)
        set_val(V, 0*p+k_offset+0, col, 1)
        set_val(V, 0*p+k_offset+1, col, 1)
        set_val(W, (k_offset+1)*n+1, col, 1)
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00, C11
        set_val(U, 0*2+1, col, 1)
        set_val(U, 1*2+1, col, -1)
        set_val(V, 1*p+k_offset+0, col, 1)
        set_val(V, 1*p+k_offset+1, col, 1)
        set_val(W, (k_offset+0)*n+0, col, 1)
        set_val(W, (k_offset+1)*n+1, col, 1)
        col += 1

    # 1 Schoolbook block for k=14
    k_offset = 14
    
    # A00 B00 -> C00
    set_val(U, 0*2+0, col, 1)
    set_val(V, 0*p+k_offset, col, 1)
    set_val(W, (k_offset)*n+0, col, 1)
    col += 1
    
    # A01 B10 -> C00
    set_val(U, 0*2+1, col, 1)
    set_val(V, 1*p+k_offset, col, 1)
    set_val(W, (k_offset)*n+0, col, 1)
    col += 1
    
    # A10 B00 -> C10
    set_val(U, 1*2+0, col, 1)
    set_val(V, 0*p+k_offset, col, 1)
    set_val(W, (k_offset)*n+1, col, 1)
    col += 1
    
    # A11 B10 -> C10
    set_val(U, 1*2+1, col, 1)
    set_val(V, 1*p+k_offset, col, 1)
    set_val(W, (k_offset)*n+1, col, 1)
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
