"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Indices: i, j for U; j, k for V; k, i for W (local)
    # Term r (0-6)
    # U[i, j, r], V[j, k, r], W[k, i, r]
    
    # Term 1: m1 = (x00+x11)(y00+y11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    strassen_U = [
        [0, 1, 1, 0, 1, 1, 0, 1], # i=0, j=0
        [0, 0, 0, 0, 0, 0, 1, 0], # i=0, j=1
        [0, 1, 1, 0, 1, 1, 0, 1], # i=1, j=0  -- Wait, x10+x11 is term 2. Let's re-verify.
        [0, 0, 0, 1, 0, 0, 0, 0]  # i=1, j=1
    ]
    # Let's list them properly.
    # m1: x00+x11, y00+y11, z00+=1, z11+=1
    # m2: x10+x11, y00, z10+=1, z11-=1
    # m3: x00, y10-y11, z01+=1, z11+=1
    # m4: x11, y01-y00, z00+=1, z10+=1
    # m5: x00+x10, y11, z00-=1, z01+=1
    # m6: x10-x00, y00+y10, z11+=1
    # m7: x01-x11, y11+y01, z00+=1
    
    # U[i, j] coefficients for r=0..6
    # r0 (m1): i0j0=1, i1j1=1
    # r1 (m2): i1j0=1, i1j1=1
    # r2 (m3): i0j0=1
    # r3 (m4): i1j1=1
    # r4 (m5): i0j0=1, i1j0=1
    # r5 (m6): i1j0=1, i0j0=-1
    # r6 (m7): i0j1=1, i1j1=-1
    
    U_local = [
        [1, 0, 1, 0, 1, -1, 1], # i=0, j=0
        [0, 0, 0, 0, 0, 0, 1],  # i=0, j=1
        [0, 1, 0, 1, 1, 1, -1],# i=1, j=0
        [1, 1, 0, 1, 0, 0, -1] # i=1, j=1
    ]
    
    # V[j, k] coefficients
    # r0 (m1): j0k0=1, j1k1=1
    # r1 (m2): j0k0=1
    # r2 (m3): j1k0=1, j1k1=-1
    # r3 (m4): j0k1=1, j0k0=-1
    # r4 (m5): j1k1=1
    # r5 (m6): j0k0=1, j1k0=1
    # r6 (m7): j1k1=1, j0k1=1
    
    V_local = [
        [1, 1, 0, -1, 0, 1, 1], # j=0, k=0
        [0, 0, 0, 1, 0, 0, 1],  # j=0, k=1
        [0, 0, 1, 0, 1, 1, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    
    # W[k, i] coefficients
    # r0 (m1): k0i0=1, k1i1=1
    # r1 (m2): k0i1=1, k1i1=-1
    # r2 (m3): k1i0=1, k1i1=1
    # r3 (m4): k0i0=1, k0i1=1
    # r4 (m5): k0i0=-1, k1i0=1
    # r5 (m6): k1i1=1
    # r6 (m7): k0i0=1
    
    W_local = [
        [1, 1, 0, 1, -1, 0, 1], # k=0, i=0
        [0, -1, 1, 1, 1, 0, 0]  # k=1, i=1 -- Wait, k1i0?
    ]
    # Let's re-verify W indices. W[k*n + i]. Local k, i.
    # m1: z00 (k=0,i=0)=1, z11 (k=1,i=1)=1
    # m2: z10 (k=0,i=1)=1, z11 (k=1,i=1)=-1
    # m3: z01 (k=1,i=0)=1, z11 (k=1,i=1)=1
    # m4: z00 (k=0,i=0)=1, z10 (k=0,i=1)=1
    # m5: z00 (k=0,i=0)=-1, z01 (k=1,i=0)=1
    # m6: z11 (k=1,i=1)=1
    # m7: z00 (k=0,i=0)=1
    
    W_local = [
        [1, 1, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 1, 1, 0, 1, 0, 0]   # k=0, i=1
        # Wait, W shape is (p*n, rank). Local p=2, n=2.
        # Indices: k=0,1. i=0,1.
        # k=0, i=0: m1=1, m4=1, m5=-1, m7=1. Others 0.
        # k=0, i=1: m2=1, m4=1. Others 0.
        # k=1, i=0: m3=1, m5=1. Others 0.
        # k=1, i=1: m1=1, m2=-1, m3=1, m6=1. Others 0.
        
        [1, 0, 1, 0, 0, 0, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # Blocks:
    # Block 0: i_off=0, j_off=0, k_off=0
    # Block 1: i_off=0, j_off=2, k_off=0
    # Block 2: i_off=0, j_off=0, k_off=2
    # Block 3: i_off=0, j_off=2, k_off=2
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    base_rank = 0
    for b_idx, (i_off, j_off, k_off) in enumerate(blocks):
        for r in range(7):
            col = base_rank + r
            for i_loc in range(2):
                for j_loc in range(2):
                    val_u = U_local[i_loc][j_loc][r]
                    if val_u != 0:
                        i_glob = i_off + i_loc
                        j_glob = j_off + j_loc
                        U[i_glob * m + j_glob][col] += val_u
                        
            for j_loc in range(2):
                for k_loc in range(2):
                    val_v = V_local[j_loc][k_loc][r]
                    if val_v != 0:
                        j_glob = j_off + j_loc
                        k_glob = k_off + k_loc
                        V[j_glob * p + k_glob][col] += val_v
                        
            for k_loc in range(2):
                for i_loc in range(2):
                    val_w = W_local[k_loc][i_loc][r]
                    if val_w != 0:
                        k_glob = k_off + k_loc
                        i_glob = i_off + i_loc
                        W[k_glob * n + i_glob][col] += val_w
                        
        base_rank += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
