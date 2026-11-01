"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # n=2, m=2, p=12
    # a in 0..3, b in 0..23, c in 0..23
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def time_left():
        return deadline - time.time()

    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-6, err

    def strassen_42():
        """
        Correct Strassen-based rank-42 decomposition for 2x2x12.
        Splits the 12 columns of B (and corresponding rows of C) into 6 pairs.
        Each pair is a 2x2x2 multiplication.
        A is shared.
        """
        rank = 42
        U = np.zeros((4, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        # Strassen terms for 2x2x2
        # Local A indices: 0=a00, 1=a01, 2=a10, 3=a11
        # Local B indices: 0=b00, 1=b01, 2=b10, 3=b11
        # Local C indices: 0=c00, 1=c01, 2=c10, 3=c11
        # C[k*n+i] -> for local 2x2, n=2.
        # c00 -> k=0, i=0 -> 0
        # c01 -> k=0, i=1 -> 1
        # c10 -> k=1, i=0 -> 2
        # c11 -> k=1, i=1 -> 3
        
        strassen_terms = [
            # (U entries, V entries, W entries)
            # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # M2 = (a10+a11)b00 -> c10 += 1
            ([(2,1),(3,1)], [(0,1)], [(2,1)]),
            # M3 = a00(b01-b11) -> c01 += 1
            ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
            # M4 = a11(b10-b00) -> c00 += 1, c10 += 1
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # M5 = (a00+a01)b11 -> c00 -= 1, c01 += 1
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # M6 = (a10-a00)(b00+b01) -> c11 += 1
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # M7 = (a01-a11)(b10+b11) -> c00 -= 1, c11 += 1
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1),(3,1)]),
        ]
        
        for block in range(6):
            # Block b handles k = 2*block, 2*block+1
            # V global indices for this block:
            # j=0, k=2b -> 0*12 + 2b = 2b
            # j=0, k=2b+1 -> 2b+1
            # j=1, k=2b -> 1*12 + 2b = 12+2b
            # j=1, k=2b+1 -> 12+2b+1
            
            # W global indices for this block:
            # k=2b, i=0 -> 2b*2 + 0 = 4b
            # k=2b, i=1 -> 2b*2 + 1 = 4b+1
            # k=2b+1, i=0 -> (2b+1)*2 + 0 = 4b+2
            # k=2b+1, i=1 -> (2b+1)*2 + 1 = 4b+3
            
            v_map = [2*block, 2*block+1, 12+2*block, 12+2*block+1]
            w_map = [4*block, 4*block+1, 4*block+2, 4*block+3]
            
            for s in range(7):
                col = block * 7 + s
                u_e, v_e, w_e = strassen_terms[s]
                
                for idx, coeff in u_e:
                    U[idx, col] = coeff
                
                for idx, coeff in v_e:
                    V[v_map[idx], col] = coeff
                    
                for idx, coeff in w_e:
                    W[w_map[idx], col] = coeff
                    
        return U, V, W

    # Get the rank-42 solution
    U42, V42, W42 = strassen_42()
    valid42, err42 = verify(U42, V42, W42)
    
    if valid42:
        # Try to reduce rank by removing linearly dependent columns?
        # For now, return the valid 42.
        # We can attempt a simple check if any column is redundant by solving a linear system,
        # but that's complex. 42 is a significant improvement over 48.
        
        # Quick check: can we do better with a different tiling?
        # E.g. 2x2x6 x 2? No.
        
        return {
            "U": U42.tolist(),
            "V": V42.tolist(),
            "W": W42.tolist()
        }
    else:
        # Fallback: Schoolbook rank 48
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
# EVOLVE-BLOCK-END
