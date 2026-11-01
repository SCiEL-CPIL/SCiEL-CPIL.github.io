"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 11 = 44
    rank = 44
    
    # U shape: (n*m, rank) = (6, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (16, 44)
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Define the 11 terms for a 2x3x2 block
    # Each term is a tuple (U_vec, V_vec_template, W_vec_template)
    # U_vec is length 6
    # V_vec_template is a dict {offset: value} where offset is relative to block start in V
    # W_vec_template is a dict {offset: value} where offset is relative to block start in W
    
    # Indices in U (i*m + j):
    # a: 0, b: 1, c: 2
    # d: 3, e: 4, f: 5
    
    # Indices in V (j*p + k):
    # For block b, k is 2b or 2b+1.
    # j=0: 0*8 + 2b = 2b, 0*8 + 2b+1 = 2b+1
    # j=1: 1*8 + 2b = 8+2b, 1*8 + 2b+1 = 9+2b
    # j=2: 2*8 + 2b = 16+2b, 2*8 + 2b+1 = 17+2b
    
    # Indices in W (k*n + i):
    # For block b, k is 2b or 2b+1.
    # k=2b: 2b*2 + 0 = 4b, 2b*2 + 1 = 4b+1
    # k=2b+1: (2b+1)*2 + 0 = 4b+2, (2b+1)*2 + 1 = 4b+3
    
    terms = []
    
    # Term 1: M1 = (a+e)(x+w)
    # U: a(0)=1, e(4)=1
    # V: x(j=0,k=2b)=1, w(j=1,k=2b+1)=1 -> offsets 0, 9
    # W: C00(0)=1, C01(1)=1, C10(2)=-1
    terms.append(([1,0,0,0,1,0], {0:1, 9:1}, {0:1, 1:1, 2:-1}))
    
    # Term 2: M2 = (d+e)x
    # U: d(3)=1, e(4)=1
    # V: x(j=0,k=2b)=1 -> offset 0
    # W: C10(2)=1, C11(3)=1
    terms.append(([0,0,0,1,1,0], {0:1}, {2:1, 3:1}))
    
    # Term 3: M3 = a(y-w)
    # U: a(0)=1
    # V: y(j=0,k=2b+1)=1, -w(j=1,k=2b+1)=-1 -> offsets 1, 9
    # W: C01(1)=1, C11(3)=1
    terms.append(([1,0,0,0,0,0], {1:1, 9:-1}, {1:1, 3:1}))
    
    # Term 4: M4 = e(z-x)
    # U: e(4)=1
    # V: z(j=1,k=2b)=1, -x(j=0,k=2b)=-1 -> offsets 8, 0
    # W: C00(0)=1, C10(2)=1, C11(3)=-1
    terms.append(([0,0,0,0,1,0], {8:1, 0:-1}, {0:1, 2:1, 3:-1}))
    
    # Term 5: M5 = (a+d)(x+z)
    # U: a(0)=1, d(3)=1
    # V: x(j=0,k=2b)=1, z(j=1,k=2b)=1 -> offsets 0, 8
    # W: C01(1)=-1, C10(2)=1
    terms.append(([1,0,0,1,0,0], {0:1, 8:1}, {1:-1, 2:1}))
    
    # Term 6: M6 = (b+e)(w+z)
    # U: b(1)=1, e(4)=1
    # V: w(j=1,k=2b+1)=1, z(j=1,k=2b)=1 -> offsets 9, 8
    # W: C00(0)=-1, C11(3)=1
    terms.append(([0,1,0,0,1,0], {9:1, 8:1}, {0:-1, 3:1}))
    
    # Term 7: M7 = (b+d)(y-x)
    # U: b(1)=1, d(3)=1
    # V: y(j=0,k=2b+1)=1, -x(j=0,k=2b)=-1 -> offsets 1, 0
    # W: C00(0)=1, C01(1)=1
    terms.append(([0,1,0,1,0,0], {1:1, 0:-1}, {0:1, 1:1}))
    
    # Term 8: M8 = c u
    # U: c(2)=1
    # V: u(j=2,k=2b)=1 -> offset 16
    # W: C00(0)=1
    terms.append(([0,0,1,0,0,0], {16:1}, {0:1}))
    
    # Term 9: M9 = c v
    # U: c(2)=1
    # V: v(j=2,k=2b+1)=1 -> offset 17
    # W: C01(1)=1
    terms.append(([0,0,1,0,0,0], {17:1}, {1:1}))
    
    # Term 10: M10 = f u
    # U: f(5)=1
    # V: u(j=2,k=2b)=1 -> offset 16
    # W: C10(2)=1
    terms.append(([0,0,0,0,0,1], {16:1}, {2:1}))
    
    # Term 11: M11 = f v
    # U: f(5)=1
    # V: v(j=2,k=2b+1)=1 -> offset 17
    # W: C11(3)=1
    terms.append(([0,0,0,0,0,1], {17:1}, {3:1}))
    
    # Fill the matrices
    for b in range(4):
        base_rank = b * 11
        for t_idx, (u_vec, v_offsets, w_offsets) in enumerate(terms):
            col = base_rank + t_idx
            
            # Fill U
            for idx, val in enumerate(u_vec):
                if val != 0:
                    U[idx, col] = val
            
            # Fill V
            for offset, val in v_offsets.items():
                v_idx = offset + 2 * b  # Wait, V index is j*p + k.
                # My offsets were calculated assuming k=2b is 0 in the block?
                # Let's re-verify V index calculation.
                # V index = j * p + k.
                # For block b, k = 2b or 2b+1.
                # j=0: idx = 0*8 + 2b = 2b. Offset 0 -> 2b.
                # j=0: idx = 0*8 + 2b+1 = 2b+1. Offset 1 -> 2b+1.
                # j=1: idx = 1*8 + 2b = 8+2b. Offset 8 -> 8+2b.
                # j=1: idx = 1*8 + 2b+1 = 9+2b. Offset 9 -> 9+2b.
                # j=2: idx = 2*8 + 2b = 16+2b. Offset 16 -> 16+2b.
                # j=2: idx = 2*8 + 2b+1 = 17+2b. Offset 17 -> 17+2b.
                # So v_idx = offset + 2b is CORRECT.
                V[v_idx, col] = val
            
            # Fill W
            for offset, val in w_offsets.items():
                w_idx = offset + 4 * b  # W index = k*n + i.
                # For block b, k=2b or 2b+1.
                # k=2b: idx = 2b*2 + i = 4b + i. Offset 0,1 -> 4b, 4b+1.
                # k=2b+1: idx = (2b+1)*2 + i = 4b+2 + i. Offset 2,3 -> 4b+2, 4b+3.
                # So w_idx = offset + 4b is CORRECT.
                W[w_idx, col] = val
    
    # Convert to lists of floats
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
