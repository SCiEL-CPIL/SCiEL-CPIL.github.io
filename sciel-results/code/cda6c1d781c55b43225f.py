import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-10):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        for denom in [2, 4, 3, 6, 5, 10, 1, 8, 12, 15, 20]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None

    # Block decomposition using Strassen for 2x2x2 blocks
    # We partition the 5 columns of A and 5 rows of B into blocks of size 2, 2, 1
    # Block boundaries: 0-1, 2-3, 4
    # Same for rows of B and columns of C

    # The tensor decomposes into a direct sum of blocks because:
    # For a given (j_block, k_block), the v indices are disjoint from other (j_block, k_block)
    # v = j * 5 + k, so for fixed j_block and k_block, the set of v values is unique

    # Block types and their ranks:
    # 2x2x2 blocks: rank 7 (Strassen)
    # 2x2x1 blocks: rank 4 (trivial for 2x2 times 2x1)
    # 2x1x2 blocks: rank 4 (trivial for 2x1 times 1x2)
    # 2x1x1 blocks: rank 2 (trivial)

    # Let's enumerate the blocks:
    # j_blocks: [0,1], [2,3], [4] -> sizes 2, 2, 1
    # k_blocks: [0,1], [2,3], [4] -> sizes 2, 2, 1

    # Block (j_start, j_size, k_start, k_size)
    blocks = []
    j_starts = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_starts = [0, 2, 4]
    k_sizes = [2, 2, 1]

    for ji in range(3):
        for ki in range(3):
            js = j_starts[ji]
            jsz = j_sizes[ji]
            ks = k_starts[ki]
            ksz = k_sizes[ki]
            # Rank of this block
            if jsz == 2 and ksz == 2:
                rank = 7
            elif jsz == 2 and ksz == 1:
                rank = 4
            elif jsz == 1 and ksz == 2:
                rank = 4
            else:  # 1x1
                rank = 2
            blocks.append((js, jsz, ks, ksz, rank))

    total_rank = sum(b[4] for b in blocks)
    # 7+7+4 + 7+7+4 + 4+4+2 = 18 + 18 + 10 = 46

    U = np.zeros((U_dim, total_rank))
    V = np.zeros((V_dim, total_rank))
    W = np.zeros((W_dim, total_rank))

    col = 0
    for (js, jsz, ks, ksz, rank) in blocks:
        if jsz == 2 and ksz == 2:
            # Strassen decomposition for 2x2 * 2x2
            # A block: rows i=0,1; cols j=js, js+1
            # B block: rows j=js, js+1; cols k=ks, ks+1
            # C block: rows i=0,1; cols k=ks, ks+1
            
            # A[i, j] maps to u = i*m + j
            # B[j, k] maps to v = j*p + k
            # C[i, k] maps to w = k*n + i
            
            # Let a=A[0,js], b=A[0,js+1], c=A[1,js], d=A[1,js+1]
            # Let e=B[js,ks], f=B[js,ks+1], g=B[js+1,ks], h=B[js+1,ks+1]
            
            # Strassen:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # C[0,ks] = M1 + M4 - M5 + M7
            # C[0,ks+1] = M3 + M5
            # C[1,ks] = M2 + M4
            # C[1,ks+1] = M1 - M2 + M3 + M6
            
            # U entries: for each M, the u indices where A contributes
            # V entries: for each M, the v indices where B contributes
            # W entries: for each M, the w indices where C is produced
            
            # M1: (a+d)(e+h)
            # U: a at (0,js), d at (1,js+1) -> u=js, u=5+js+1
            # V: e at (js,ks), h at (js+1,ks+1) -> v=js*5+ks, v=(js+1)*5+ks+1
            # W: contributes to C[0,ks] (+1), C[1,ks+1] (+1) -> w=ks*2+0, w=(ks+1)*2+1
            
            # Let's define the 7 Strassen terms with their U, V, W contributions
            strassen_terms = [
                # (U_indices, U_values, V_indices, V_values, W_indices, W_values)
                # M1 = (a+d)(e+h)
                ([js, 5+js+1], [1, 1], [js*5+ks, (js+1)*5+ks+1], [1, 1], [ks*2+0, (ks+1)*2+1], [1, 1]),
                # M2 = (c+d)e
                ([5+js, 5+js+1], [1, 1], [js*5+ks], [1], [ks*2+1], [1]),
                # M3 = a(f-h)
                ([js], [1], [js*5+ks+1, (js+1)*5+ks+1], [1, -1], [(ks+1)*2+0], [1]),
                # M4 = d(g-e)
                ([5+js+1], [1], [(js+1)*5+ks, js*5+ks], [1, -1], [ks*2+0, ks*2+1], [1, 1]),
                # M5 = (a+b)h
                ([js, js+1], [1, 1], [(js+1)*5+ks+1], [1], [(ks+1)*2+0, (ks+1)*2+1], [-1, 1]),
                # M6 = (c-a)(e+f)
                ([5+js, js], [1, -1], [js*5+ks, js*5+ks+1], [1, 1], [(ks+1)*2+1], [1]),
                # M7 = (b-d)(g+h)
                ([js+1, 5+js+1], [1, -1], [(js+1)*5+ks, (js+1)*5+ks+1], [1, 1], [ks*2+0], [1]),
            ]
            
            for term in strassen_terms:
                u_idx, u_val, v_idx, v_val, w_idx, w_val = term
                for ui, uv in zip(u_idx, u_val):
                    U[ui, col] += uv
                for vi, vv in zip(v_idx, v_val):
                    V[vi, col] += vv
                for wi, wv in zip(w_idx, w_val):
                    W[wi, col] += wv
                col += 1
        
        elif jsz == 2 and ksz == 1:
            # 2x2 * 2x1 -> 2x1
            # A[0,js], A[0,js+1], A[1,js], A[1,js+1]
            # B[js,ks], B[js+1,ks]
            # C[0,ks], C[1,ks]
            # C[0,ks] = A[0,js]*B[js,ks] + A[0,js+1]*B[js+1,ks]
            # C[1,ks] = A[1,js]*B[js,ks] + A[1,js+1]*B[js+1,ks]
            
            # 4 terms
            terms = [
                # (U_idx, V_idx, W_idx)
                (js, js*5+ks, ks*2+0),
                (js+1, (js+1)*5+ks, ks*2+0),
                (5+js, js*5+ks, ks*2+1),
                (5+js+1, (js+1)*5+ks, ks*2+1),
            ]
            for u_i, v_i, w_i in terms:
                U[u_i, col] = 1
                V[v_i, col] = 1
                W[w_i, col] = 1
                col += 1
        
        elif jsz == 1 and ksz == 2:
            # 2x1 * 1x2 -> 2x2
            # A[0,js], A[1,js]
            # B[js,ks], B[js,ks+1]
            # C[0,ks], C[0,ks+1], C[1,ks], C[1,ks+1]
            # C[i,ks] = A[i,js]*B[js,ks]
            # C[i,ks+1] = A[i,js]*B[js,ks+1]
            
            terms = [
                (js, js*5+ks, ks*2+0),
                (js, js*5+ks+1, (ks+1)*2+0),
                (5+js, js*5+ks, ks*2+1),
                (5+js, js*5+ks+1, (ks+1)*2+1),
            ]
            for u_i, v_i, w_i in terms:
                U[u_i, col] = 1
                V[v_i, col] = 1
                W[w_i, col] = 1
                col += 1
        
        else:
            # 2x1 * 1x1 -> 2x1
            # A[0,js], A[1,js]
            # B[js,ks]
            # C[0,ks], C[1,ks]
            
            terms = [
                (js, js*5+ks, ks*2+0),
                (5+js, js*5+ks, ks*2+1),
            ]
            for u_i, v_i, w_i in terms:
                U[u_i, col] = 1
                V[v_i, col] = 1
                W[w_i, col] = 1
                col += 1

    # Validate
    valid, U_s, V_s, W_s = snap_and_validate(U, V, W)
    
    if not valid:
        # Fallback to trivial solution
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    return {
        "U": [[float(x) for x in row] for row in U_s],
        "V": [[float(x) for x in row] for row in V_s],
        "W": [[float(x) for x in row] for row in W_s]
    }
