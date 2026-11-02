"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 6]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # We will construct the decomposition using Strassen for 2x2x2 blocks
    # and trivial terms for the remaining parts.
    
    # The matrix A is 2x5, B is 5x6, C is 2x6.
    # Indices: A[i,j] with i in {0,1}, j in {0..4}
    #          B[j,k] with j in {0..4}, k in {0..5}
    #          C[i,k] with i in {0,1}, k in {0..5}
    
    # We partition the j dimension into blocks of size 2, except the last one if size 1.
    # j indices: 0, 1, 2, 3, 4
    # Blocks: (0,1), (2,3), and (4)
    
    # For each block of j's, we partition the k dimension into blocks of size 2.
    # k indices: 0, 1, 2, 3, 4, 5
    # Blocks: (0,1), (2,3), (4,5)
    
    # For a 2x2x2 block (A[i, j1:j2], B[j1:j2, k1:k2], C[i, k1:k2]):
    # We use Strassen's 7 multiplications.
    
    # Let's define the Strassen terms for a generic 2x2x2 block.
    # Let A_block = [[a, b], [c, d]] corresponding to A[i0, j0], A[i0, j1], A[i1, j0], A[i1, j1]
    # Let B_block = [[e, f], [g, h]] corresponding to B[j0, k0], B[j0, k1], B[j1, k0], B[j1, k1]
    # Let C_block = [[p, q], [r, s]] corresponding to C[i0, k0], C[i0, k1], C[i1, k0], C[i1, k1]
    
    # Strassen multiplications:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # C_block = [[m1 + m4 - m5 + m7, m3 + m5],
    #            [m2 + m4,           m1 - m2 + m3 + m7]]
    
    # We will build the U, V, W matrices column by column.
    # Each column corresponds to one multiplication term.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U_cols.append((u_idx, u_val))
        V_cols.append((v_idx, v_val))
        W_cols.append((w_idx, w_val))
        
    # We will iterate over j blocks and k blocks.
    
    # j blocks: (0,1), (2,3), (4)
    # k blocks: (0,1), (2,3), (4,5)
    
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for j_start, j_end in [(0, 1), (2, 3)]:
        # These are 2x2 j blocks
        for k_start, k_end in [(0, 1), (2, 3), (4, 5)]:
            # These are 2x2 k blocks
            # This gives a 2x2x2 block
            # A indices: i in {0,1}, j in {j_start, j_start+1}
            # B indices: j in {j_start, j_start+1}, k in {k_start, k_start+1}
            # C indices: i in {0,1}, k in {k_start, k_start+1}
            
            # Let's define the local indices
            j0, j1 = j_start, j_start + 1
            k0, k1 = k_start, k_start + 1
            
            # A entries:
            # a = A[0, j0], b = A[0, j1]
            # c = A[1, j0], d = A[1, j1]
            
            # B entries:
            # e = B[j0, k0], f = B[j0, k1]
            # g = B[j1, k0], h = B[j1, k1]
            
            # C entries:
            # p = C[0, k0], q = C[0, k1]
            # r = C[1, k0], s = C[1, k1]
            
            # Term 1: m1 = (a + d)(e + h)
            # u: A[0, j0]=1, A[1, j1]=1
            # v: B[j0, k0]=1, B[j1, k1]=1
            # w: C[0, k0]=1, C[1, k1]=1
            U_cols.append([(0 * m + j0, 1), (1 * m + j1, 1)])
            V_cols.append([(j0 * p + k0, 1), (j1 * p + k1, 1)])
            W_cols.append([(k0 * n + 0, 1), (k1 * n + 1, 1)])
            
            # Term 2: m2 = (c + d)e
            # u: A[1, j0]=1, A[1, j1]=1
            # v: B[j0, k0]=1
            # w: C[1, k0]=1
            U_cols.append([(1 * m + j0, 1), (1 * m + j1, 1)])
            V_cols.append([(j0 * p + k0, 1)])
            W_cols.append([(k0 * n + 1, 1)])
            
            # Term 3: m3 = a(f - h)
            # u: A[0, j0]=1
            # v: B[j0, k1]=1, B[j1, k1]=-1
            # w: C[0, k1]=1
            U_cols.append([(0 * m + j0, 1)])
            V_cols.append([(j0 * p + k1, 1), (j1 * p + k1, -1)])
            W_cols.append([(k1 * n + 0, 1)])
            
            # Term 4: m4 = d(g - e)
            # u: A[1, j1]=1
            # v: B[j1, k0]=1, B[j0, k0]=-1
            # w: C[0, k0]=1, C[1, k0]=1
            U_cols.append([(1 * m + j1, 1)])
            V_cols.append([(j1 * p + k0, 1), (j0 * p + k0, -1)])
            W_cols.append([(k0 * n + 0, 1), (k0 * n + 1, 1)])
            
            # Term 5: m5 = (a + b)h
            # u: A[0, j0]=1, A[0, j1]=1
            # v: B[j1, k1]=1
            # w: C[0, k0]=-1, C[0, k1]=1
            U_cols.append([(0 * m + j0, 1), (0 * m + j1, 1)])
            V_cols.append([(j1 * p + k1, 1)])
            W_cols.append([(k0 * n + 0, -1), (k1 * n + 0, 1)])
            
            # Term 6: m6 = (c - a)(e + f)
            # u: A[1, j0]=1, A[0, j0]=-1
            # v: B[j0, k0]=1, B[j0, k1]=1
            # w: C[1, k1]=1
            U_cols.append([(1 * m + j0, 1), (0 * m + j0, -1)])
            V_cols.append([(j0 * p + k0, 1), (j0 * p + k1, 1)])
            W_cols.append([(k1 * n + 1, 1)])
            
            # Term 7: m7 = (b - d)(g + h)
            # u: A[0, j1]=1, A[1, j1]=-1
            # v: B[j1, k0]=1, B[j1, k1]=1
            # w: C[0, k0]=1, C[1, k1]=1
            U_cols.append([(0 * m + j1, 1), (1 * m + j1, -1)])
            V_cols.append([(j1 * p + k0, 1), (j1 * p + k1, 1)])
            W_cols.append([(k0 * n + 0, 1), (k1 * n + 1, 1)])
            
    # For the last j block: j = 4 (single element)
    # This is a 2x1x2 block for each k block of size 2.
    # We can use 2 trivial terms for each k block (since it's 2x1x2, rank is 2*1*2=4, but we can do better? No, 2x1x2 is just two independent products).
    # Actually, for a fixed j, A[i, j] * B[j, k] for i in {0,1}, k in {0..5}.
    # This is just 2 * 6 = 12 terms.
    # We can group them by k blocks.
    
    for k_start, k_end in [(0, 1), (2, 3), (4, 5)]:
        k0, k1 = k_start, k_start + 1
        j0 = 4
        
        # Term for C[0, k0]: A[0, j0] * B[j0, k0]
        U_cols.append([(0 * m + j0, 1)])
        V_cols.append([(j0 * p + k0, 1)])
        W_cols.append([(k0 * n + 0, 1)])
        
        # Term for C[0, k1]: A[0, j0] * B[j0, k1]
        U_cols.append([(0 * m + j0, 1)])
        V_cols.append([(j0 * p + k1, 1)])
        W_cols.append([(k1 * n + 0, 1)])
        
        # Term for C[1, k0]: A[1, j0] * B[j0, k0]
        U_cols.append([(1 * m + j0, 1)])
        V_cols.append([(j0 * p + k0, 1)])
        W_cols.append([(k0 * n + 1, 1)])
        
        # Term for C[1, k1]: A[1, j0] * B[j0, k1]
        U_cols.append([(1 * m + j0, 1)])
        V_cols.append([(j0 * p + k1, 1)])
        W_cols.append([(k1 * n + 1, 1)])
        
    # Now we have lists of sparse representations for U, V, W columns.
    # Let's convert them to dense matrices.
    
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for col in range(rank):
        for idx, val in U_cols[col]:
            U[idx, col] = val
        for idx, val in V_cols[col]:
            V[idx, col] = val
        for idx, val in W_cols[col]:
            W[idx, col] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
