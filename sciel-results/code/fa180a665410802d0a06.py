import time
import numpy as np

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    t0 = time.time()
    deadline = t0 + 115.0

    # Build the decomposition by recursively applying Strassen.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # A decomposition of rank r gives U, V, W where:
    # U: shape (16, r) -- indices (i*m+j) for i,j in {0,1,2,3}, m=4
    # V: shape (16, r) -- indices (j*p+k) for j,k in {0,1,2,3}, p=4
    # W: shape (16, r) -- indices (k*n+i) for k,i in {0,1,2,3}, n=4
    # with T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    terms = []  # list of (u_dict, v_dict, w_dict)

    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    # Strassen 2x2 inner decomposition for a product of two 2x2 matrices
    # Standard Strassen 2x2:
    # M1 = (A11+A22)(B11+B22): C11 += M1, C22 += M1
    # M2 = (A21+A22)(B11):     C21 += M2, C22 -= M2
    # M3 = (A11)(B12-B22):     C12 += M3, C22 += M3
    # M4 = (A22)(B21-B11):     C11 += M4, C21 += M4
    # M5 = (A11+A12)(B22):     C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12): C22 += M6
    # M7 = (A12-A22)(B21+B22): C11 += M7

    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
    ]

    # Outer Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00):     C10 += M2, C11 -= M2
    # M3 = (A00)(B01-B11):     C01 += M3, C11 += M3
    # M4 = (A11)(B10-B00):     C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11):     C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01): C11 += M6
    # M7 = (A01-A11)(B10+B11): C00 += M7

    outer_strassen = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
    ]

    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_strassen):
        for inner_idx, (a_comb, b_comb, c_comb) in enumerate(strassen_2x2):
            u_dict = {}
            v_dict = {}
            w_dict = {}

            for (lr, lc, ca) in a_comb:
                for (br, bc, coeff) in a_blocks:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    u_idx = gr * 4 + gc
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff

            for (lr, lc, cb) in b_comb:
                for (br, bc, coeff) in b_blocks:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    v_idx = gr * 4 + gc
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff

            for (lr, lc, cc) in c_comb:
                for (br, bc, coeff) in c_blocks:
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    w_idx = gc * 4 + gr
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff

            add_term(u_dict, v_dict, w_dict)

    rank = len(terms)
    assert rank == 49, f"Expected rank 49, got {rank}"

    # Build U, V, W matrices as numpy arrays for efficiency
    U = np.zeros((16, rank), dtype=np.int8)
    V = np.zeros((16, rank), dtype=np.int8)
    W = np.zeros((16, rank), dtype=np.int8)

    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            if val != 0:
                U[idx, col] = val
        for idx, val in v_dict.items():
            if val != 0:
                V[idx, col] = val
        for idx, val in w_dict.items():
            if val != 0:
                W[idx, col] = val

    # Verify the decomposition using exact integer arithmetic
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # This should equal 1 for all i,j,k in {0,1,2,3}.

    is_valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :]))
                if s != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if not is_valid:
        # Fallback to schoolbook if verification fails
        rank = 64
        U = np.zeros((16, rank), dtype=np.int8)
        V = np.zeros((16, rank), dtype=np.int8)
        W = np.zeros((16, rank), dtype=np.int8)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1

    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
