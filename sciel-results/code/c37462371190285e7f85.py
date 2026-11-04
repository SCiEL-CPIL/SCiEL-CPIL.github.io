import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    """Rank-189 exact tensor decomposition of 6x6 matrix multiplication.

    Structure: 2x2 Strassen (rank 7) outer block grid, each block is a 3x3
    schoolbook product (rank 27). Total rank = 7 * 27 = 189.
    Convention: W[k*n+i, r] (output indexed by column-major of (k,i)).
    """
    # --- 2x2 Strassen tensor (rank 7) ---
    # Each term: U (coeff on A entries), V (coeff on B entries), W (coeff on C entries)
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A*B
    # C[0,0]=a00b00+a01b10, C[0,1]=a00b01+a01b11, C[1,0]=a10b00+a11b10, C[1,1]=a10b01+a11b11

    # Strassen 7 multiplications
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b00-b01)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    sU = np.zeros((4, 7), dtype=np.float64)
    sV = np.zeros((4, 7), dtype=np.float64)
    sW = np.zeros((4, 7), dtype=np.float64)

    # Term 0: M1 = (a00+a11)(b00+b11)
    sU[0, 0] = 1; sU[3, 0] = 1
    sV[0, 0] = 1; sV[3, 0] = 1
    sW[0, 0] = 1; sW[3, 0] = 1

    # Term 1: M2 = (a10+a11)b00
    sU[2, 1] = 1; sU[3, 1] = 1
    sV[0, 1] = 1
    sW[2, 1] = 1; sW[3, 1] = -1

    # Term 2: M3 = a00(b10-b11)
    sU[0, 2] = 1
    sV[2, 2] = 1; sV[3, 2] = -1
    sW[1, 2] = 1; sW[3, 2] = 1

    # Term 3: M4 = a11(b00-b01)
    sU[3, 3] = 1
    sV[0, 3] = 1; sV[1, 3] = -1
    sW[0, 3] = 1; sW[2, 3] = 1

    # Term 4: M5 = (a00+a01)b11
    sU[0, 4] = 1; sU[1, 4] = 1
    sV[3, 4] = 1
    sW[0, 4] = -1; sW[1, 4] = 1

    # Term 5: M6 = (a10-a00)(b00+b01)
    sU[2, 5] = 1; sU[0, 5] = -1
    sV[0, 5] = 1; sV[1, 5] = 1
    sW[3, 5] = 1

    # Term 6: M7 = (a01-a11)(b10+b11)
    sU[1, 6] = 1; sU[3, 6] = -1
    sV[2, 6] = 1; sV[3, 6] = 1
    sW[0, 6] = 1

    # --- 3x3 schoolbook tensor (rank 27) ---
    # Built via one-hot Kronecker: for each (i,j,k) in {0,1,2}^3,
    # U has 1 at position i*3+j, V has 1 at j*3+k, W has 1 at k*3+i.
    bU = np.zeros((9, 27), dtype=np.float64)
    bV = np.zeros((9, 27), dtype=np.float64)
    bW = np.zeros((9, 27), dtype=np.float64)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bU[i * 3 + j, col] = 1.0
                bV[j * 3 + k, col] = 1.0
                bW[k * 3 + i, col] = 1.0
                col += 1

    rank = 7 * 27
    U = np.zeros((36, rank), dtype=np.float64)
    V = np.zeros((36, rank), dtype=np.float64)
    W = np.zeros((36, rank), dtype=np.float64)

    # Vectorized Kronecker-style construction
    # For outer term c_out and inner term c_in, combined column c = c_out*27 + c_in
    # U[row, c] = sum over (r_out, r_in) of sU[r_out, c_out] * bU[r_in, c_in]
    #   where row = (r_out//2)*3*6 + (r_out%2)*3*6 + (r_in//3)*6 + (r_in%3)
    # Simplify: row = ((r_out//2)*3 + (r_in//3))*6 + ((r_out%2)*3 + (r_in%3))

    # Precompute row indices for each (r_out, r_in) pair
    row_idx_U = np.zeros((4, 9), dtype=np.int64)
    row_idx_V = np.zeros((4, 9), dtype=np.int64)
    row_idx_W = np.zeros((4, 9), dtype=np.int64)

    for ro in range(4):
        io = ro // 2
        jo = ro % 2
        for ri in range(9):
            ii = ri // 3
            ji = ri % 3
            row_idx_U[ro, ri] = (io * 3 + ii) * 6 + (jo * 3 + ji)

            # For V: B indexed by (j, k)
            # ro corresponds to (j_out, k_out) in B
            jout = ro // 2
            kout = ro % 2
            # ri corresponds to (j_in, k_in)
            jin = ri // 3
            kin = ri % 3
            row_idx_V[ro, ri] = (jout * 3 + jin) * 6 + (kout * 3 + kin)

            # For W: C indexed by (k, i) per trace convention W[k*n+i, r]
            # ro corresponds to (k_out, i_out)
            kout_w = ro // 2
            iout_w = ro % 2
            # ri corresponds to (k_in, i_in)
            kin_w = ri // 3
            iin_w = ri % 3
            row_idx_W[ro, ri] = (kout_w * 3 + kin_w) * 6 + (iout_w * 3 + iin_w)

    # Build U, V, W using outer-product structure
    # U[:, c_out*27+c_in] = sum_{ro,ri} sU[ro,c_out]*bU[ri,c_in] * e_{row_idx_U[ro,ri]}
    # This is: for each c_out, the block U[:, c_out*27:(c_out+1)*27]
    #   has columns indexed by c_in, and each column is a linear combination.

    for c_out in range(7):
        if _remaining() < 5.0:
            break
        # u_block[c_in, row] = sum_{ro,ri} sU[ro,c_out]*bU[ri,c_in]*[row==row_idx_U[ro,ri]]
        # = sum_{ro,ri} sU[ro,c_out]*bU[ri,c_in] * e_{row_idx_U[ro,ri]}
        # Vectorize: for fixed c_out, sU_col = sU[:, c_out] (shape 4,)
        # For each c_in: vec = sum_{ro,ri} sU_col[ro]*bU[ri,c_in]*e_{row_idx_U[ro,ri]}
        # = sum_{ro} sU_col[ro] * (sum_{ri} bU[ri,c_in]*e_{row_idx_U[ro,ri]})
        # inner[ro, row] = sum_{ri} bU[ri,c_in]*[row==row_idx_U[ro,ri]]
        # So U[:, c_out*27+c_in] = sum_{ro} sU_col[ro] * inner[ro, :]

        sU_col = sU[:, c_out]
        sV_col = sV[:, c_out]
        sW_col = sW[:, c_out]

        for c_in in range(27):
            c = c_out * 27 + c_in

            # U column
            # For each ro: contribution is sU_col[ro] * sum_{ri: bU[ri,c_in]!=0} bU[ri,c_in] * e_{row_idx_U[ro,ri]}
            # Since bU is one-hot per column, only one ri is nonzero
            ri_U = np.argmax(bU[:, c_in])  # the one nonzero entry
            val_U = bU[ri_U, c_in]
            for ro in range(4):
                if sU_col[ro] != 0:
                    U[row_idx_U[ro, ri_U], c] += sU_col[ro] * val_U

            # V column
            ri_V = np.argmax(bV[:, c_in])
            val_V = bV[ri_V, c_in]
            for ro in range(4):
                if sV_col[ro] != 0:
                    V[row_idx_V[ro, ri_V], c] += sV_col[ro] * val_V

            # W column
            ri_W = np.argmax(bW[:, c_in])
            val_W = bW[ri_W, c_in]
            for ro in range(4):
                if sW_col[ro] != 0:
                    W[row_idx_W[ro, ri_W], c] += sW_col[ro] * val_W

    # Convert to integer lists (values are all ±1 or 0)
    U = np.rint(U).astype(np.int64)
    V = np.rint(V).astype(np.int64)
    W = np.rint(W).astype(np.int64)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
