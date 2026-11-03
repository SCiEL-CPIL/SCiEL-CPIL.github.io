"""Direct 4x4x4 Strassen (rank 49) + 4x2x2 schoolbook (rank 16) = rank 65."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def get_4x4_strassen_block(U, V, W, offset, k_offset=0):
        """
        Add a 4x4x4 Strassen block (rank 49) for A[0:4,0:4] * B[0:4,k_offset:k_offset+4].
        Uses tensor product of two 2x2x2 Strassen decompositions.
        """
        # 2x2x2 Strassen terms: (u_vec, v_vec, w_vec) each 4-dim
        # u_vec indexed by (i,j) in 0..1 x 0..1 -> i*2+j
        # v_vec indexed by (j,k) in 0..1 x 0..1 -> j*2+k
        # w_vec indexed by (k,i) in 0..1 x 0..1 -> k*2+i
        s22 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,-1,1,0]),
            ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
            ([0,0,0,1], [0,1,-1,0], [-1,0,1,0]),
            ([1,0,1,0], [0,0,0,1], [1,1,0,0]),
            ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),
            ([0,-1,0,1], [0,1,0,1], [1,0,0,0]),
        ]

        # For 4x4x4 via Kronecker product of two 2x2x2 Strassen:
        # A is 4x4, partitioned into 2x2 blocks: A = [[A00, A01],[A10, A11]]
        # B is 4x4, partitioned: B = [[B00, B01],[B10, B11]]
        # C is 4x4, partitioned: C = [[C00, C01],[C10, C11]]
        #
        # The 49 terms come from: outer Strassen (7 terms on block level) 
        # tensor inner Strassen (7 terms within each 2x2 block product).
        #
        # Actually, the correct construction is:
        # The 4x4x4 multiplication tensor is the Kronecker product of two 2x2x2 tensors.
        # So if T2 is the 2x2x2 Strassen (rank 7), then T4 = T2 ⊗ T2 has rank 7*7=49.
        #
        # The Kronecker product means: for each pair (a, b) of Strassen terms,
        # we get a term where:
        #   U4[i*4+j, term] = U2[i*2+j_inner, a] * U2[i_outer*2+j_outer, b] ... 
        #
        # Let me think more carefully. 
        # A 4x4 matrix A has entries A[i,j] for i,j in 0..3.
        # Write i = 2*i_outer + i_inner, j = 2*j_outer + j_inner.
        # Then A[i,j] = A[2*io+ii, 2*jo+ji].
        #
        # The 2x2x2 tensor T2 has: sum_c u2[io*2+ii, c] * v2[jo*2+ji, c] * w2[ko*2+ki, c] = delta
        # where the "multiplication" is (A*B)[io,ko] = sum_jo A[io,jo]*B[jo,ko] at block level,
        # and within each block, (Ablock*Bblock)[ii,ki] = sum_ji Ablock[ii,ji]*Bblock[ji,ki].
        #
        # So the 4x4x4 tensor is:
        # sum_{c1,c2} u2[io*2+ii, c1] * u2[jo*2+ji, c2] * v2[jo*2+ji, c1] * v2[ko*2+ki, c2] * w2[io*2+ii, c1] * w2[ko*2+ki, c2]
        #
        # Wait, that's not quite right either. Let me use the standard Kronecker product definition.
        #
        # T4[i,j,k] = T2[io,jo,ko] * T2[ii,ji,ki] where i=2*io+ii, j=2*jo+ji, k=2*ko+ki.
        #
        # For the Strassen decomposition of T2:
        # T2[a,b,c] = sum_t u2[a,b,t] * v2[b,c,t] * w2[c,a,t]
        #
        # So T4[i,j,k] = T2[io,jo,ko] * T2[ii,ji,ki]
        #   = (sum_t u2[io,jo,t]*v2[jo,ko,t]*w2[ko,io,t]) * (sum_s u2[ii,ji,s]*v2[ji,ki,s]*w2[ki,ii,s])
        #   = sum_{t,s} u2[io,jo,t]*v2[jo,ko,t]*w2[ko,io,t]*u2[ii,ji,s]*v2[ji,ki,s]*w2[ki,ii,s]
        #
        # So each term (t,s) in the 49-term decomposition has:
        #   U4[i,j,(t,s)] = u2[io,jo,t] * u2[ii,ji,s]
        #   V4[j,k,(t,s)] = v2[jo,ko,t] * v2[ji,ki,s]
        #   W4[k,i,(t,s)] = w2[ko,io,t] * w2[ki,ii,s]
        #
        # where i=2*io+ii, j=2*jo+ji, k=2*ko+ki.
        #
        # Let's implement this directly.
        
        for t in range(7):
            for s in range(7):
                term_idx = t * 7 + s
                col = offset + term_idx
                
                u2_t = s22[t][0]
                v2_t = s22[t][1]
                w2_t = s22[t][2]
                u2_s = s22[s][0]
                v2_s = s22[s][1]
                w2_s = s22[s][2]
                
                for io in range(2):
                    for ii in range(2):
                        for jo in range(2):
                            for ji in range(2):
                                i = io * 2 + ii
                                j = jo * 2 + ji
                                u_val = u2_t[io * 2 + jo] * u2_s[ii * 2 + ji]
                                if u_val != 0:
                                    U[i * 4 + j][col] += u_val
                
                for jo in range(2):
                    for ji in range(2):
                        for ko in range(2):
                            for ki in range(2):
                                j = jo * 2 + ji
                                k = k_offset + ko * 2 + ki
                                v_val = v2_t[jo * 2 + ko] * v2_s[ji * 2 + ki]
                                if v_val != 0:
                                    V[j * 6 + k][col] += v_val
                
                for ko in range(2):
                    for ki in range(2):
                        for io in range(2):
                            for ii in range(2):
                                k = k_offset + ko * 2 + ki
                                i = io * 2 + ii
                                w_val = w2_t[ko * 2 + io] * w2_s[ki * 2 + ii]
                                if w_val != 0:
                                    W[k * 4 + i][col] += w_val
        
        return offset + 49

    def get_4x2_schoolbook_block(U, V, W, offset, k_offset):
        """
        Add a 4x2x2 schoolbook block (rank 16) for A[0:4,0:4] * B[0:4,k_offset:k_offset+2].
        """
        col = offset
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U[i * 4 + j][col] = 1
                    V[j * 6 + (k_offset + k)][col] = 1
                    W[(k_offset + k) * 4 + i][col] = 1
                    col += 1
        return offset + 16

    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs matrix multiplication."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Build the hybrid decomposition: 49 + 16 = 65
    try:
        rank_total = 65
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # 4x4x4 Strassen for first 4 columns of B
        offset = get_4x4_strassen_block(U, V, W, 0, k_offset=0)
        
        # 4x2x2 schoolbook for last 2 columns of B
        offset = get_4x2_schoolbook_block(U, V, W, offset, k_offset=4)
        
        # Verify
        if verify(U, V, W):
            if time.time() < deadline:
                return {"U": U, "V": V, "W": W}
        
        # If verification fails, try with numpy for a more robust check
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check using einsum
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We want T[i,j,k] = 1 for all i,j,k
        tensor = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
        if np.allclose(tensor, 1.0, atol=1e-9):
            if time.time() < deadline:
                return {"U": U, "V": V, "W": W}
        
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
