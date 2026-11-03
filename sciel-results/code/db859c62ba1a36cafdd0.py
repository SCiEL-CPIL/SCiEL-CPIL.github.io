import numpy as np
import time

N, M, P = 2, 4, 5
NM, MP, PN = N * M, M * P, P * N

def construct():
    t0 = time.time()
    deadline = t0 + 100.0

    def build_schoolbook():
        r = N * M * P
        U = np.zeros((NM, r))
        V = np.zeros((MP, r))
        W = np.zeros((PN, r))
        idx = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, idx] = 1.0
                    V[j * P + k, idx] = 1.0
                    W[k * N + i, idx] = 1.0
                    idx += 1
        return U, V, W, r

    def validate(U, V, W):
        r = U.shape[1]
        # Efficient validation using einsum
        # T_ijk = sum_r U_{iM+j, r} V_{jP+k, r} W_{kN+i, r}
        # We need T_ijk = 1 if (i,j,k) is a valid index triple, 0 otherwise.
        # Actually, the tensor is sparse.
        # U shape (8, R), V shape (20, R), W shape (10, R)
        # Result shape (8, 20, 10)
        
        # To speed up validation, we can check specific slices.
        # But for R=36, full einsum is fast enough.
        result = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check non-zero entries
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    if abs(result[i * M + j, j * P + k, k * N + i] - 1.0) > 1e-8:
                        return False
                    # Check zeros
                    # We can't check all zeros easily without a mask, 
                    # but we can check a few random ones or rely on the structure.
                    # Actually, the structure guarantees zeros if the Strassen patterns are correct.
                    # Let's check a sample of zeros to be safe.
        
        # Check zeros:
        # Any (a,b,c) where a=i*M+j, b=j'*P+k, c=k'*N+i'
        # is non-zero only if j==j', k==k', i==i'.
        # We can verify this by checking that for any fixed a, the slice result[a,:,:]
        # is zero except where b corresponds to j of a and c corresponds to i of a.
        
        for a in range(NM):
            i = a // M
            j = a % M
            for b in range(MP):
                j_b = b // P
                k_b = b % P
                if j_b != j:
                    if np.any(np.abs(result[a, b, :]) > 1e-8):
                        return False
                else:
                    for c in range(PN):
                        k_c = c // N
                        i_c = c % N
                        if k_c != k_b or i_c != i:
                            if abs(result[a, b, c]) > 1e-8:
                                return False
        return True

    def to_output(U, V, W, r):
        def conv(mat):
            # Ensure integers or half-integers
            return [[float(mat[a, b]) for b in range(r)] for a in range(mat.shape[0])]
        return {"U": conv(U), "V": conv(V), "W": conv(W)}

    # Rank-36 construction
    # Strassen 2x2x2 patterns
    strassen_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # 2x2x1 patterns (for the 5th column of B)
    pat_221 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    R36 = 36
    U = np.zeros((NM, R36))
    V = np.zeros((MP, R36))
    W = np.zeros((PN, R36))
    col = 0
    
    # Blocks:
    # A is 2x4. Split into A1 (cols 0-1) and A2 (cols 2-3).
    # B is 4x5. Split into B1 (rows 0-1) and B2 (rows 2-3).
    # C = A1 B1 + A2 B2.
    # A1 B1: 2x2 * 2x5.
    # A2 B2: 2x2 * 2x5.
    
    # We process A1 B1 first.
    # A1 is rows 0-1 of A? No, A is 2x4.
    # i=0,1. j=0..3.
    # A1 corresponds to j=0,1. A2 corresponds to j=2,3.
    # B1 corresponds to k=0,1? No, B is 4x5.
    # B rows are j=0..3.
    # So B1 is rows 0,1 of B (which multiply with A cols 0,1).
    # B2 is rows 2,3 of B (which multiply with A cols 2,3).
    
    # Wait, the multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So for fixed i, we sum over j.
    # The tensor T_{i,j,k} has support on all (i,j,k).
    # The decomposition separates the j index.
    # Term 1: j=0,1. A[i, 0:2] * B[0:2, k].
    # Term 2: j=2,3. A[i, 2:4] * B[2:4, k].
    
    # So we need to decompose:
    # 1. (2x2) * (2x5) -> 2x5.
    # 2. (2x2) * (2x5) -> 2x5.
    
    # Let's build one (2x2)*(2x5) block.
    # A_block is 2x2 (i=0,1; j_local=0,1).
    # B_block is 2x5 (j_local=0,1; k=0..4).
    # C_block is 2x5 (i=0,1; k=0..4).
    
    # Decompose (2x2)*(2x5):
    # Split B_block into B_01 (2x2, k=0,1) and B_23 (2x2, k=2,3) and B_4 (2x1, k=4).
    # Rank = Rank(2x2*2x2) + Rank(2x2*2x2) + Rank(2x2*2x1)
    # Rank = 7 + 7 + 4 = 18.
    
    # Total Rank = 18 (for j=0,1) + 18 (for j=2,3) = 36.
    
    def add_block_2x2x5(i_offset, j_offset, col_offset):
        # i_offset: 0 or 1 (row in A)
        # j_offset: 0 or 2 (col block in A / row block in B)
        
        # Sub-block 1: k=0,1
        # A indices: (i_offset, j_offset), (i_offset, j_offset+1)
        # B indices: (j_offset, 0), (j_offset+1, 0), (j_offset, 1), (j_offset+1, 1)
        # C indices: (0, 0), (1, 0), (0, 1), (1, 1) -> k*n+i
        
        for up, vp, wp in strassen_222:
            for ix, val in enumerate(up):
                if val != 0:
                    il = ix // 2
                    jl = ix % 2
                    U[(i_offset + il) * M + (j_offset + jl), col_offset] = val
            for ix, val in enumerate(vp):
                if val != 0:
                    jl = ix // 2
                    kl = ix % 2
                    V[(j_offset + jl) * P + kl, col_offset] = val
            for ix, val in enumerate(wp):
                if val != 0:
                    il = ix // 2
                    kl = ix % 2
                    W[kl * N + (i_offset + il), col_offset] = val
            col_offset += 1
            
        # Sub-block 2: k=2,3
        for up, vp, wp in strassen_222:
            for ix, val in enumerate(up):
                if val != 0:
                    il = ix // 2
                    jl = ix % 2
                    U[(i_offset + il) * M + (j_offset + jl), col_offset] = val
            for ix, val in enumerate(vp):
                if val != 0:
                    jl = ix // 2
                    kl = ix % 2
                    V[(j_offset + jl) * P + 2 + kl, col_offset] = val
            for ix, val in enumerate(wp):
                if val != 0:
                    il = ix // 2
                    kl = ix % 2
                    W[(2 + kl) * N + (i_offset + il), col_offset] = val
            col_offset += 1
            
        # Sub-block 3: k=4
        for up, vp, wp in pat_221:
            for ix, val in enumerate(up):
                if val != 0:
                    il = ix // 2
                    jl = ix % 2
                    U[(i_offset + il) * M + (j_offset + jl), col_offset] = val
            for ix, val in enumerate(vp):
                if val != 0:
                    jl = ix
                    V[(j_offset + jl) * P + 4, col_offset] = val
            for ix, val in enumerate(wp):
                if val != 0:
                    il = ix
                    W[4 * N + (i_offset + il), col_offset] = val
            col_offset += 1
            
        return col_offset

    # Block 1: i=0, j=0..1
    col = add_block_2x2x5(0, 0, col)
    # Block 2: i=1, j=0..1
    col = add_block_2x2x5(1, 0, col)
    # Block 3: i=0, j=2..3
    col = add_block_2x2x5(0, 2, col)
    # Block 4: i=1, j=2..3
    col = add_block_2x2x5(1, 2, col)
    
    assert col == R36, f"Col mismatch: {col}"
    
    if validate(U, V, W):
        return to_output(U, V, W, R36)

    # Fallback
    Us, Vs, Ws, rs = build_schoolbook()
    return to_output(Us, Vs, Ws, rs)
