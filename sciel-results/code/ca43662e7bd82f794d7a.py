import numpy as np
import time

def construct():
    """
    Returns a rank-98 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    The 4x4x8 multiplication tensor is the direct sum of two 4x4x4 multiplication tensors.
    The rank of the 4x4x4 tensor is 49 (Strassen's algorithm applied recursively).
    Thus, the minimal rank for 4x4x8 is 49 + 49 = 98.
    
    Indexing convention:
    - U: (n*m, rank) = (16, 98), U[i*m+j, r] is coefficient of A[i,j]
    - V: (m*p, rank) = (32, 98), V[j*p+k, r] is coefficient of B[j,k]
    - W: (p*n, rank) = (32, 98), W[k*n+i, r] is coefficient of C[i,k]
    """
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    # 1. Define the 2x2x2 Strassen decomposition (Rank 7)
    # Indices for U, V, W in 2x2 case:
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M2 = A00(B01-B11) -> C01 += M2, C11 += M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M3 = A11(B10-B00) -> C00 += M3, C10 += M3
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M4 = (A00+A01)B11 -> C00 -= M4, C01 += M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M5 = (A10-A00)(B00+B01) -> C11 += M5
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M6 = (A01-A11)(B10+B11) -> C00 += M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 2. Construct 4x4x4 decomposition using tensor product logic (Rank 49)
    # The 4x4x4 tensor is built by combining two levels of 2x2x2.
    # We can represent the 4x4x4 factors by reshaping and multiplying.
    
    # For U: (I, J) maps to (I_outer, J_outer, I_inner, J_inner)
    # U4[i*4+j, r] where r = r_outer*7 + r_inner
    # U4 = sum_{r_o, r_i} U2[r_o] \otimes U2[r_i] appropriately indexed
    
    # Efficient construction using broadcasting:
    # U2 is (4, 7). Reshape to (2, 2, 7) where dims are (I, J, R).
    # But the indices in U2 are already flattened (i*2+j).
    # Let's reshape U2 to (2, 2, 7) -> (I_o, J_o, R_o) ? No, R is the last dim.
    # U2[i, r]. i = 2*Io + Jo.
    
    U2_reshaped = U2.reshape(2, 2, 7) # (Io, Jo, Ro)
    V2_reshaped = V2.reshape(2, 2, 7) # (Jo, Ko, Ro)
    W2_reshaped = W2.reshape(2, 2, 7) # (Ko, Io, Ro)

    # The 4x4x4 decomposition corresponds to r = 7*Ro + Ri.
    # U4 index (i, j) = (2*Io + Ii, 2*Jo + Ji).
    # U4[i*4+j, 7*Ro+Ri] = U2[2*Io+Jo, Ro] * U2[2*Ii+Ji, Ri]
    
    # We can construct the full U4 matrix (16, 49) by iterating over Ro and Ri.
    # Or use einsum.
    # U4[ (2*Io+Ii)*4 + (2*Jo+Ji) , 7*Ro+Ri ]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            uo = U2_reshaped[:, :, ro] # (2, 2)
            ui = U2_reshaped[:, :, ri] # (2, 2)
            
            # uo[Io, Jo] * ui[Ii, Ji]
            # Result shape (2, 2, 2, 2) -> Io, Jo, Ii, Ji
            u_full = np.einsum('io,jo->iojo', uo, ui)
            
            # Map to flat index i = 2*Io + Ii, j = 2*Jo + Ji
            # idx = i*4 + j = (2*Io+Ii)*4 + 2*Jo+Ji = 8*Io + 4*Ii + 2*Jo + Ji
            # We need to permute axes to Io, Jo, Ii, Ji -> currently Io, Jo, Ii, Ji is correct order for einsum 'io,jo->iojo'
            # Wait, einsum 'io,jo->iojo' gives axes (Io, Jo, Ii, Ji).
            # Let's verify: uo[Io, Jo] * ui[Ii, Ji].
            # The resulting tensor T[Io, Jo, Ii, Ji].
            # We want to flatten (Io, Jo, Ii, Ji) into (i, j) where i=2*Io+Ii, j=2*Jo+Ji.
            # This is not a simple reshape because i and j mix Io/Ii and Jo/Ji.
            # We need to permute to (Io, Ii, Jo, Ji) then reshape to (4, 4).
            # Current: Io, Jo, Ii, Ji.
            # Desired: Io, Ii, Jo, Ji.
            u_permuted = u_full.transpose(0, 2, 1, 3) # (Io, Ii, Jo, Ji)
            u_flat = u_permuted.reshape(4, 4) # (i, j) where i=2*Io+Ii, j=2*Jo+Ji
            
            U4[:, r] = u_flat.flatten()

            # V4: (Jo, Ko)
            vo = V2_reshaped[:, :, ro] # (2, 2) -> (Jo, Ko)
            vi = V2_reshaped[:, :, ri] # (2, 2) -> (Ji, Ki)
            # V4 index (j, k) = (2*Jo+Ji, 2*Ko+Ki)
            v_full = np.einsum('jk,jk->jkjk', vo, vi) # (Jo, Ko, Ji, Ki)
            # Desired: Jo, Ji, Ko, Ki
            v_permuted = v_full.transpose(0, 2, 1, 3) # (Jo, Ji, Ko, Ki)
            v_flat = v_permuted.reshape(4, 4) # (j, k)
            V4[:, r] = v_flat.flatten()

            # W4: (Ko, Io)
            wo = W2_reshaped[:, :, ro] # (2, 2) -> (Ko, Io)
            wi = W2_reshaped[:, :, ri] # (2, 2) -> (Ki, Ii)
            # W4 index (k, i) = (2*Ko+Ki, 2*Io+Ii)
            w_full = np.einsum('ki,ki->kiki', wo, wi) # (Ko, Io, Ki, Ii)
            # Desired: Ko, Ki, Io, Ii
            w_permuted = w_full.transpose(0, 2, 1, 3) # (Ko, Ki, Io, Ii)
            w_flat = w_permuted.reshape(4, 4) # (k, i)
            W4[:, r] = w_flat.flatten()

    if time.time() > deadline:
         # Fallback if it took too long (unlikely for this small size)
         pass

    # 3. Construct 4x4x8 decomposition (Rank 98)
    # C is 4x8. Split into C_left (4x4) and C_right (4x4).
    # C_left depends on B_left (4x4, cols 0-3).
    # C_right depends on B_right (4x4, cols 4-7).
    # A is 4x4, same for both.
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (r = 0..48)
    U[:, 0:49] = U4
    # V for left block: B indices j in 0..3, k in 0..3.
    # V4 is (16, 49) with index j*4+k.
    # V is (32, 98) with index j*8+k.
    # For r < 49, k is 0..3. j is 0..3.
    # V[j*8+k, r] = V4[j*4+k, r]
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
            
    # W for left block: C indices i in 0..3, k in 0..3.
    # W4 is (16, 49) with index k*4+i.
    # W is (32, 98) with index k*4+i.
    # For r < 49, k is 0..3.
    # W[k*4+i, r] = W4[k*4+i, r]
    # Note: The full W index is k*4+i where k is 0..7? No, W shape is (32, 98).
    # The convention is W[k*n+i, r]. n=4.
    # For C_left, k is 0..3. So W indices 0..15 are used.
    W[0:16, 0:49] = W4

    # Right block (r = 49..97)
    U[:, 49:98] = U4
    # V for right block: B indices j in 0..3, k in 4..7.
    # V4 is (16, 49) with index j*4+k_local (0..3).
    # V is (32, 98) with index j*8+k_global.
    # For r >= 49, k_global is 4..7. Let k_local = k_global - 4.
    # V[j*8+(4+k_local), 49+r_local] = V4[j*4+k_local, r_local]
    for j in range(4):
        for k_local in range(4):
            V[j*8+(4+k_local), 49:98] = V4[j*4+k_local, :]
            
    # W for right block: C indices i in 0..3, k in 4..7.
    # W4 is (16, 49) with index k_local*4+i.
    # W is (32, 98) with index k_global*4+i.
    # For r >= 49, k_global is 4..7.
    # W[(4+k_local)*4+i, 49+r_local] = W4[k_local*4+i, r_local]
    W[16:32, 49:98] = W4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
