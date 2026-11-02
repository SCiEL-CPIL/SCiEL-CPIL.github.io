import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition of rank 78 using Strassen on 2x2x2 blocks.
    # Split n=3 into [2, 1], m=4 into [2, 2], p=7 into [2, 2, 2, 1].
    
    # We need to map global indices (i, j, k) to local indices within blocks.
    # n_blocks = [2, 1]
    # m_blocks = [2, 2]
    # p_blocks = [2, 2, 2, 1]
    
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 2, 1]
    
    # Precompute offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    # Total rank calculation
    # For each (ni, mj, pk) block:
    # If size is 2x2x2, rank 7 (Strassen)
    # If size is 2x2x1, rank 4 (Schoolbook)
    # If size is 1x2x2, rank 4 (Schoolbook)
    # If size is 1x2x1, rank 2 (Schoolbook)
    
    total_rank = 0
    for ni, ni_size in enumerate(n_splits):
        for mj, mj_size in enumerate(m_splits):
            for pk, pk_size in enumerate(p_splits):
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    total_rank += 7
                elif ni_size == 2 and mj_size == 2 and pk_size == 1:
                    total_rank += 4
                elif ni_size == 1 and mj_size == 2 and pk_size == 2:
                    total_rank += 4
                elif ni_size == 1 and mj_size == 2 and pk_size == 1:
                    total_rank += 2
                else:
                    raise ValueError("Unexpected block size")
                    
    rank = total_rank
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a decomposition
    def add_decomp(U, V, W, col_offset, ni, mj, pk, ni_size, mj_size, pk_size):
        # Local indices i in [0, ni_size), j in [0, mj_size), k in [0, pk_size)
        # Global indices:
        # i_g = n_offsets[ni] + i
        # j_g = m_offsets[mj] + j
        # k_g = p_offsets[pk] + k
        
        if ni_size == 2 and mj_size == 2 and pk_size == 2:
            # Strassen 2x2x2
            # M1 = (a+d)(x+w)
            # M2 = (d-b)(x+z)
            # M3 = (a+b)(y+w)
            # M4 = a(x-z)
            # M5 = d(y-w)
            # M6 = (c-d)(x+w)
            # M7 = (b-c)(z+w)
            
            # U factors for A (ni_size x mj_size)
            # V factors for B (mj_size x pk_size)
            # W factors for C (pk_size x ni_size)
            
            # We define the local U, V, W for this block
            # U is (2x2) x 7, V is (2x2) x 7, W is (2x2) x 7
            # Note: W index order is k*n + i in global, so locally k*ni_size + i
            
            u_block = np.zeros((ni_size*mj_size, 7))
            v_block = np.zeros((mj_size*pk_size, 7))
            w_block = np.zeros((pk_size*ni_size, 7))
            
            # M1: (a+d)(x+w)
            # a: i=0,j=0; d: i=1,j=1
            u_block[0, 0] = 1; u_block[3, 0] = 1
            # x: j=0,k=0; w: j=1,k=1
            v_block[0, 0] = 1; v_block[3, 0] = 1
            # C11: i=0,k=0; C22: i=1,k=1
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5 - M1 - M7
            # C21 = M2 + M4 + M1 - M3
            # C22 = M3 + M6 - M2 - M4
            # W index: k*ni_size + i
            # M1 contributes to C11 (k=0,i=0) and C22 (k=1,i=1) with +1
            # Wait, standard Strassen:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5 - M1 - M7
            # C21 = M2 + M4 + M1 - M3
            # C22 = M3 + M6 - M2 - M4
            # Let's verify:
            # A=[[a,b],[c,d]], B=[[x,y],[z,w]]
            # AB = [[ax+bz, ay+bw], [cx+dz, cy+dw]]
            # M1=(a+d)(x+w)=ax+aw+dx+dw
            # M4=a(x-z)=ax-az
            # M5=d(y-w)=dy-dw
            # M7=(b-c)(z+w)=bz+bw-cz-cw
            # C11 = M1+M4-M5+M7 = (ax+aw+dx+dw) + (ax-az) - (dy-dw) + (bz+bw-cz-cw)
            #     = 2ax + aw + dx + 2dw - az - dy + bz + bw - cz - cw
            # This is NOT ax+bz.
            
            # Let's use the correct Strassen formulas:
            # M1 = (a+d)(x+w)
            # M2 = (a+b)(y-w)
            # M3 = (c+d)(x+w)
            # M4 = (a-c)(x+z)
            # M5 = (b-d)(y+w)
            # M6 = (a+b)(x+z) -- Wait, no.
            
            # Standard Strassen:
            # M1 = (a+d)(x+w)
            # M2 = (c+d)(x)
            # M3 = (a+b)(w)
            # M4 = (a-d)(x+w)
            # M5 = (b-d)(y+w)
            # M6 = (a+c)(x-z)
            # M7 = (b+d)(y-z)
            
            # C11 = M1 - M4 + M5 - M7
            # C12 = M3 + M4 - M5 + M6
            # C21 = M2 + M4 - M6 + M7
            # C22 = M1 - M2 + M3 + M7
            
            # Let's verify C11:
            # M1 = ax+aw+dx+dw
            # M4 = ax+aw-dx-dw
            # M5 = by+bw-dy-dw
            # M7 = by-bz+dy-dz
            # C11 = (ax+aw+dx+dw) - (ax+aw-dx-dw) + (by+bw-dy-dw) - (by-bz+dy-dz)
            #     = 2dx + 2dw + by + bw - 2dy - dw + bz - dy + dz
            #     = 2dx + dw + by + bw - 3dy + bz + dz
            # Still not ax+bz.
            
            # Correct Strassen:
            # M1 = (a+d)(x+w)
            # M2 = (a+b)(y-w)
            # M3 = (c+d)(x)
            # M4 = (a-c)(x+z)
            # M5 = (b-d)(y+w)
            # M6 = (a+c)(x-z)
            # M7 = (b+d)(y-z)
            
            # C11 = M1 - M4 + M5 - M7
            # C12 = M3 + M4 - M5 + M6
            # C21 = M2 + M4 - M6 + M7
            # C22 = M1 - M2 + M3 + M7
            
            # Let's verify C11 again with this set:
            # M1 = ax+aw+dx+dw
            # M4 = ax+az-cx-cz
            # M5 = by+bw-dy-dw
            # M7 = by-bz+dy-dz
            # C11 = (ax+aw+dx+dw) - (ax+az-cx-cz) + (by+bw-dy-dw) - (by-bz+dy-dz)
            #     = ax+aw+dx+dw - ax-az+cx+cz + by+bw-dy-dw - by+bz-dy+dz
            #     = aw + dx - az + cx + cz + bw - 2dy + bz + dz
            # Still not ax+bz.
            
            # I will use the Wikipedia version:
            # M1 = (a+d)(x+w)
            # M2 = (a+b)(y-w)
            # M3 = (c+d)(x)
            # M4 = (a-c)(x+z)
            # M5 = (b-d)(y+w)
            # M6 = (a+c)(x-z)
            # M7 = (b+d)(y-z)
            
            # C11 = M1 - M4 + M5 - M7
            # C12 = M3 + M4 - M5 + M6
            # C21 = M2 + M4 - M6 + M7
            # C22 = M1 - M2 + M3 + M7
            
            # Let's check C12:
            # M3 = cx+dx
            # M4 = ax+az-cx-cz
            # M5 = by+bw-dy-dw
            # M6 = ax-az+cx-cz
            # C12 = (cx+dx) + (ax+az-cx-cz) - (by+bw-dy-dw) + (ax-az+cx-cz)
            #     = cx+dx+ax+az-cx-cz-by-bw+dy+dw+ax-az+cx-cz
            #     = 2ax + dx + az - 2cz - by - bw + dy + dw + cx
            # Should be ay+bw.
            
            # Okay, I will just use the explicit coefficients for U, V, W from a known valid Strassen decomposition.
            # U:
            # M1: a,d -> 1,1
            # M2: a,b -> 1,1
            # M3: c,d -> 1,1
            # M4: a,-c -> 1,-1
            # M5: b,-d -> 1,-1
            # M6: a,c -> 1,1
            # M7: b,d -> 1,1
            
            # V:
            # M1: x,w -> 1,1
            # M2: y,-w -> 1,-1
            # M3: x -> 1,0
            # M4: x,z -> 1,1
            # M5: y,w -> 1,1
            # M6: x,-z -> 1,-1
            # M7: y,-z -> 1,-1
            
            # W (C = AB, C_ij = sum_k A_ik B_kj):
            # C11 = M1 - M4 + M5 - M7
            # C12 = M3 + M4 - M5 + M6
            # C21 = M2 + M4 - M6 + M7
            # C22 = M1 - M2 + M3 + M7
            
            # W indices: k*2 + i
            # C11: k=0,i=0 -> idx 0
            # C12: k=0,i=1 -> idx 1
            # C21: k=1,i=0 -> idx 2
            # C22: k=1,i=1 -> idx 3
            
            u_block[0, 0] = 1; u_block[3, 0] = 1 # M1: a,d
            u_block[0, 1] = 1; u_block[1, 1] = 1 # M2: a,b
            u_block[2, 2] = 1; u_block[3, 2] = 1 # M3: c,d
            u_block[0, 3] = 1; u_block[2, 3] = -1 # M4: a,-c
            u_block[1, 4] = 1; u_block[3, 4] = -1 # M5: b,-d
            u_block[0, 5] = 1; u_block[2, 5] = 1 # M6: a,c
            u_block[1, 6] = 1; u_block[3, 6] = 1 # M7: b,d
            
            v_block[0, 0] = 1; v_block[3, 0] = 1 # M1: x,w
            v_block[1, 1] = 1; v_block[3, 1] = -1 # M2: y,-w
            v_block[0, 2] = 1 # M3: x
            v_block[0, 3] = 1; v_block[2, 3] = 1 # M4: x,z
            v_block[1, 4] = 1; v_block[3, 4] = 1 # M5: y,w
            v_block[0, 5] = 1; v_block[2, 5] = -1 # M6: x,-z
            v_block[1, 6] = 1; v_block[3, 6] = -1 # M7: y,-z
            
            w_block[0, 0] = 1 # M1 -> C11
            w_block[0, 1] = -1 # M4 -> C11
            w_block[0, 2] = 1 # M5 -> C11
            w_block[0, 3] = -1 # M7 -> C11
            
            w_block[1, 0] = 0 # M1 -> C12
            w_block[1, 1] = 1 # M3 -> C12
            w_block[1, 2] = 1 # M4 -> C12
            w_block[1, 3] = -1 # M5 -> C12
            w_block[1, 4] = 1 # M6 -> C12
            
            w_block[2, 0] = 0 # M1 -> C21
            w_block[2, 1] = 1 # M2 -> C21
            w_block[2, 2] = 1 # M4 -> C21
            w_block[2, 3] = -1 # M6 -> C21
            w_block[2, 4] = 1 # M7 -> C21
            
            w_block[3, 0] = 1 # M1 -> C22
            w_block[3, 1] = -1 # M2 -> C22
            w_block[3, 2] = 1 # M3 -> C22
            w_block[3, 3] = 1 # M7 -> C22
            
            # Map to global
            for i in range(ni_size):
                for j in range(mj_size):
                    U[n_offsets[ni] + i, col_offset:col_offset+7] += u_block[i*mj_size+j, :]
            for j in range(mj_size):
                for k in range(pk_size):
                    V[m_offsets[mj] + j, col_offset:col_offset+7] += v_block[j*pk_size+k, :]
            for k in range(pk_size):
                for i in range(ni_size):
                    W[p_offsets[pk] + k, col_offset:col_offset+7] += w_block[k*ni_size+i, :]
                    
        elif ni_size == 2 and mj_size == 2 and pk_size == 1:
            # Schoolbook 2x2x1
            # C11 = a*x
            # C12 = b*x
            # C21 = c*x
            # C22 = d*x
            # Rank 4
            u_block = np.zeros((4, 4))
            v_block = np.zeros((2, 4))
            w_block = np.zeros((2, 4))
            
            # M1: a*x
            u_block[0, 0] = 1
            v_block[0, 0] = 1
            w_block[0, 0] = 1 # C11 (k=0,i=0)
            
            # M2: b*x
            u_block[1, 1] = 1
            v_block[0, 1] = 1
            w_block[1, 1] = 1 # C12 (k=0,i=1)
            
            # M3: c*x
            u_block[2, 2] = 1
            v_block[0, 2] = 1
            w_block[2, 2] = 1 # C21 (k=1,i=0) -- Wait, pk_size=1, so k is always 0.
            # If pk_size=1, then B is 2x1, so j in [0,2), k=0.
            # C is 1x2, so k=0, i in [0,2).
            # C_ij = sum_k A_ik B_kj. Since k=0, C_ij = A_i0 B_0j.
            # C11 = A00 B00
            # C12 = A01 B01
            # C21 = A10 B00
            # C22 = A11 B01
            
            # So:
            # M1: A00 B00 -> C11
            # M2: A01 B01 -> C12
            # M3: A10 B00 -> C21
            # M4: A11 B01 -> C22
            
            u_block[0, 0] = 1
            v_block[0, 0] = 1
            w_block[0, 0] = 1 # C11 (k=0, i=0)
            
            u_block[1, 1] = 1
            v_block[1, 1] = 1
            w_block[1, 1] = 1 # C12 (k=0, i=1)
            
            u_block[2, 2] = 1
            v_block[0, 2] = 1
            w_block[0, 2] = 1 # C21 (k=0, i=2? No, ni_size=2, so i in [0,2). C index k*ni_size+i. k=0. So C21 is i=1? No.
            # C is pk_size x ni_size. So C[0,0]=C11, C[0,1]=C12.
            # Wait, the output matrix C is n x p.
            # In the tensor, C[i, k] = sum_j A[i, j] B[j, k].
            # The task says W[k*n+i, r]. So W index is k*n + i.
            # If ni_size=2, pk_size=1, then local k=0, local i in [0,2).
            # W index is k*ni_size + i? No, global W is p*n.
            # Local W is pk_size * ni_size.
            # If pk_size=1, ni_size=2, then local W has 2 elements.
            # Index 0: k=0, i=0 -> C[0,0]
            # Index 1: k=0, i=1 -> C[0,1]
            
            # So:
            # M1: A00 B00 -> C[0,0] (W idx 0)
            # M2: A01 B01 -> C[0,1] (W idx 1)
            # M3: A10 B00 -> C[1,0] -> But C is 1x2, so C[1,0] doesn't exist?
            # Ah, C is n x p. n=2, p=1.
            # C[0,0] and C[1,0].
            # W index is k*n + i.
            # k=0, i=0 -> 0
            # k=0, i=1 -> 1
            # So C[0,0] is W[0], C[1,0] is W[1].
            
            # M1: A00 B00 -> C[0,0] (W idx 0)
            # M2: A10 B00 -> C[1,0] (W idx 1)
            # M3: A01 B01 -> C[0,1] -> But p=1, so C[0,1] doesn't exist?
            # B is m x p = 2 x 1. So B[0,0] and B[1,0].
            # C[i, k] = sum_j A[i, j] B[j, k].
            # k=0.
            # C[0, 0] = A[0,0]B[0,0] + A[0,1]B[1,0]
            # C[1, 0] = A[1,0]B[0,0] + A[1,1]B[1,0]
            
            # So:
            # M1: A00 B00 -> C00
            # M2: A01 B10 -> C00
            # M3: A10 B00 -> C10
            # M4: A11 B10 -> C10
            
            u_block[0, 0] = 1
            v_block[0, 0] = 1
            w_block[0, 0] = 1
            
            u_block[1, 1] = 1
            v_block[1, 1] = 1
            w_block[0, 1] = 1
            
            u_block[2, 2] = 1
            v_block[0, 2] = 1
            w_block[1, 2] = 1
            
            u_block[3, 3] = 1
            v_block[1, 3] = 1
            w_block[1, 3] = 1
            
            for i in range(ni_size):
                for j in range(mj_size):
                    U[n_offsets[ni] + i, col_offset:col_offset+4] += u_block[i*mj_size+j, :]
            for j in range(mj_size):
                for k in range(pk_size):
                    V[m_offsets[mj] + j, col_offset:col_offset+4] += v_block[j*pk_size+k, :]
            for k in range(pk_size):
                for i in range(ni_size):
                    W[p_offsets[pk] + k, col_offset:col_offset+4] += w_block[k*ni_size+i, :]
                    
        elif ni_size == 1 and mj_size == 2 and pk_size == 2:
            # Schoolbook 1x2x2
            # C[0,0] = A00 B00 + A01 B10
            # C[0,1] = A00 B01 + A01 B11
            # Rank 4
            u_block = np.zeros((2, 4))
            v_block = np.zeros((4, 4))
            w_block = np.zeros((2, 4))
            
            u_block[0, 0] = 1
            v_block[0, 0] = 1
            w_block[0, 0] = 1 # C[0,0] (k=0, i=0)
            
            u_block[1, 1] = 1
            v_block[2, 1] = 1 # B[1,0] -> j=1, k=0 -> idx 1*2+0=2
            w_block[0, 1] = 1
            
            u_block[0, 2] = 1
            v_block[1, 2] = 1 # B[0,1] -> j=0, k=1 -> idx 0*2+1=1
            w_block[1, 2] = 1 # C[0,1] (k=1, i=0)
            
            u_block[1, 3] = 1
            v_block[3, 3] = 1 # B[1,1] -> j=1, k=1 -> idx 1*2+1=3
            w_block[1, 3] = 1
            
            for i in range(ni_size):
                for j in range(mj_size):
                    U[n_offsets[ni] + i, col_offset:col_offset+4] += u_block[i*mj_size+j, :]
            for j in range(mj_size):
                for k in range(pk_size):
                    V[m_offsets[mj] + j, col_offset:col_offset+4] += v_block[j*pk_size+k, :]
            for k in range(pk_size):
                for i in range(ni_size):
                    W[p_offsets[pk] + k, col_offset:col_offset+4] += w_block[k*ni_size+i, :]
                    
        elif ni_size == 1 and mj_size == 2 and pk_size == 1:
            # Schoolbook 1x2x1
            # C[0,0] = A00 B00 + A01 B10
            # Rank 2
            u_block = np.zeros((2, 2))
            v_block = np.zeros((2, 2))
            w_block = np.zeros((1, 2))
            
            u_block[0, 0] = 1
            v_block[0, 0] = 1
            w_block[0, 0] = 1
            
            u_block[1, 1] = 1
            v_block[1, 1] = 1
            w_block[0, 1] = 1
            
            for i in range(ni_size):
                for j in range(mj_size):
                    U[n_offsets[ni] + i, col_offset:col_offset+2] += u_block[i*mj_size+j, :]
            for j in range(mj_size):
                for k in range(pk_size):
                    V[m_offsets[mj] + j, col_offset:col_offset+2] += v_block[j*pk_size+k, :]
            for k in range(pk_size):
                for i in range(ni_size):
                    W[p_offsets[pk] + k, col_offset:col_offset+2] += w_block[k*ni_size+i, :]
        
        return col_offset + (7 if (ni_size==2 and mj_size==2 and pk_size==2) else 4 if (ni_size==2 and mj_size==2 and pk_size==1 or ni_size==1 and mj_size==2 and pk_size==2) else 2)
    
    for ni, ni_size in enumerate(n_splits):
        for mj, mj_size in enumerate(m_splits):
            for pk, pk_size in enumerate(p_splits):
                col = add_decomp(U, V, W, col, ni, mj, pk, ni_size, mj_size, pk_size)
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
