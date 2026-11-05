import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Target tensor T is sparse: T[i, j, k] = 1 at (i*m+j, j*p+k, k*n+i)
    # We want to find U, V, W such that sum_r U[:,r] V[:,r] W[:,r] = T
    # This is equivalent to finding a set of "terms" (u_r, v_r, w_r)
    # Each term corresponds to a block multiplication.
    
    # Strategy:
    # 1. Define a library of known optimal decompositions for small rectangular matrix multiplications.
    #    For a block of size (n_i, m_j, p_k), we have a decomposition of rank R_ij_k.
    #    The decomposition provides coefficients for the sub-tensor.
    # 2. Generate all possible partitions of the dimensions n, m, p.
    #    A partition of n is a list of block sizes summing to n.
    #    Similarly for m and p.
    # 3. For each combination of partitions, the total rank is the sum of the ranks
    #    of the decompositions used for each block (n_i, m_j, p_k).
    #    We assume we can independently decompose each block.
    #    This is valid if we use block diagonal structures or if the blocks are disjoint in indices.
    #    Since the tensor is a direct sum of blocks if we partition the indices,
    #    the rank is the sum of the ranks of the block decompositions.
    
    # Let's build a dictionary of optimal ranks for small blocks.
    # We need to handle blocks where n_i, m_j, p_k are small (1, 2, 3, 4).
    # Known optimal ranks:
    # 1x1x1: 1
    # 2x2x2: 7 (Strassen)
    # 2x2x1: 3 (Schoolbook is 4, but 2x2x1 is rank 3? 
    #         C = A B. A is 2x2, B is 2x1.
    #         C1 = A11 B1 + A12 B2
    #         C2 = A21 B1 + A22 B2
    #         This is 4 terms. Can we do 3?
    #         Actually, 2x2x1 is isomorphic to 1x2x2.
    #         Rank of 2x2x1 is 3. Let's verify.
    #         Terms: (A11+A12, B1, C1+C2), (A11, B1, C1), (A12, B2, C1)... no.
    #         Let's just use schoolbook for non-2x2x2 blocks to be safe, 
    #         unless we can verify a better one.
    #         Actually, for 2x2x1, the tensor is T[i, j, k=0] = 1 if i,j in 0,1.
    #         It's a 2x2 matrix of 1s in the first slice.
    #         Rank of 2x2 matrix of 1s is 1. So rank of 2x2x1 is 1?
    #         Wait, the tensor is T[i, j, k].
    #         For 2x2x1, k is fixed. T is a 2x2 matrix.
    #         The matrix multiplication tensor for 2x2x1 is NOT the all-ones matrix.
    #         T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    #         If p=1, C is 2x1.
    #         C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    #         So T[i, j, 0] = 1 for all i, j.
    #         Yes, it is the all-ones matrix of size 2x2.
    #         The rank of the all-ones matrix (as a tensor slice) is 1.
    #         So rank(2x2x1) = 1?
    #         Let's check the decomposition.
    #         Term 1: u=[1,1], v=[1,1], w=[1].
    #         u_i v_j w_k = 1*1*1 = 1.
    #         Yes, rank 1.
    #         Similarly, rank(1x2x2) = 1, rank(2x1x2) = 1.
    #         Rank(2x2x2) = 7.
    #         Rank(3x3x3) = 23 (Winograd) or 21 (Bini et al)? 23 is safe.
    #         Rank(2x2x3)?
    #         Let's just implement a generic "cost function" for a block (ni, mj, pk).
    #         If ni==2 and mj==2 and pk==2: cost = 7
    #         If ni==1 or mj==1 or pk==1: cost = ni * mj * pk (Schoolbook is often optimal for thin blocks? 
    #                 Actually for 1xMxP, it's rank 1 if M,P arbitrary? 
    #                 T[i=0, j, k] = 1. 
    #                 u=[1], v=[1..1], w=[1..1]. 
    #                 u_0 v_j w_k = 1. 
    #                 So rank(1xMxP) = 1.
    #                 Similarly rank(Mx1xP) = 1, rank(MxPx1) = 1.
    #                 What about 2x2x1? Rank 1.
    #                 2x3x1? Rank 1.
    #                 So if any dimension is 1, rank is 1?
    #                 Let's check 2x2x1 again. 
    #                 C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    #                 C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    #                 Terms:
    #                 1: u=[1,1], v=[1,1], w=[1] -> C[0,0]+=1, C[1,0]+=1. 
    #                    But we need A[0,0]B[0,0] to go to C[0,0] ONLY?
    #                    No, the tensor T defines the linear map.
    #                    T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    #                    The decomposition sum u_i v_j w_k must equal T[i,j,k].
    #                    If we use u=[1,1], v=[1,1], w=[1], then term is 1 for all i,j.
    #                    So T is reproduced.
    #                    So yes, rank is 1.
    #                 What about 2x2x2?
    #                 T[i,j,k]=1.
    #                 Rank 1 would imply T is u v w. 
    #                 T[0,0,0]=1, T[0,0,1]=1 => u0 v0 w0 = 1, u0 v0 w1 = 1 => w0=w1.
    #                 T[0,1,0]=1 => u0 v1 w0 = 1 => v1=v0.
    #                 T[1,0,0]=1 => u1 v0 w0 = 1 => u1=u0.
    #                 So u=v=w=[1,1].
    #                 Then T[0,0,0] = 1*1*1 = 1. OK.
    #                 But T has 1s everywhere. 
    #                 Wait, the matrix multiplication tensor is NOT the all-ones tensor.
    #                 T[i, j, k] is 1 if the index j connects i and k.
    #                 In standard MM, A is n x m, B is m x p, C is n x p.
    #                 C[i, k] = sum_j A[i, j] B[j, k].
    #                 The tensor T is in V^* (dim nm) x W^* (dim mp) x X^* (dim np).
    #                 T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i.
    #                 T[a, b, c] = 1 if the indices are consistent.
    #                 Specifically, for a block (ni, mj, pk), the sub-tensor is non-zero
    #                 only at indices (i, j, k) within the block.
    #                 And for those indices, T = 1.
    #                 So the sub-tensor IS the all-ones tensor of size ni x mj x pk?
    #                 No.
    #                 Let's check 2x2x2.
    #                 i in {0,1}, j in {0,1}, k in {0,1}.
    #                 a = i*2+j, b = j*2+k, c = k*2+i.
    #                 T[a,b,c] = 1.
    #                 There are 8 non-zero entries.
    #                 The "all-ones tensor" of size 2x2x2 would have 8 non-zero entries.
    #                 Are they at the same positions?
    #                 All-ones tensor U[i,j,k] = 1 for all i,j,k.
    #                 Our tensor T is 1 at (i*m+j, j*p+k, k*n+i).
    #                 If we permute the indices of the all-ones tensor, do we get T?
    #                 Yes, T is a permutation of the all-ones tensor?
    #                 No. The all-ones tensor has T[i,j,k]=1 for ALL i,j,k.
    #                 Our tensor T[a,b,c] is 1 ONLY if b corresponds to the same j as a,
    #                 and c corresponds to the same k as b and same i as a.
    #                 So T is SPARSE.
    #                 The all-ones tensor is DENSE.
    #                 My previous deduction that rank(2x2x1)=1 was based on T being all-ones.
    #                 But T is NOT all-ones.
    #                 Let's re-evaluate rank(2x2x1).
    #                 n=2, m=2, p=1.
    #                 a in {0,1,2,3} (i*2+j)
    #                 b in {0,1,2,3} (j*1+k) -> j*1+0 = j. So b in {0,1,2,3}?
    #                 Wait, m=2, p=1. b = j*p+k = j*1+0 = j.
    #                 So b is in {0, 1}? No, j is in {0,1}. So b is in {0,1}.
    #                 But the dimension of V is m*p = 2.
    #                 So b is in {0,1}.
    #                 c = k*n+i = 0*2+i = i.
    #                 So c is in {0,1}.
    #                 T[a, b, c] = 1 if a=i*2+j, b=j, c=i.
    #                 Non-zero entries:
    #                 i=0, j=0: a=0, b=0, c=0 -> T[0,0,0]=1
    #                 i=0, j=1: a=1, b=1, c=0 -> T[1,1,0]=1
    #                 i=1, j=0: a=2, b=0, c=1 -> T[2,0,1]=1
    #                 i=1, j=1: a=3, b=1, c=1 -> T[3,1,1]=1
    #                 The tensor is sparse.
    #                 Can we decompose this with rank 1?
    #                 u v w = T.
    #                 T[0,0,0]=1 => u0 v0 w0 = 1
    #                 T[1,1,0]=1 => u1 v1 w0 = 1
    #                 T[2,0,1]=1 => u2 v0 w1 = 1
    #                 T[3,1,1]=1 => u3 v1 w1 = 1
    #                 All other entries are 0.
    #                 T[0,1,0]=0 => u0 v1 w0 = 0. Since u0 v0 w0=1, v1 must be 0?
    #                 If v1=0, then T[1,1,0] = u1 v1 w0 = 0. Contradiction.
    #                 So rank > 1.
    #                 Rank 2?
    #                 This is getting complicated.
    #                 Let's stick to known optimal ranks for standard MM tensors.
    #                 The rank of the MM tensor M_{n,m,p} is the minimal number of multiplications.
    #                 We can look up or derive small cases.
    #                 M_{1,1,1}: 1
    #                 M_{2,2,2}: 7
    #                 M_{2,2,1}: 3 (Laderman? No, Laderman is 4x4x4=21).
    #                 For 2x2x1, it is known that rank is 3.
    #                 For 2x2x3, rank is 11?
    #                 
    #                 To be safe and fast, I will implement a search over partitions
    #                 and use a cost function that is:
    #                 - 7 for 2x2x2
    #                 - n*m*p for other blocks (Schoolbook)
    #                 This might not be optimal for all small blocks, but it captures the
    #                 major savings from Strassen blocks.
    #                 Given the small size (3,4,8), the number of partitions is small.
    #                 Partitions of 3: [3], [2,1], [1,2], [1,1,1]
    #                 Partitions of 4: [4], [3,1], [1,3], [2,2], [2,1,1], [1,2,1], [1,1,2], [1,1,1,1]
    #                 Partitions of 8: [8], [7,1], ..., many.
    #                 Total combinations: ~4 * 8 * ~22 = ~700.
    #                 This is very fast.
    
    def get_partitions(n):
        parts = []
        def gen(rem, curr):
            if rem == 0:
                parts.append(tuple(curr))
                return
            for x in range(1, rem + 1):
                gen(rem - x, curr + [x])
        gen(n, [])
        return parts

    parts_n = get_partitions(n)
    parts_m = get_partitions(m)
    parts_p = get_partitions(p)
    
    def block_rank(ni, mj, pk):
        # Return the minimal rank for a block of size ni x mj x pk
        # Using known results or schoolbook
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        # Add more known optimal ranks here if known
        # For now, schoolbook
        return ni * mj * pk

    best_rank = n * m * p
    best_config = None
    
    for pn in parts_n:
        for pm in parts_m:
            for pp in parts_p:
                # The blocks must be aligned. 
                # This partition approach assumes we can tile the matrix multiplication
                # into independent blocks.
                # This is only possible if the partitions are "compatible".
                # Actually, if we partition the indices i, j, k, the tensor decomposes
                # into a direct sum of sub-tensors ONLY IF the partitions are such that
                # each sub-tensor is closed.
                # For MM, if we partition i into I1, I2 and j into J1, J2,
                # the sub-tensor for (I1, J1) is not closed because k is shared.
                # The term A[I1, J1] B[J1, K] contributes to C[I1, K].
                # So we cannot simply sum the ranks of independent (ni, mj, pk) blocks
                # unless we partition ALL three dimensions consistently such that
                # each block (I_a, J_b, K_c) is a valid MM problem.
                # This requires that for each a, b, c, the block A[I_a, J_b] and B[J_b, K_c]
                # multiply to C[I_a, K_c].
                # This is always true for any partition of i, j, k.
                # The tensor is the direct sum of the tensors for each (I_a, J_b, K_c) block?
                # No. The indices a, b, c are global.
                # T is a 3-way array.
                # If we partition the indices, we are effectively blocking the matrices.
                # The rank of the blocked MM is the sum of the ranks of the block MM's
                # IF we use a block-diagonal structure in the decomposition.
                # But the decomposition U, V, W is global.
                # However, it is a known result that the rank of MM for (n,m,p)
                # is at most the sum of ranks of the blocks if we use a specific tiling.
                # Specifically, if we partition n into n1, n2, m into m1, m2, p into p1, p2,
                # we have 8 blocks: (n1,m1,p1), (n1,m1,p2), etc.
                # The rank is the sum of the ranks of these 8 block MM's.
                # This is because we can construct U, V, W by concatenating the block decompositions.
                
                total_rank = 0
                valid = True
                # We need to ensure that the partitions of n, m, p define the blocks.
                # The number of blocks is len(pn) * len(pm) * len(pp).
                # Each block has size (pn[a], pm[b], pp[c]).
                for a in range(len(pn)):
                    for b in range(len(pm)):
                        for c in range(len(pp)):
                            ni = pn[a]
                            mj = pm[b]
                            pk = pp[c]
                            r = block_rank(ni, mj, pk)
                            total_rank += r
                            
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_config = (pn, pm, pp)
                    
    # Now, we need to construct the actual U, V, W for the best configuration.
    # We will use the schoolbook decomposition for all blocks except 2x2x2,
    # where we will use Strassen.
    
    if best_config is None:
        # Fallback to schoolbook
        pn, pm, pp = ([n], [m], [p])
    else:
        pn, pm, pp = best_config
        
    # Construct U, V, W
    # We need to map the global indices to the block indices.
    # Let's define the offset for each block.
    # U has rows indexed by i*m + j.
    # V has rows indexed by j*p + k.
    # W has rows indexed by k*n + i.
    
    # We will accumulate the contributions.
    # Since we are summing independent block decompositions,
    # we can just append the columns of U, V, W for each block.
    
    U_list = []
    V_list = []
    W_list = []
    
    # We need to know the rank of each block to allocate columns.
    # And we need to place the entries in the correct rows.
    
    # Let's define a function to get the decomposition for a block (ni, mj, pk)
    # It returns local U (ni*mj x r), V (mj*pk x r), W (pk*ni x r)
    # Note: The local indices are relative to the block.
    # Global i = i_offset + i_local
    # Global j = j_offset + j_local
    # Global k = k_offset + k_local
    
    def get_block_decomp(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            # Strassen 2x2x2
            # Terms:
            # 1: (A11+A22, B11-B22, C11+C22) -> Wait, standard Strassen
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B22)
            # M4 = A22(B12 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to express this as sum of rank-1 tensors u v w.
            # Each M corresponds to a product of two linear combinations.
            # This is NOT a rank-1 decomposition of the MM tensor.
            # The MM tensor decomposition is a sum of terms u_r v_r w_r.
            # Strassen uses 7 multiplications, which correspond to 7 terms in the decomposition.
            # However, the coefficients in U, V, W are not just 1, 0, -1.
            # They are the coefficients of the linear combinations.
            
            # Let's construct the 7 terms explicitly.
            # Term 1: u = [1,0,0,1] (A11, A22), v = [1,-1,0,0] (B11, B22)?
            # No, v is for B. B is 2x2. v is size 4.
            # v = [1, 0, 0, -1] corresponds to B11 - B22?
            # B11 is at (0,0), B22 is at (1,1).
            # So v = [1, 0, 0, -1].
            # u = [1, 0, 0, 1] (A11, A22).
            # w = [1, 1, 0, 0] (C11, C12)? No, C11+C22.
            # C11 is (0,0), C22 is (1,1).
            # w = [1, 0, 0, 1].
            
            # This is getting error-prone to do manually.
            # Let's use the standard Strassen decomposition coefficients.
            
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # Indices:
            # U: 0=A11, 1=A12, 2=A21, 3=A22
            # V: 0=B11, 1=B12, 2=B21, 3=B22
            # W: 0=C11, 1=C12, 2=C21, 3=C22
            
            # M1 = (A11+A22)(B11+B22)
            U[:,0] = [1, 0, 0, 1]
            V[:,0] = [1, 0, 0, 1]
            W[:,0] = [1, 1, 0, 0] # C11+C22? No, C11 and C22.
            # Wait, C11 = M1 + M4 - M5 + M7
            # C22 = M1 - M2 + M3 + M6
            # So M1 contributes to both C11 and C22.
            # W for M1 should be [1, 0, 0, 1] (C11 and C22).
            W[:,0] = [1, 0, 0, 1]
            
            # M2 = (A21+A22)B11
            U[:,1] = [0, 0, 1, 1]
            V[:,1] = [1, 0, 0, 0]
            W[:,1] = [0, 0, 1, 0] # C21
            
            # M3 = A11(B21-B22)
            U[:,2] = [1, 0, 0, 0]
            V[:,2] = [0, 0, 1, -1]
            W[:,2] = [0, 1, 0, 0] # C12
            
            # M4 = A22(B12-B11)
            U[:,3] = [0, 0, 0, 1]
            V[:,3] = [-1, 1, 0, 0]
            W[:,3] = [1, 0, 1, 0] # C11, C21
            
            # M5 = (A11+A12)B22
            U[:,4] = [1, 1, 0, 0]
            V[:,4] = [0, 0, 0, 1]
            W[:,4] = [1, 0, 0, 0] # C11
            
            # M6 = (A21-A11)(B11+B12)
            U[:,5] = [-1, 0, 1, 0]
            V[:,5] = [1, 1, 0, 0]
            W[:,5] = [0, 0, 0, 1] # C22
            
            # M7 = (A12-A22)(B21+B22)
            U[:,6] = [0, 1, 0, -1]
            V[:,6] = [0, 0, 1, 1]
            W[:,6] = [1, 0, 0, 0] # C11
            
            return U, V, W
        else:
            # Schoolbook
            r = ni * mj * pk
            U = np.zeros((ni * mj, r))
            V = np.zeros((mj * pk, r))
            W = np.zeros((pk * ni, r))
            col = 0
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        U[i * mj + j, col] = 1
                        V[j * pk + k, col] = 1
                        W[k * ni + i, col] = 1
                        col += 1
            return U, V, W

    # Now, build the global U, V, W
    # We need to iterate over all blocks and append their decompositions.
    # But we must place the entries in the correct GLOBAL rows.
    
    # Total rank is best_rank.
    # We will create U, V, W of size (n*m, best_rank), etc.
    
    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    
    col_offset = 0
    
    # Calculate offsets for i, j, k
    i_offsets = [0]
    for x in pn[:-1]:
        i_offsets.append(i_offsets[-1] + x)
        
    j_offsets = [0]
    for x in pm[:-1]:
        j_offsets.append(j_offsets[-1] + x)
        
    k_offsets = [0]
    for x in pp[:-1]:
        k_offsets.append(k_offsets[-1] + x)
        
    for a in range(len(pn)):
        for b in range(len(pm)):
            for c in range(len(pp)):
                ni = pn[a]
                mj = pm[b]
                pk = pp[c]
                
                i_off = i_offsets[a]
                j_off = j_offsets[b]
                k_off = k_offsets[c]
                
                U_blk, V_blk, W_blk = get_block_decomp(ni, mj, pk)
                r_blk = U_blk.shape[1]
                
                # Copy U_blk to U
                # Row mapping: local i, j -> global i, j
                # local i in [0, ni), local j in [0, mj)
                # global i = i_off + local_i
                # global j = j_off + local_j
                # global row index = global_i * m + global_j
                
                for r in range(r_blk):
                    for li in range(ni):
                        for lj in range(mj):
                            local_row = li * mj + lj
                            global_i = i_off + li
                            global_j = j_off + lj
                            global_row = global_i * m + global_j
                            U[global_row, col_offset + r] = U_blk[local_row, r]
                            
                # Copy V_blk to V
                # local j, k -> global j, k
                # global j = j_off + local_j
                # global k = k_off + local_k
                # global row index = global_j * p + global_k
                
                for r in range(r_blk):
                    for lj in range(mj):
                        for lk in range(pk):
                            local_row = lj * pk + lk
                            global_j = j_off + lj
                            global_k = k_off + lk
                            global_row = global_j * p + global_k
                            V[global_row, col_offset + r] = V_blk[local_row, r]
                            
                # Copy W_blk to W
                # local k, i -> global k, i
                # global k = k_off + local_k
                # global i = i_off + local_i
                # global row index = global_k * n + global_i
                
                for r in range(r_blk):
                    for lk in range(pk):
                        for li in range(ni):
                            local_row = lk * ni + li
                            global_k = k_off + lk
                            global_i = i_off + li
                            global_row = global_k * n + global_i
                            W[global_row, col_offset + r] = W_blk[local_row, r]
                            
                col_offset += r_blk
                
    # Verify
    S = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Create target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    # S is (n, m, p) ? No, einsum 'ir,jr,kr->ijk' gives (n, m, p) if U is (n*m, r)?
    # No, U is (n*m, r). i is index of U, which is i*m+j.
    # So S[i, j, k] where i is 0..n*m-1, j is 0..m*p-1, k is 0..p*n-1.
    # This matches T's shape.
    
    if not np.allclose(S, T, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
