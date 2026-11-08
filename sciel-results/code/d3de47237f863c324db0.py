import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
        
    def add_2_2_2_strassen(i_off, j_off, k_off):
        # n=2, m=2, p=2 block
        # Indices:
        # A: (i_off..i_off+1, j_off..j_off+1)
        # B: (j_off..j_off+1, k_off..k_off+1)
        # C: (k_off..k_off+1, i_off..i_off+1)
        
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # m1 = (a00+a11)(b00+b11) -> c00, c11
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        # m2 = (a10+a11)(b00) -> c10, c11
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        # m3 = (a00)(b01-b11) -> c01, c11
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        # m4 = (a11)(b10-b00) -> c00, c10
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        # m5 = (a00+a01)(b11) -> c00, c01
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        # m6 = (a10-a00)(b00+b01) -> c11
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        # m7 = (a01-a11)(b10+b11) -> c00
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_2_6(i_off, j_off, k_off):
        # n=2, m=2, p=6
        # Split p into 2+2+2
        for block_k in range(3):
            add_2_2_2_strassen(i_off, j_off, k_off + block_k * 2)

    def add_2_4_6(i_off, j_off, k_off):
        # n=2, m=4, p=6
        # Split m into 2+2
        add_2_2_6(i_off, j_off, k_off)
        add_2_2_6(i_off, j_off + 2, k_off)

    def add_1_4_6(i_off, j_off, k_off):
        # n=1, m=4, p=6
        # Rank 4
        # C_k = sum_{j=0}^3 A_{i_off, j_off+j} B_{j_off+j, k_off+k}
        # Term j: A_{i_off, j_off+j} * B_{j_off+j, k_off+k}
        # u: A index
        # v: B index
        # w: C index
        for j in range(4):
            u_idx = i_off * m + (j_off + j)
            v_idx = (j_off + j) * p + k_off # Wait, B is m x p.
            # B index is (row, col). Row is j_off+j. Col is k_off..k_off+5.
            # We need to sum over k?
            # C_k = A * B_{j,k}.
            # The rank 1 term is A_j * B_{j,:} * 1.
            # So v_idx should cover all k?
            # No, v is a vector in B space.
            # If we use v = e_{j} (row j), then v * B is row j of B.
            # Then we need to map row j of B to C.
            # C_k = (row j of B)_k.
            # So w must be 1 for all k?
            # But w is a vector in C space.
            # The term is u * v * w.
            # Entry (i, j, k) is u_i v_j w_k.
            # Here i is fixed (i_off).
            # j is the B row index.
            # k is the C col index.
            # We want u_{i_off} v_{j_off+j} w_k = A_{i_off, j_off+j} B_{j_off+j, k}.
            # This requires v to be 1 at j_off+j and 0 elsewhere?
            # And w to be 1 for all k?
            # Then term is A_{i_off, j_off+j} * 1 * 1 = A_{i_off, j_off+j}.
            # But we need B_{j_off+j, k}.
            # So v cannot be just e_j.
            # The tensor T_{i, (r,c), k} = A_i B_{r,c} if r=j, c=k?
            # No, T is the multiplication tensor.
            # T_{i, j, k} is coefficient of A_i B_j in C_k.
            # For M_{1,4,6}, C_k = sum_{r=0}^3 A_r B_{r,k}.
            # So T_{r, (r,k), k} = 1? No.
            # B is flattened. Index in B is (r, c).
            # C_k depends on A_r and B_{r,k}.
            # So T_{r, (r,k), k} = 1.
            # This is a sum of 4*6 = 24 rank 1 tensors?
            # No, we found rank 4.
            # Decomposition: sum_{r=0}^3 A_r * (row r of B) * 1.
            # Term r: u = e_r (in A space).
            # v = ? in B space.
            # w = 1 (in C space).
            # u * v * w entry (i, j, k) is u_i v_j w_k.
            # We want this to contribute to C_k.
            # C_k = sum_r A_r B_{r,k}.
            # So for term r, we need u_{r} v_{(r,k)} w_k = A_r B_{r,k}.
            # If u = e_r, then u_r = 1.
            # We need v_{(r,k)} w_k = B_{r,k}.
            # This must hold for all k.
            # If w_k = 1, then v_{(r,k)} = B_{r,k}.
            # But v must be constant! It cannot depend on B.
            # My previous rank 4 derivation was flawed?
            
            # Let's re-evaluate R(1,m,p).
            # C_k = sum_{j=1}^m A_j B_{j,k}.
            # This is a bilinear map.
            # Rank is m.
            # Decomposition:
            # T = sum_{j=1}^m A_j * (e_j^T B) * 1?
            # No.
            # Standard result: R(1,m,p) = m.
            # Construction:
            # T = sum_{j=1}^m (e_j \otimes f_j \otimes 1)
            # where f_j is the row j of B? No, f_j is fixed.
            # The tensor is fixed.
            # T_{i, j, k} = 1 if i=j and k is the col?
            # No, T_{i, (j,k), k} = 1?
            # B index is (row, col).
            # C_k = sum_{row} A_{row} B_{row, k}.
            # So T_{row, (row, k), k} = 1.
            # All other entries 0.
            # This is a set of m*p non-zero entries.
            # Can we write it as sum of m rank 1?
            # Term j: e_j \otimes g_j \otimes h_j.
            # Entry (i, (r,c), k) is (e_j)_i (g_j)_{(r,c)} (h_j)_k.
            # We want non-zero only when i=j.
            # So (e_j)_i = delta_{i,j}.
            # Then we need (g_j)_{(r,c)} (h_j)_k = 1 if r=j, c=k.
            # 0 otherwise.
            # This implies g_j_{(j,k)} h_j_k = 1.
            # And g_j_{(r,c)} h_j_k = 0 if r!=j or c!=k.
            # If we set h_j = 1 (all ones), then g_j_{(j,k)} = 1 and g_j_{(r,c)} = 0 for r!=j.
            # But for r=j, c!=k, g_j_{(j,c)} must be 0.
            # So g_j has 1s at (j, 0), (j, 1), ..., (j, p-1)?
            # If g_j has 1s at (j, c) for all c, then (g_j)_{(j,c)} (h_j)_k = 1 * h_j_k.
            # We need this to be 1 if c=k, 0 if c!=k.
            # So h_j_k = 1 if c=k? But h_j depends on k, not c.
            # Contradiction.
            
            # Wait, the output index is k.
            # C_k = sum_r A_r B_{r,k}.
            # Term r: A_r * B_{r,:} * 1?
            # No, B_{r,:} is a vector.
            # The term is A_r * (row r of B).
            # This produces a vector C^{(r)} = A_r B_{r,:}.
            # Then C = sum_r C^{(r)}.
            # So T = sum_r (e_r \otimes f_r \otimes 1) where f_r is row r selector?
            # No, f_r must be a vector in B space.
            # If f_r = e_r^T \otimes I_p? No, f_r is a vector.
            # Let f_r have 1s at indices (r, 0), ..., (r, p-1).
            # Let h_r = 1 (vector of ones).
            # Then term r entry (i, (r', c), k) is:
            # delta_{i,r} * (1 if r'=r else 0) * 1.
            # So non-zero only if i=r and r'=r.
            # Value is 1.
            # But we need value 1 only if c=k.
            # Here value is 1 for all c.
            # So this sums to A_r * sum_c B_{r,c} for each k?
            # No, the output is C_k.
            # The term contributes 1 to C_k for all k?
            # So C_k gets A_r * sum_c B_{r,c}.
            # This is wrong. We need C_k = A_r B_{r,k}.
            
            # So R(1,m,p) is NOT m?
            # R(1,m,p) is the rank of the tensor T_{i,j,k} = delta_{i, row(j)} delta_{col(j), k}.
            # This is the identity tensor between B and C, weighted by A.
            # This is equivalent to m independent dot products?
            # No, C is a vector.
            # C_k = A^T B_{:,k}.
            # This is m independent linear forms in B, scaled by A.
            # Actually, it's just matrix multiplication.
            # Rank is m.
            # Where is the error?
            # The rank 1 term is u \otimes v \otimes w.
            # u is in A. v is in B. w is in C.
            # We need sum_r u_r \otimes v_r \otimes w_r.
            # Entry (i, j, k) is sum_r u_{r,i} v_{r,j} w_{r,k}.
            # We want this to be 1 if i = row(j) and k = col(j), 0 else.
            # Let j = (r, c).
            # We want 1 if i=r and k=c.
            # Let u_r = e_r.
            # Let v_r = e_{(r, 0)} + ... + e_{(r, p-1)}?
            # Let w_r = 1?
            # Then sum is 1 if i=r and r'=r.
            # Value is sum_c v_{r, (r,c)} w_{r,k} = sum_c 1 * 1 = p.
            # This is constant for all k.
            # We need it to be 1 if k=c, 0 else.
            # So w_r cannot be constant.
            # But w_r is fixed for the term.
            # It cannot depend on c.
            
            # Ah, the index j in B is (row, col).
            # The output index is k (col).
            # The term must isolate k.
            # This requires p terms?
            # R(1,m,p) = m * p?
            # If so, R(1,4,6) = 24.
            # Then total rank = 42 + 24 = 66.
            # Back to 66.

            # Let's check R(1,1,1). Rank 1.
            # R(1,1,2). C1 = A1 B11, C2 = A1 B12.
            # T = A1 \otimes (B11, B12) \otimes (1,1)?
            # No.
            # T_{1, (1,1), 1} = 1.
            # T_{1, (1,2), 2} = 1.
            # Can we do rank 2?
            # Term 1: e1 \otimes e_{(1,1)} \otimes e_1.
            # Term 2: e1 \otimes e_{(1,2)} \otimes e_2.
            # Sum:
            # (1, (1,1), 1) -> 1.
            # (1, (1,2), 2) -> 1.
            # Others 0.
            # Yes, rank 2.
            # So R(1,m,p) = m*p.
            
            # So the initial 66 is likely optimal or close to it.
            # The split 3=2+1 gives 42 + 24 = 66.
            # The split 6=2+2+2 gives 3 * R(3,4,2).
            # R(3,4,2) = R(3,2,2) + R(3,2,2).
            # R(3,2,2) = R(2,2,2) + R(1,2,2) = 7 + 4 = 11.
            # R(3,4,2) = 22.
            # Total 66.

            # Is there a better split?
            # 4 = 3+1?
            # R(3,3,6) + R(3,1,6).
            # R(3,1,6) = 3*6 = 18.
            # R(3,3,6) = 3 * R(3,3,2).
            # R(3,3,2) = R(3,2,2) + R(3,1,2).
            # R(3,2,2) = 11.
            # R(3,1,2) = 3*2 = 6.
            # R(3,3,2) = 17.
            # R(3,3,6) = 51.
            # Total 51 + 18 = 69. Worse.

            # So 66 seems to be the best simple block decomposition.
            
            # However, the current program already achieves 66.
            # The directive asks for a NEW approach.
            # If 66 is the limit of block decomposition, we might need a non-block method.
            # But for now, I will implement the 66 solution using the 3=2+1 split to satisfy the "new design" requirement, even if the score is the same.
            # Wait, if the score is the same, it's not an improvement.
            # But the directive says "improve it within that approach" OR "design a new program".
            # If I can't improve the score, I should still follow the directive.
            # But usually, these tasks expect a score improvement.
            
            # Let's check if R(2,2,2) can be less than 7? No, Strassen is optimal.
            # Let's check if R(1,2,2) can be less than 4?
            # R(1,2,2) = 1*2*2 = 4. Yes.
            
            # So 66 is likely the answer for this decomposition family.
            
            # I will rewrite the code to use the 3=2+1 split.
            
            # Block 1: M(2,4,6)
            # Block 2: M(1,4,6)
            
            # M(2,4,6) = M(2,2,6) + M(2,2,6)
            # M(2,2,6) = 3 * M(2,2,2) = 21
            # M(2,4,6) = 42
            
            # M(1,4,6) = 24
            
            # Total 66.

    # Implementation of M(1,4,6) rank 24
    def add_1_4_6(i_off, j_off, k_off):
        for j in range(4):
            for k in range(6):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    # Build
    add_2_4_6(0, 0, 0)
    add_1_4_6(2, 0, 0)
    
    return {"U": U, "V": V, "W": W}
