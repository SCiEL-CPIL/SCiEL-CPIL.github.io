import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # The tensor for 3x4 x 4x4 multiplication:
    # A: 3x4 (12 entries), B: 4x4 (16 entries), C: 3x4 (12 entries)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # W convention: W[k*n+i, r] where k is col of B, i is row of A
    
    # Strategy: Try to find a low-rank decomposition directly
    # or use known results / clever constructions
    
    # For 2x2x2: rank 7 (Strassen)
    # For 3x3x3: rank 23 (Bini et al.)
    # For 3x4x4: we can think of it as combining blocks
    
    # Approach 1: Direct search with optimization
    # Approach 2: Block decomposition with shared factors
    
    # Let's try a direct approach: build the tensor and use optimization
    # But first, let's try known constructions
    
    # Method: Use the fact that 3x4x4 can be decomposed
    # Think of A as 3x4, B as 4x4
    # We can write B as 2x2 blocks of 2x2: B = [[B00, B01], [B10, B11]]
    # A = [A_left (3x2), A_right (3x2)]
    # C = [A_left * B_left_half, A_right * B_right_half]... not quite
    
    # Better: C[i,j] = sum_k A[i,k]*B[k,j]
    # Split k into k1 (0,1) and k2 (2,3):
    # C[i,j] = sum_{k1} A[i,k1]*B[k1,j] + sum_{k2} A[i,k2]*B[k2,j]
    # = A[i,0:2] * B[0:2,j] + A[i,2:4] * B[2:4,j]
    
    # So we need:
    # Part 1: 3x2 * 2x4 -> 3x4 (two 3x2x2 blocks)
    # Part 2: 3x2 * 2x4 -> 3x4 (two 3x2x2 blocks)
    
    # Each 3x2x2 block: rank at least 7? No, 3x2x2 is smaller.
    # Actually for 3x2x2: A is 3x2, B is 2x2, C is 3x2
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # This is just: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # We can do: M1 = A[i,0]*B[0,0], M2 = A[i,0]*B[0,1], M3 = A[i,1]*B[1,0], M4 = A[i,1]*B[1,1]
    # That's 4 per row, 12 total. But we can share B factors.
    
    # Actually: M1 = (A[i,0]+A[i,1])*B[0,0], M2 = A[i,1]*B[1,0], M3 = A[i,0]*B[0,1], M4 = A[i,1]*B[1,1]
    # C[i,0] = M1 + M2 - A[i,1]*B[0,0]... no.
    
    # Let me just try: for 3x2x2, the minimum rank is 4 (trivial) or can we do better?
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i=0,1,2 and j=0,1
    # 6 output entries. Each rank-1 term gives 1 product.
    # Lower bound: need at least 4 (since 2x2x2 needs 7, but this is smaller)
    # Actually, I think 4 is optimal for 3x2x2.
    
    # So 4 blocks of 3x2x2 at rank 4 each = rank 16? That seems too low.
    # Wait, let me reconsider.
    
    # Actually, the issue is that in 3x2x2, we have:
    # A: 3x2 (6 entries), B: 2x2 (4 entries), C: 3x2 (6 entries)
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    
    # Trivial: 4 products per row pair... no.
    # For row i: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    #            C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    
    # We can define:
    # M_{i,0} = A[i,0]*B[0,0]  -> contributes to C[i,0]
    # M_{i,1} = A[i,0]*B[0,1]  -> contributes to C[i,1]  
    # M_{i,2} = A[i,1]*B[1,0]  -> contributes to C[i,0]
    # M_{i,3} = A[i,1]*B[1,1]  -> contributes to C[i,1]
    # Total: 12 products. But we can share!
    
    # M0 = A[0,0]*B[0,0], M1 = A[0,0]*B[0,1], M2 = A[0,1]*B[1,0], M3 = A[0,1]*B[1,1]
    # M4 = A[1,0]*B[0,0], M5 = A[1,0]*B[0,1], M6 = A[1,1]*B[1,0], M7 = A[1,1]*B[1,1]
    # M8 = A[2,0]*B[0,0], M9 = A[2,0]*B[0,1], M10 = A[2,1]*B[1,0], M11 = A[2,1]*B[1,1]
    # That's 12. Can we do better?
    
    # Try: M0 = (A[0,0]+A[1,0])*B[0,0], M1 = A[1,0]*B[0,0], M2 = A[0,0]*B[0,1], M3 = A[1,0]*B[0,1]
    # M4 = (A[0,1]+A[1,1])*B[1,0], M5 = A[1,1]*B[1,0], M6 = A[0,1]*B[1,1], M7 = A[1,1]*B[1,1]
    # M8 = A[2,0]*B[0,0], M9 = A[2,0]*B[0,1], M10 = A[2,1]*B[1,0], M11 = A[2,1]*B[1,1]
    # Still 12. The sharing doesn't help much here.
    
    # Let me just go with a known good approach and try to optimize.
    
    # Direct construction: trivial rank = 12*16 = 192, but we can do much better.
    # Actually trivial is: for each (i,k,j) where k matches, one product.
    # That's 3*4*4 = 48 products.
    
    # Let me try a structured approach with rank ~32-40
    
    # Use the block structure:
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x4
    # C = A1*B1 + A2*B2
    
    # A1*B1: 3x2 * 2x4 = 3x4. This is like two 3x2x2 multiplications side by side.
    # A2*B2: same.
    
    # For 3x2x2, let's use rank 7 (Strassen-like for the 2x2 part, extended to 3 rows)
    # Actually, let me just try to find a good decomposition numerically.
    
    # Build the full tensor
    tensor = np.zeros((12, 16, 12))
    for i in range(3):
        for k in range(4):
            for j in range(4):
                a_idx = i * 4 + k
                b_idx = k * 4 + j
                c_idx = j * 3 + i  # W[k*n+i, r] convention: c = k*n+i where k is B's col, i is A's row
                # Wait, the convention says W[k*n+i, r]
                # So for output C[i,j], the index is j*n + i = j*3 + i
                tensor[a_idx, b_idx, c_idx] = 1.0
    
    # Try rank 32, 33, ... with optimization
    # This is a hard problem. Let me try a simpler approach.
    
    # Actually, let me try the approach from the current program but verify and potentially improve.
    # The current program gets rank 44. Let me see if I can do better.
    
    # Key insight: for 3x2x2, what's the minimum rank?
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # This is a "slice" of the 3x2x2 tensor.
    # I believe the rank is 4 (you need 4 products per "column" of the output in some sense).
    
    # Let me just try to construct a rank-36 solution:
    # Use Strassen for 2x2x2 blocks where possible.
    
    # Alternative: think of it as 3 * (1x4 * 4x4) but that's not helpful.
    
    # Let me try a direct numerical optimization for small ranks.
    
    def try_rank(r, max_time=30):
        """Try to find rank-r decomposition using gradient descent."""
        np.random.seed(42)
        U = np.random.randn(12, r) * 0.1
        V = np.random.randn(16, r) * 0.1
        W = np.random.randn(12, r) * 0.1
        
        lr = 0.001
        for it in range(2000):
            if time.time() > start_time + max_time:
                break
            
            # Compute gradient efficiently
            # loss = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            # grad_U[a,r] = 2 * sum_{b,c} (pred[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            
            # This is expensive. Let's use a simpler approach.
            # Just check if we can find a solution.
            
            # For now, skip this and use a known construction.
            pass
        
        return None
    
    # Let me just use a solid construction.
    # Rank 48 is trivial. Let me try to get to rank 40 or so.
    
    # Construction: Use the fact that we can share some products.
    # 
    # For the 4x4 B matrix, think of it as 4x4.
    # For each column j of B, we compute C[:,j] = A * B[:,j]
    # C[:,j] is 3x1, computed as 3x4 * 4x1.
    # 
    # 3x4 * 4x1: C[i] = sum_k A[i,k]*B[k,j]
    # This is like 4 dot products of length 4... no, it's 3 outputs.
    # 
    # Actually, let me just go with a known result.
    # 
    # I'll use a block-circulant or similar approach.
    
    # For now, let me return a valid rank-48 solution as fallback,
    # and try to find something better.
    
    # Trivial rank-48 solution:
    U_triv = np.zeros((12, 48))
    V_triv = np.zeros((16, 48))
    W_triv = np.zeros((12, 48))
    
    idx = 0
    for i in range(3):
        for k in range(4):
            for j in range(4):
                a_idx = i * 4 + k
                b_idx = k * 4 + j
                c_idx = j * 3 + i
                U_triv[a_idx, idx] = 1
                V_triv[b_idx, idx] = 1
                W_triv[c_idx, idx] = 1
                idx += 1
    
    # Verify
    def verify(U, V, W):
        for a in range(12):
            a_row, a_col = divmod(a, 4)
            for b in range(16):
                b_row, b_col = divmod(b, 4)
                if a_col != b_row:
                    continue
                c_idx = b_col * 3 + a_row
                expected = 1.0
                computed = np.sum(U[a, :] * V[b, :] * W[c_idx, :])
                if abs(computed - expected) > 1e-10:
                    return False
        return True
    
    if not verify(U_triv, V_triv, W_triv):
        # Shouldn't happen
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }
    
    # Now try to improve. Use the current program's approach: rank 44.
    # Let me implement that and see if I can do better.
    
    # The current approach: 4 blocks of 3x2x2, each at rank 11 (7 Strassen + 4 trivial)
    # Total: 4 * 11 = 44
    
    # Can we do better? What if we can share factors between blocks?
    
    # Let me try a direct optimization with a reasonable rank.
    # Given time constraints, let me just return the rank-44 solution.
    
    # Actually, let me try rank 40 with a quick optimization.
    
    # For simplicity and reliability, return the known rank-44 solution.
    
    # Build rank-44 solution using the block approach from the current program.
    rank_block = 11
    total_rank = 4 * rank_block
    
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # For each block (a_col_start, b_row_start, b_col_start):
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    def add_block(block_idx, a_col_start, b_row_start, b_col_start):
        col_offset = block_idx * rank_block
        
        # Local 3x2x2 decomposition
        # A_local: 3x2, B_local: 2x2, C_local: 3x2
        # Using rank 11: 7 Strassen + 4 trivial for row 2
        
        # Strassen for rows 0,1 (2x2x2 part):
        # M0 = (A00+A11)*(B00+B11)
        U[0*4+a_col_start, col_offset+0] += 1  # A[0,0]
        U[1*4+a_col_start+1, col_offset+0] += 1  # A[1,1]
        V[b_row_start*4+b_col_start, col_offset+0] += 1  # B[0,0]
        V[(b_row_start+1)*4+b_col_start+1, col_offset+0] += 1  # B[1,1]
        W[0*3+0, col_offset+0] += 1  # C[0,0]
        W[1*3+1, col_offset+0] += 1  # C[1,1]
        
        # M1 = (A10+A11)*B00
        U[1*4+a_col_start, col_offset+1] += 1  # A[1,0]
        U[1*4+a_col_start+1, col_offset+1] += 1  # A[1,1]
        V[b_row_start*4+b_col_start, col_offset+1] += 1  # B[0,0]
        W[1*3+0, col_offset+1] += 1  # C[1,0]
        W[1*3+1, col_offset+1] += -1  # C[1,1]
        
        # M2 = A00*(B10-B00)
        U[0*4+a_col_start, col_offset+2] += 1  # A[0,0]
        V[(b_row_start+1)*4+b_col_start, col_offset+2] += 1  # B[1,0]
        V[b_row_start*4+b_col_start, col_offset+2] += -1  # B[0,0]
        W[0*3+1, col_offset+2] += 1  # C[0,1]
        W[1*3+1, col_offset+2] += 1  # C[1,1]
        
        # M3 = A11*(B01-B11)
        U[1*4+a_col_start+1, col_offset+3] += 1  # A[1,1]
        V[b_row_start*4+b_col_start+1, col_offset+3] += 1  # B[0,1]
        V[(b_row_start+1)*4+b_col_start+1, col_offset+3] += -1  # B[1,1]
        W[0*3+0, col_offset+3] += 1  # C[0,0]
        W[1*3+0, col_offset+3] += 1  # C[1,0]
        
        # M4 = (A00+A01)*B11
        U[0*4+a_col_start, col_offset+4] += 1  # A[0,0]
        U[0*4+a_col_start+1, col_offset+4] += 1  # A[0,1]
        V[(b_row_start+1)*4+b_col_start+1, col_offset+4] += 1  # B[1,1]
        W[0*3+0, col_offset+4] += -1  # C[0,0]
        W[0*3+1, col_offset+4] += 1  # C[0,1]
        
        # M5 = (A10-A00)*(B00+B01)
        U[1*4+a_col_start, col_offset+5] += 1  # A[1,0]
        U[0*4+a_col_start, col_offset+5] += -1  # A[0,0]
        V[b_row_start*4+b_col_start, col_offset+5] += 1  # B[0,0]
        V[b_row_start*4+b_col_start+1, col_offset+5] += 1  # B[0,1]
        W[1*3+1, col_offset+5] += 1  # C[1,1]
        
        # M6 = (A01-A11)*(B10+B11)
        U[0*4+a_col_start+1, col_offset+6] += 1  # A[0,1]
        U[1*4+a_col_start+1, col_offset+6] += -1  # A[1,1]
        V[(b_row_start+1)*4+b_col_start, col_offset+6] += 1  # B[1,0]
        V[(b_row_start+1)*4+b_col_start+1, col_offset+6] += 1  # B[1,1]
        W[0*3+0, col_offset+6] += 1  # C[0,0]
        
        # Row 2: trivial 4 products
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        
        # M7 = A[2,0]*B[0,0]
        U[2*4+a_col_start, col_offset+7] += 1
        V[b_row_start*4+b_col_start, col_offset+7] += 1
        W[2*3+0, col_offset+7] += 1
        
        # M8 = A[2,1]*B[1,0]
        U[2*4+a_col_start+1, col_offset+8] += 1
        V[(b_row_start+1)*4+b_col_start, col_offset+8] += 1
        W[2*3+0, col_offset+8] += 1
        
        # M9 = A[2,0]*B[0,1]
        U[2*4+a_col_start, col_offset+9] += 1
        V[b_row_start*4+b_col_start+1, col_offset+9] += 1
        W[2*3+1, col_offset+9] += 1
        
        # M10 = A[2,1]*B[1,1]
        U[2*4+a_col_start+1, col_offset+10] += 1
        V[(b_row_start+1)*4+b_col_start+1, col_offset+10] += 1
        W[2*3+1, col_offset+10] += 1
    
    add_block(0, 0, 0, 0)  # A[:,0:2] * B[0:2,0:2]
    add_block(1, 2, 2, 0)  # A[:,2:4] * B[2:4,0:2]
    add_block(2, 0, 0, 2)  # A[:,0:2] * B[0:2,2:4]
    add_block(3, 2, 2, 2)  # A[:,2:4] * B[2:4,2:4]
    
    # Verify
    if not verify(U, V, W):
        # Fallback to trivial
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
