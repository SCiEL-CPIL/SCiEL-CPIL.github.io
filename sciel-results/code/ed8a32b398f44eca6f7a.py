import time
import numpy as np
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = 2, 2, 11
    rank_target = 39  # Known lower bound / Strassen-based rank
    
    # Target tensor T[i, j, k] = 1 if (i,j,k) is a valid multiplication index
    # T is shape (n*m, m*p, p*n) = (4, 22, 22)
    # Indices:
    # U: i (0..3), V: j (0..21), W: k (0..21)
    # T[i, j, k] = 1 if i = i1*m + j1, j = j1*p + k1, k = k1*n + i1
    # Let's build the target tensor explicitly
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                i_idx = i1 * m + j1
                j_idx = j1 * p + k1
                k_idx = k1 * n + i1
                T[i_idx, j_idx, k_idx] = 1.0
    
    # We want U, V, W of shape (4, R), (22, R), (22, R)
    # such that sum_r U[:,r] V[:,r]^T W[:,r]^T == T
    # This is equivalent to: for each r, U[:,r] V[:,r] W[:,r] is a rank-1 slice
    # Let u_r, v_r, w_r be columns.
    # Sum_r u_r v_r^T w_r^T = T
    
    # Strategy: Randomized search with sparsity constraints.
    # We know the solution is sparse. Schoolbook is very sparse.
    # Strassen-like solutions are also sparse.
    # We will try to find a decomposition of rank R.
    
    best_rank = n*m*p # 44 baseline
    best_solution = None
    
    # Generate baseline solution (Schoolbook) to ensure we have a valid return
    U_base = np.zeros((n*m, n*m*p))
    V_base = np.zeros((m*p, n*m*p))
    W_base = np.zeros((p*n, n*m*p))
    col = 0
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                i_idx = i1 * m + j1
                j_idx = j1 * p + k1
                k_idx = k1 * n + i1
                U_base[i_idx, col] = 1.0
                V_base[j_idx, col] = 1.0
                W_base[k_idx, col] = 1.0
                col += 1
    best_solution = (U_base, V_base, W_base)
    best_rank = n*m*p
    
    # Try to find lower rank solutions
    # We start from the lower bound. For 2x2x11, lower bound is likely 39 or 40?
    # Strassen for 2x2 is rank 7.
    # 2x2x11 can be seen as 2x2x2 (rank 7) + 2x2x2 (rank 7) + ...
    # 11 = 2*5 + 1.
    # 5 blocks of 2x2x2 (rank 7 each) = 35, plus 1 block of 2x2x1 (rank 2)?
    # Total rank 37? Or maybe overlapping allows reduction.
    # The prompt mentions rank 39 in inspirations. Let's aim for 39, 40, 41, 42, 43.
    
    ranks_to_try = list(range(39, best_rank))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try to find a rank R decomposition
        # We use a randomized gradient-free approach: 
        # 1. Initialize random sparse matrices
        # 2. Project onto the manifold of rank-R decompositions? 
        #    Actually, the problem is: find U,V,W such that T = sum U_r V_r W_r
        #    This is a system of bilinear/quadratic equations.
        #    
        #    A simpler heuristic: 
        #    Since the tensor is 0/1, and we expect integer/half-integer solutions,
        #    we can try to solve for one factor at a time.
        
        # Let's try a specific structured search for R=39, 40.
        # If we assume the solution is sparse, we can try to recover the support.
        
        # Alternative: Use the fact that we can combine Strassen blocks.
        # A 2x2x11 multiplication can be decomposed into:
        # 5 blocks of 2x2x2 (using Strassen, rank 7 each) = 35
        # 1 block of 2x2x1 (using schoolbook, rank 2)
        # Total rank 37? 
        # Wait, the dimensions are 2x2 * 2x11.
        # Output is 2x11.
        # We can split the 11 columns into 5 pairs and 1 single.
        # Each 2x2x2 block costs rank 7 (Strassen).
        # Each 2x2x1 block costs rank 2 (schoolbook: C11 = A11B11 + A12B21, C21 = A21B11 + A22B21... wait)
        # For 2x2 * 2x1:
        # C = A B
        # C11 = A11 B1 + A12 B2
        # C21 = A21 B1 + A22 B2
        # Rank 4 schoolbook. Can we do better?
        # 2x2x1 tensor rank is 4? No, border rank is 4, rank is 4?
        # Actually, 2x2x1 is just matrix-vector. Rank is 4 (schoolbook).
        # So 5*7 + 4 = 39.
        
        # So Rank 39 is achievable by block decomposition!
        # Let's construct it explicitly.
        
        U_list = []
        V_list = []
        W_list = []
        
        # 5 blocks of size 2x2x2
        # Indices for B (2x11):
        # Block 0: cols 0,1
        # Block 1: cols 2,3
        # Block 2: cols 4,5
        # Block 3: cols 6,7
        # Block 4: cols 8,9
        # Block 5: col 10
        
        # Strassen for 2x2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 + M6
        
        # We need to map these to the global indices.
        # A is 2x2 (indices 0,1,2,3)
        # B is 2x11 (indices 0..21)
        # C is 2x11 (indices 0..21)
        
        # Let's define the Strassen factors for a generic 2x2x2 block starting at B column offset `offset`
        def get_strassen_block(offset):
            # Returns U_cols, V_cols, W_cols for this block
            # U is 4x7, V is 4x7 (for the 2x2 part of B), W is 4x7
            # Note: V here corresponds to the local 2x2 B block.
            # We need to map local B indices (0,1,2,3) to global (offset, offset+1, offset+2, offset+3)
            
            # Local indices for B: 0,1,2,3
            # Global indices for B: offset, offset+1, offset+2, offset+3
            
            # A indices are always 0,1,2,3
            # C indices are always 0,1,2,3 (local to the 2x2 output block)
            # Global C indices: offset, offset+1, offset+2, offset+3
            
            U_block = np.zeros((4, 7))
            V_block = np.zeros((4, 7))
            W_block = np.zeros((4, 7))
            
            # M1
            # (A11 + A22) * (B11 + B22)
            # A11 is idx 0, A22 is idx 3
            # B11 is local 0, B22 is local 3
            U_block[0, 0] = 1; U_block[3, 0] = 1
            V_block[0, 0] = 1; V_block[3, 0] = 1
            # C11 += M1, C22 += M1
            # Local C11 is idx 0, C22 is idx 3
            W_block[0, 0] = 1; W_block[3, 0] = 1
            
            # M2
            # (A21 + A22) * B11
            # A21 is idx 2, A22 is idx 3
            # B11 is local 0
            U_block[2, 1] = 1; U_block[3, 1] = 1
            V_block[0, 1] = 1
            # C21 += M2
            # Local C21 is idx 2
            W_block[2, 1] = 1
            
            # M3
            # A11 * (B12 - B22)
            # A11 is idx 0
            # B12 is local 1, B22 is local 3
            U_block[0, 2] = 1
            V_block[1, 2] = 1; V_block[3, 2] = -1
            # C12 += M3
            # Local C12 is idx 1
            W_block[1, 2] = 1
            
            # M4
            # A22 * (B21 - B11)
            # A22 is idx 3
            # B21 is local 2, B11 is local 0
            U_block[3, 3] = 1
            V_block[2, 3] = 1; V_block[0, 3] = -1
            # C11 -= M4, C21 += M4
            # Local C11 is idx 0, C21 is idx 2
            W_block[0, 3] = -1; W_block[2, 3] = 1
            
            # M5
            # (A11 + A12) * B22
            # A11 is idx 0, A12 is idx 1
            # B22 is local 3
            U_block[0, 4] = 1; U_block[1, 4] = 1
            V_block[3, 4] = 1
            # C11 += M5, C12 += M5
            # Local C11 is idx 0, C12 is idx 1
            W_block[0, 4] = 1; W_block[1, 4] = 1
            
            # M6
            # (A21 - A11) * (B11 + B12)
            # A21 is idx 2, A11 is idx 0
            # B11 is local 0, B12 is local 1
            U_block[2, 5] = 1; U_block[0, 5] = -1
            V_block[0, 5] = 1; V_block[1, 5] = 1
            # C22 += M6
            # Local C22 is idx 3
            W_block[3, 5] = 1
            
            # M7
            # (A12 - A22) * (B21 + B22)
            # A12 is idx 1, A22 is idx 3
            # B21 is local 2, B22 is local 3
            U_block[1, 6] = 1; U_block[3, 6] = -1
            V_block[2, 6] = 1; V_block[3, 6] = 1
            # C11 += M7, C22 += M7
            # Local C11 is idx 0, C22 is idx 3
            W_block[0, 6] = 1; W_block[3, 6] = 1
            
            # Now map local indices to global
            # U is same (A is always 2x2)
            # V: local col c -> global col offset + c
            # W: local col c -> global col offset + c
            
            # We return the blocks, but we need to assemble them into global matrices.
            # Since the blocks are independent in terms of B and C columns,
            # we can just stack the columns.
            # However, U columns might overlap in indices? No, U is just A.
            # The "rank" is the number of columns.
            # So we can just concatenate the U, V, W blocks horizontally.
            # BUT, we must be careful: V and W have different global indices.
            # V_block has non-zeros only in rows [offset, offset+4).
            # W_block has non-zeros only in rows [offset, offset+4).
            # So we can create global V and W by placing these blocks at the correct row offsets.
            
            return U_block, V_block, W_block

        U_global = np.zeros((4, 39))
        V_global = np.zeros((22, 39))
        W_global = np.zeros((22, 39))
        
        col_offset = 0
        # 5 Strassen blocks
        for b in range(5):
            offset = b * 2
            Ub, Vb, Wb = get_strassen_block(offset)
            U_global[:, col_offset:col_offset+7] = Ub
            # Place Vb in global V
            V_global[offset:offset+4, col_offset:col_offset+7] = Vb
            # Place Wb in global W
            W_global[offset:offset+4, col_offset:col_offset+7] = Wb
            col_offset += 7
            
        # 1 Schoolbook block for the last column (col 10)
        # 2x2 * 2x1
        # C11 = A11 B1 + A12 B2
        # C21 = A21 B1 + A22 B2
        # Rank 4
        # U is 4x4, V is 2x4, W is 2x4
        
        U_sb = np.zeros((4, 4))
        V_sb = np.zeros((2, 4))
        W_sb = np.zeros((2, 4))
        
        # Term 1: A11 * B1
        # A11 is idx 0. B1 is local 0. C11 is local 0.
        U_sb[0, 0] = 1
        V_sb[0, 0] = 1
        W_sb[0, 0] = 1
        
        # Term 2: A12 * B2
        # A12 is idx 1. B2 is local 1. C11 is local 0.
        U_sb[1, 1] = 1
        V_sb[1, 1] = 1
        W_sb[0, 1] = 1
        
        # Term 3: A21 * B1
        # A21 is idx 2. B1 is local 0. C21 is local 1.
        U_sb[2, 2] = 1
        V_sb[0, 2] = 1
        W_sb[1, 2] = 1
        
        # Term 4: A22 * B2
        # A22 is idx 3. B2 is local 1. C21 is local 1.
        U_sb[3, 3] = 1
        V_sb[1, 3] = 1
        W_sb[1, 3] = 1
        
        offset = 10
        U_global[:, col_offset:col_offset+4] = U_sb
        V_global[offset:offset+2, col_offset:col_offset+4] = V_sb
        W_global[offset:offset+2, col_offset:col_offset+4] = W_sb
        
        # Verify the solution
        # Reconstruct T from U, V, W
        T_recon = np.zeros((4, 22, 22), dtype=np.float64)
        for r in range(39):
            u = U_global[:, r]
            v = V_global[:, r]
            w = W_global[:, r]
            # T_recon += np.einsum('i,j,k->ijk', u, v, w)
            T_recon += np.outer(np.outer(u, v), w)
            
        if np.allclose(T_recon, T, atol=1e-9):
            best_rank = 39
            best_solution = (U_global, V_global, W_global)
            break # Found rank 39, which is likely minimal or close to it.
        
    # If we didn't find 39, we still have the baseline 44.
    # But we should convert to the required format: list of lists of fractions/integers.
    
    U, V, W = best_solution
    
    def to_list(matrix):
        # Convert numpy array to list of lists of Fractions
        # Entries are integers or half-integers? 
        # Our construction uses integers.
        res = []
        for row in matrix:
            r = []
            for val in row:
                # Check if it's close to an integer
                if abs(val - round(val)) < 1e-9:
                    r.append(Fraction(int(round(val))))
                else:
                    # Try half-integer
                    half = val * 2
                    if abs(half - round(half)) < 1e-9:
                        r.append(Fraction(int(round(half)), 2))
                    else:
                        r.append(Fraction(val))
            res.append(r)
        return res
        
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
