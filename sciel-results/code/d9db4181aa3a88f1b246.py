"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    # Indexing:
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[k,i] (trace convention)
    
    # Strategy: Decompose into 4x4x4 and 4x4x2 parts.
    # The 4x4x4 part can use a rank-21 decomposition (based on Strassen/Winograd).
    # The 4x4x2 part can use a rank-16 decomposition (schoolbook is 32, but we can do better? 
    # Actually, rank(4x4x2) >= 16? Let's just use schoolbook for the 2 columns to be safe and fast, 
    # or find a better decomposition. 
    # Wait, the task asks for rank-minimal. 
    # Known ranks:
    # 4x4x4: rank = 21
    # 4x4x2: rank = 16? Let's check.
    # The lower bound for 4x4x2 is max(16, 8, 8) = 16? No, that's not the right lower bound.
    # The "slice rank" lower bound is max(n*p, m*n, m*p) for the tensor T_{ijk}.
    # For 4x4x2, slices are 4x2 (8), 4x2 (8), 4x4 (16). So lower bound is 16.
    # Is rank 16 achievable? Yes, it's just the schoolbook for the 2 columns if we treat them separately?
    # Schoolbook for 4x4x2 is 4*4*2 = 32.
    # But we can compute C[:, 0:2] = A * B[:, 0:2].
    # This is equivalent to two 4x4x1 problems? No.
    # Let's use a known decomposition for 4x4x2.
    # Actually, we can just use the 21-rank decomposition for the first 4 columns and add 16 more for the last 2?
    # Total rank 37.
    # But can we do better by sharing structure?
    # Let's first implement the rank-21 for 4x4x4 and rank-32 for 4x4x2 (schoolbook) to get a valid score < 96.
    # Then we can try to optimize the 4x4x2 part.
    
    # Rank-21 decomposition for 4x4x4:
    # This is a known result. I will hardcode the indices and values.
    # The decomposition is based on splitting 4x4 into four 2x2 blocks and using Strassen,
    # but with optimizations.
    # Actually, let's use a simpler approach:
    # We can use the rank-7 Strassen for 2x2x2.
    # To multiply 4x4 by 4x4, we can split into 4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11], [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    # Each 2x2x2 multiplication has rank 7.
    # There are 4 blocks in C, each requiring 4 2x2x2 multiplications?
    # No, each 2x2 block of C is a sum of two 2x2 products.
    # C00 = A00*B00 + A01*B10 -> 2 mults (rank 14)
    # C01 = A00*B01 + A01*B11 -> 2 mults (rank 14)
    # C10 = A10*B00 + A11*B10 -> 2 mults (rank 14)
    # C11 = A10*B01 + A11*B11 -> 2 mults (rank 14)
    # Total rank = 14*4 = 56. This is worse than schoolbook (64) but better than naive sum of ranks?
    # No, schoolbook for 4x4x4 is 64. 56 is better.
    # But we know rank is 21. How?
    # The rank-21 decomposition is more clever. It shares intermediate products.
    # I will use a known explicit rank-21 decomposition.
    
    # Let's try to find a rank-32 decomposition for 4x4x6 first? 
    # No, let's just get a valid one.
    # Schoolbook is 96.
    # Split into 4x4x4 (rank 21) and 4x4x2 (rank 16). Total 37.
    # Let's implement rank-21 for 4x4x4.
    
    # I will use a Python list of tuples (i, j, k, u_val, v_val, w_val) for the rank-21 decomposition.
    # Since I don't have the exact 21-term decomposition memorized perfectly, I will use a reliable algorithm.
    # Actually, I can use the "Bostan-Gaudry-Schost" or similar known tables.
    # Or, I can use a simpler rank-27 decomposition (3x3 Strassen?) No, 4x4 is 2x2 blocks.
    # Let's use the 56-rank Strassen-based decomposition for 4x4x4 and 32-rank schoolbook for 4x4x2.
    # Total rank = 56 + 32 = 88. This is better than 96.
    
    # Can I do better for 4x4x2?
    # 4x4x2 schoolbook is 32.
    # Lower bound is 16.
    # Is there a rank-16 decomposition?
    # Yes, if we view it as computing 2 columns of C.
    # C_k0 = sum_j A_ij B_jk0
    # This is essentially two independent 4x4x1 multiplications? No, they share A.
    # But the tensor rank is additive for disjoint supports?
    # If the supports in the B and C indices are disjoint, yes.
    # Here, B indices for col 0 and 1 are disjoint? No, they share the same B rows.
    # But the C columns are distinct.
    # Actually, rank(4x4x2) = 16 is known?
    # Let's assume rank 16 is possible for 4x4x2.
    # If so, total rank = 21 + 16 = 37.
    
    # Let's hardcode the rank-21 decomposition for 4x4x4.
    # I will retrieve it from a known source or derive it.
    # Deriving it on the fly is hard.
    # Let's use the rank-27 decomposition for 4x4x4 (3x3 Strassen?) No.
    # Let's use the rank-56 Strassen decomposition for 4x4x4 and rank-32 schoolbook for 4x4x2.
    # Total 88.
    
    # Wait, I can use a rank-21 decomposition for 4x4x4 if I can find it.
    # I will use a placeholder and verify it.
    
    # Actually, let's just use the schoolbook for the whole thing to be safe, then try to improve.
    # No, the current program scores 0.0 because it likely times out or fails verification.
    # I need a valid decomposition quickly.
    
    # Let's build the rank-88 decomposition (56 for 4x4x4 + 32 for 4x4x2).
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U.append((u_idx, u_val))
        V.append((v_idx, v_val))
        W.append((w_idx, w_val))
        
    # Part 1: 4x4x4 (columns 0-3 of B, rows 0-3 of C)
    # Strassen 2x2x2 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Block indices for 4x4:
    # A00: i=0,1 j=0,1
    # A01: i=0,1 j=2,3
    # A10: i=2,3 j=0,1
    # A11: i=2,3 j=2,3
    # B00: i=0,1 j=0,1 (B rows 0-1, cols 0-1)
    # B01: i=0,1 j=2,3
    # B10: i=2,3 j=0,1
    # B11: i=2,3 j=2,3
    # C00: i=0,1 j=0,1
    # C01: i=0,1 j=2,3
    # C10: i=2,3 j=0,1
    # C11: i=2,3 j=2,3
    
    # We need to map the 2x2x2 Strassen terms to the 4x4x4 indices.
    # This is tedious but doable.
    
    # Let's write a helper to add a 2x2x2 Strassen block.
    def add_strassen_2x2x2(a_i_base, a_j_base, b_i_base, b_j_base, c_i_base, c_j_base):
        # Adds 7 terms for multiplying 2x2 A block at (a_i, a_j) by 2x2 B block at (b_i, b_j)
        # to produce 2x2 C block at (c_i, c_j).
        # Indices are (row, col) for A, (row, col) for B, (row, col) for C.
        # U index: i*m + j
        # V index: i*p + j (here p=6, but for 4x4x4 part, we only use j=0..3)
        # W index: k*n + i (here n=4, for 4x4x4 part, we only use k=0..3)
        
        def u_idx(i, j): return i * m + j
        def v_idx(i, j): return i * p + j
        def w_idx(k, i): return k * n + i
        
        # M1
        add_term(u_idx(a_i_base+0, a_j_base+0), 1, v_idx(b_i_base+0, b_j_base+0), 1, w_idx(c_i_base+0, c_j_base+0), 1)
        add_term(u_idx(a_i_base+0, a_j_base+0), 1, v_idx(b_i_base+1, b_j_base+1), 1, w_idx(c_i_base+0, c_j_base+0), 1)
        add_term(u_idx(a_i_base+1, a_j_base+0), 1, v_idx(b_i_base+0, b_j_base+0), 1, w_idx(c_i_base+1, c_j_base+0), 1)
        add_term(u_idx(a_i_base+1, a_j_base+0), 1, v_idx(b_i_base+1, b_j_base+1), 1, w_idx(c_i_base+1, c_j_base+0), 1)
        add_term(u_idx(a_i_base+1, a_j_base+1), 1, v_idx(b_i_base+0, b_j_base+0), 1, w_idx(c_i_base+1, c_j_base+1), 1)
        add_term(u_idx(a_i_base+1, a_j_base+1), 1, v_idx(b_i_base+1, b_j_base+1), 1, w_idx(c_i_base+1, c_j_base+1), 1)
        
        # This manual expansion is error-prone.
        # Let's use a standard library or a simpler known decomposition.
        # I will use the schoolbook for the entire 4x4x6 to ensure validity, then try to reduce.
        # No, I need to beat 96.
        
    # Let's try a different approach:
    # Use the rank-21 decomposition for 4x4x4 from a known list.
    # I will embed the 21 terms.
    
    # Terms for 4x4x4 rank 21:
    # (i, j, k, u, v, w)
    # This is too complex to write from scratch.
    
    # Alternative: Use rank 32 for 4x4x4 (Strassen 4x4x4?)
    # Actually, rank(4x4x4) = 21.
    # rank(4x4x2) = 16.
    # Total 37.
    
    # I will use a generic tensor rank algorithm? No, too slow.
    
    # I will use the following strategy:
    # 1. Construct the 4x4x4 part using the rank-21 decomposition (hardcoded).
    # 2. Construct the 4x4x2 part using the rank-16 decomposition (hardcoded or schoolbook).
    
    # Since I cannot reliably hardcode the 21 terms, I will use the rank-56 Strassen decomposition for 4x4x4.
    # And rank-32 schoolbook for 4x4x2.
    # Total rank 88.
    
    # Let's implement the rank-56 Strassen for 4x4x4.
    
    # Initialize empty lists
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term_strassen(i_u, j_u, val_u, i_v, j_v, val_v, i_w, j_w, val_w):
        U_terms.append((i_u * m + j_u, val_u))
        V_terms.append((i_v * p + j_v, val_v))
        W_terms.append((i_w * n + j_w, val_w))
        
    # Strassen 2x2x2 block function
    def strassen_2x2(a_i, a_j, b_i, b_j, c_i, c_j):
        # 7 terms
        # M1 = (A00+A11)(B00+B11)
        add_term_strassen(a_i, a_j, 1, b_i, b_j, 1, c_i, c_j, 1)
        add_term_strassen(a_i, a_j, 1, b_i+1, b_j+1, 1, c_i, c_j, 1)
        add_term_strassen(a_i+1, a_j+1, 1, b_i, b_j, 1, c_i+1, c_j+1, 1)
        add_term_strassen(a_i+1, a_j+1, 1, b_i+1, b_j+1, 1, c_i+1, c_j+1, 1)
        # M2 = (A10+A11)B00
        add_term_strassen(a_i+1, a_j, 1, b_i, b_j, 1, c_i+1, c_j, 1)
        add_term_strassen(a_i+1, a_j+1, 1, b_i, b_j, 1, c_i+1, c_j, 1)
        # M3 = A00(B10-B11)
        add_term_strassen(a_i, a_j, 1, b_i+1, b_j, 1, c_i, c_j+1, -1) # Wait, C indices are (row, col)
        # Let's check C indices.
        # C[i,j] = sum_k A[i,k] B[k,j]
        # W index is k*n + i.
        # So for C block (c_i, c_j), the rows are c_i, c_i+1 and cols are c_j, c_j+1.
        # The term contributes to W[c_row*n + c_col].
        # So for M3, it contributes to C[c_i, c_j+1] and C[c_i+1, c_j+1]?
        # No, Strassen:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M7
        
        # Let's redo the indices carefully.
        # This is getting complicated. I will just use schoolbook for 4x4x6 to ensure it works, 
        # and then try to reduce the rank by using a known 4x4x4 decomposition.
        
    # I will use the schoolbook for 4x4x6 (rank 96) as a fallback.
    # But I need to beat 96.
    # Let's use rank 32 for 4x4x4 (which is possible? No, 21 is the rank).
    # If I can't find the 21 terms, I will use rank 56 for 4x4x4 and rank 32 for 4x4x2.
    
    # Let's try to find a rank-32 decomposition for 4x4x4.
    # It is known that rank(4x4x4) <= 21.
    # I will use a random search to find a valid decomposition with rank < 96.
    # I will start with rank 40 and increase.
    
    import random
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook fallback
    U_school = np.zeros((16, 96))
    V_school = np.zeros((24, 96))
    W_school = np.zeros((24, 96))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U_school[i*4+j, col] = 1
                V_school[j*6+k, col] = 1
                W_school[k*4+i, col] = 1
                col += 1
                
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = 96
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    val = np.dot(U[i*4+j], np.multiply(V[j*6+k], W[k*4+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a decomposition with rank < 96
    # Use a structured initialization
    for target_rank in range(40, 96, 4):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            U = np.zeros((16, target_rank))
            V = np.zeros((24, target_rank))
            W = np.zeros((24, target_rank))
            
            # Initialize with random patterns
            for r in range(target_rank):
                # Pick random indices
                i = np.random.randint(0, 4)
                j = np.random.randint(0, 4)
                k = np.random.randint(0, 6)
                
                U[i*4+j, r] = np.random.choice([-1, 1])
                V[j*6+k, r] = np.random.choice([-1, 1])
                W[k*4+i, r] = np.random.choice([-1, 1])
                
                # Add secondary terms
                if np.random.random() < 0.5:
                    i2 = np.random.randint(0, 4)
                    j2 = np.random.randint(0, 4)
                    k2 = np.random.randint(0, 6)
                    U[i2*4+j2, r] += np.random.choice([-1, 0, 1])
                    V[j2*6+k2, r] += np.random.choice([-1, 0, 1])
                    W[k2*4+i2, r] += np.random.choice([-1, 0, 1])
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = target_rank
                break
        if best_rank < 96:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
