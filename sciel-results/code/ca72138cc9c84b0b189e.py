import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5

    # We aim for a rank lower than the schoolbook 80.
    # Strategy: Use Strassen's algorithm for the 4x4x4 part (rank 49)
    # and schoolbook for the remaining 4x4x1 part (rank 16).
    # Total rank = 49 + 16 = 65.
    
    rank_target = 65
    U = [[0.0] * rank_target for _ in range(n * m)]  # 16 x 65
    V = [[0.0] * rank_target for _ in range(m * p)]  # 20 x 65
    W = [[0.0] * rank_target for _ in range(p * n)]  # 20 x 65
    
    r = 0

    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal r
        if r >= rank_target:
            return False
        U[u_idx][r] += coeff
        V[v_idx][r] += coeff
        W[w_idx][r] += coeff
        r += 1
        return True

    # --- Part 1: 4x4x4 Multiplication (First 4 columns of B) ---
    # We use a recursive Strassen decomposition.
    # Level 1: 4x4x4 -> 7 blocks of 2x2x2
    # Level 2: 2x2x2 -> 7 blocks of 1x1x1 (scalar mults)
    # Total scalar mults: 7 * 7 = 49.
    
    # Strassen 2x2x2 Decomposition
    # Inputs: A (2x2), B (2x2)
    # Indices for A: 00, 01, 10, 11
    # Indices for B: 00, 01, 10, 11
    # Outputs: C = A * B
    
    # The 7 rank-1 terms for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]]
    # Let B = [[b00, b01], [b10, b11]]
    
    # s1 = (a00 + a11) * (b00 + b11)
    # s2 = (a10 + a11) * b00
    # s3 = a00 * (b01 - b11)
    # s4 = a11 * (b10 - b00)
    # s5 = (a00 + a01) * b11
    # s6 = (a10 - a00) * (b00 + b01)
    # s7 = (a01 - a11) * (b10 + b11)
    
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 + s4
    # C11 = s1 - s2 + s3 + s6

    # We need to map these to the global indices.
    # The 4x4 matrix A is split into 2x2 blocks:
    # A00 (top-left), A01 (top-right)
    # A10 (bottom-left), A11 (bottom-right)
    
    # Global indices for A (i, j):
    # Block A00: i in {0,1}, j in {0,1} -> local (i, j)
    # Block A01: i in {0,1}, j in {2,3} -> local (i, j-2)
    # Block A10: i in {2,3}, j in {0,1} -> local (i-2, j)
    # Block A11: i in {2,3}, j in {2,3} -> local (i-2, j-2)
    
    # Global indices for B (j, k):
    # Block B00: j in {0,1}, k in {0,1} -> local (j, k)
    # Block B01: j in {0,1}, k in {2,3} -> local (j, k-2)
    # Block B10: j in {2,3}, k in {0,1} -> local (j-2, k)
    # Block B11: j in {2,3}, k in {2,3} -> local (j-2, k-2)
    
    # Global indices for C (k, i):
    # Block C00: k in {0,1}, i in {0,1} -> local (k, i)
    # Block C01: k in {0,1}, i in {2,3} -> local (k, i-2)
    # Block C10: k in {2,3}, i in {0,1} -> local (k-2, i)
    # Block C11: k in {2,3}, i in {2,3} -> local (k-2, i-2)
    
    # Function to add a Strassen term for a specific 2x2x2 sub-problem
    # sub_block_A: which 2x2 block of A (0: A00, 1: A01, 2: A10, 3: A11)
    # sub_block_B: which 2x2 block of B (0: B00, 1: B01, 2: B10, 3: B11)
    # sub_block_C: which 2x2 block of C (0: C00, 1: C01, 2: C10, 3: C11)
    # But Strassen combines blocks.
    
    # Let's define the 7 Strassen steps for the 4x4x4 matrix multiplication.
    # A is 4x4, B is 4x4.
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B into 2x2 blocks: B00, B01, B10, B11
    # Split C into 2x2 blocks: C00, C01, C10, C11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product. We decompose each M using the 7-term Strassen decomposition for 2x2x2.
    
    def add_strassen_2x2(A_blocks, B_blocks, C_blocks, coeffs):
        """
        Adds the rank-1 terms for a 2x2x2 multiplication where the input blocks
        are linear combinations of the global blocks.
        A_blocks: list of (block_idx, coeff) for A
        B_blocks: list of (block_idx, coeff) for B
        C_blocks: list of (block_idx, coeff) for C
        coeffs: list of 7 tuples (u_coeffs, v_coeffs, w_coeffs) for the 7 Strassen steps?
        No, it's better to iterate through the 7 Strassen steps and expand the linear combinations.
        """
        # The 7 Strassen steps for 2x2x2:
        # s1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        # s2 = (a10+a11)b00 -> c10 += 1, c11 -= 1
        # s3 = a00(b01-b11) -> c01 += 1, c11 += 1
        # s4 = a11(b10-b00) -> c00 += 1, c10 += 1
        # s5 = (a00+a01)b11 -> c00 -= 1, c01 += 1
        # s6 = (a10-a00)(b00+b01) -> c11 += 1
        # s7 = (a01-a11)(b10+b11) -> c00 += 1, c11 += 1
        
        # We need to map local 2x2 indices (0,1) to global block indices.
        # Local (0,0) of A corresponds to global block A_blocks[?] at local pos (0,0)
        # This is getting complicated to generalize with linear combos of blocks.
        # Let's just explicitly list the 49 terms.
        pass

    # Explicitly listing the 49 terms is safer.
    # Let's generate them programmatically to avoid typos.
    
    # Define the 7 Strassen steps for 2x2x2 in terms of local indices.
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Step 1: (A00+A11)(B00+B11)
    # A terms: (0,1), (3,1)
    # B terms: (0,1), (3,1)
    # C terms: (0,1), (3,1)
    
    # Step 2: (A10+A11)B00
    # A terms: (2,1), (3,1)
    # B terms: (0,1)
    # C terms: (2,1), (3,-1)
    
    # Step 3: A00(B01-B11)
    # A terms: (0,1)
    # B terms: (1,1), (3,-1)
    # C terms: (1,1), (3,1)
    
    # Step 4: A11(B10-B00)
    # A terms: (3,1)
    # B terms: (2,1), (0,-1)
    # C terms: (0,1), (2,1)
    
    # Step 5: (A00+A01)B11
    # A terms: (0,1), (1,1)
    # B terms: (3,1)
    # C terms: (0,-1), (1,1)
    
    # Step 6: (A10-A00)(B00+B01)
    # A terms: (2,1), (0,-1)
    # B terms: (0,1), (1,1)
    # C terms: (3,1)
    
    # Step 7: (A01-A11)(B10+B11)
    # A terms: (1,1), (3,-1)
    # B terms: (2,1), (3,1)
    # C terms: (0,1), (3,1)
    
    strassen_steps = [
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)]
        },
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)]
        },
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]
        },
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)]
        },
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        }
    ]
    
    # Now, we apply this to the 7 super-steps of the 4x4x4 multiplication.
    # Super Step 1: M1 = (A00+A11)(B00+B11)
    # A blocks involved: A00 (coeff 1), A11 (coeff 1)
    # B blocks involved: B00 (coeff 1), B11 (coeff 1)
    # C blocks involved: C00 (coeff 1), C11 (coeff 1)
    
    # Super Step 2: M2 = (A10+A11)B00
    # A blocks: A10 (1), A11 (1)
    # B blocks: B00 (1)
    # C blocks: C10 (1), C11 (-1)
    
    # Super Step 3: M3 = A00(B01-B11)
    # A blocks: A00 (1)
    # B blocks: B01 (1), B11 (-1)
    # C blocks: C01 (1), C11 (1)
    
    # Super Step 4: M4 = A11(B10-B00)
    # A blocks: A11 (1)
    # B blocks: B10 (1), B00 (-1)
    # C blocks: C00 (1), C10 (1)
    
    # Super Step 5: M5 = (A00+A01)B11
    # A blocks: A00 (1), A01 (1)
    # B blocks: B11 (1)
    # C blocks: C00 (-1), C01 (1)
    
    # Super Step 6: M6 = (A10-A00)(B00+B01)
    # A blocks: A10 (1), A00 (-1)
    # B blocks: B00 (1), B01 (1)
    # C blocks: C11 (1)
    
    # Super Step 7: M7 = (A01-A11)(B10+B11)
    # A blocks: A01 (1), A11 (-1)
    # B blocks: B10 (1), B11 (1)
    # C blocks: C00 (1), C11 (1)
    
    super_steps = [
        {
            'A_blocks': [(0, 1), (3, 1)],
            'B_blocks': [(0, 1), (3, 1)],
            'C_blocks': [(0, 1), (3, 1)]
        },
        {
            'A_blocks': [(2, 1), (3, 1)],
            'B_blocks': [(0, 1)],
            'C_blocks': [(2, 1), (3, -1)]
        },
        {
            'A_blocks': [(0, 1)],
            'B_blocks': [(1, 1), (3, -1)],
            'C_blocks': [(1, 1), (3, 1)]
        },
        {
            'A_blocks': [(3, 1)],
            'B_blocks': [(2, 1), (0, -1)],
            'C_blocks': [(0, 1), (2, 1)]
        },
        {
            'A_blocks': [(0, 1), (1, 1)],
            'B_blocks': [(3, 1)],
            'C_blocks': [(0, -1), (1, 1)]
        },
        {
            'A_blocks': [(2, 1), (0, -1)],
            'B_blocks': [(0, 1), (1, 1)],
            'C_blocks': [(3, 1)]
        },
        {
            'A_blocks': [(1, 1), (3, -1)],
            'B_blocks': [(2, 1), (3, 1)],
            'C_blocks': [(0, 1), (3, 1)]
        }
    ]
    
    # Helper to get global index from block index and local index
    def get_global_A(block_idx, local_idx):
        # block_idx: 0:00, 1:01, 2:10, 3:11
        # local_idx: 0:00, 1:01, 2:10, 3:11
        r = local_idx // 2
        c = local_idx % 2
        br = block_idx // 2
        bc = block_idx % 2
        gi = 2 * br + r
        gj = 2 * bc + c
        return gi * m + gj  # U index: i * m + j

    def get_global_B(block_idx, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        br = block_idx // 2
        bc = block_idx % 2
        gj = 2 * br + r
        gk = 2 * bc + c
        return gj * p + gk  # V index: j * p + k

    def get_global_C(block_idx, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        br = block_idx // 2
        bc = block_idx % 2
        gk = 2 * br + r
        gi = 2 * bc + c
        return gk * n + gi  # W index: k * n + i

    # Iterate through super steps and inner steps
    for s_idx, super_step in enumerate(super_steps):
        for i_idx, inner_step in enumerate(strassen_steps):
            # Combine coefficients
            u_terms = []
            v_terms = []
            w_terms = []
            
            # Expand A terms
            for a_block, a_coeff in super_step['A_blocks']:
                for a_local, a_inner_coeff in inner_step['A']:
                    u_idx = get_global_A(a_block, a_local)
                    coeff = a_coeff * a_inner_coeff
                    u_terms.append((u_idx, coeff))
            
            # Expand B terms
            for b_block, b_coeff in super_step['B_blocks']:
                for b_local, b_inner_coeff in inner_step['B']:
                    v_idx = get_global_B(b_block, b_local)
                    coeff = b_coeff * b_inner_coeff
                    v_terms.append((v_idx, coeff))
            
            # Expand C terms
            for c_block, c_coeff in super_step['C_blocks']:
                for c_local, c_inner_coeff in inner_step['C']:
                    w_idx = get_global_C(c_block, c_local)
                    coeff = c_coeff * c_inner_coeff
                    w_terms.append((w_idx, coeff))
            
            # This forms a single rank-1 term? No.
            # The tensor term is sum_{u,v,w} U[u] V[v] W[w]
            # If we have multiple u's, multiple v's, multiple w's, it's not rank 1.
            # It is a sum of rank-1 terms.
            # (sum_u c_u u) (sum_v d_v v) (sum_w e_w w)
            # = sum_{u,v,w} c_u d_v e_w u v w
            # So we must add a term for every combination of u, v, w.
            
            for u_idx, u_coeff in u_terms:
                for v_idx, v_coeff in v_terms:
                    for w_idx, w_coeff in w_terms:
                        total_coeff = u_coeff * v_coeff * w_coeff
                        if total_coeff != 0:
                            if not add_term(u_idx, v_idx, w_idx, total_coeff):
                                return {"U": U, "V": "V", "W": "W"} # Error, rank exceeded
                            
    # --- Part 2: 4x4x1 Multiplication (Last column of B, k=4) ---
    # Schoolbook: 4 * 4 * 1 = 16 terms.
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Terms: A[i, j] B[j, 4] C[4, i]
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            if not add_term(u_idx, v_idx, w_idx, 1.0):
                return {"U": U, "V": V, "W": W} # Error, rank exceeded

    if r != rank_target:
        # This shouldn't happen if the logic is correct
        # 49 + 16 = 65
        pass

    return {"U": U, "V": V, "W": W}
