import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    def get_best_2x2():
        """Returns rank and terms for 2x2 multiplication."""
        # Strassen is rank 7, Schoolbook is rank 4. 
        # Wait, Strassen is 7 multiplications. Schoolbook is 4.
        # For 2x2, Schoolbook (rank 4) is actually better than Strassen (rank 7) in terms of total rank count?
        # No, Strassen reduces 8 to 7. Schoolbook is 4.
        # So for 2x2 * 2x2, the minimal rank is 4 (Schoolbook). Strassen is only useful for larger matrices where the overhead is amortized or when combined with other blocks?
        # Actually, for 2x2, rank 4 is the theoretical minimum (and achievable).
        # So we should always use Schoolbook for 2x2 blocks if we are counting total rank.
        # Let's re-verify. 
        # C = AB. C_ij = sum_k A_ik B_kj.
        # For 2x2, there are 4 entries in C. Each is a dot product of length 2.
        # Schoolbook: 4 * 2 = 8 scalar multiplications? No, rank of tensor.
        # Rank of 2x2x2 multiplication tensor is 7? No, it's 4?
        # Let's check the seed code. 
        # Seed code uses Strassen for 2x2 and claims rank 7.
        # But schoolbook for 2x2 is rank 4? 
        # Let's trace: 
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        # This requires 8 scalar multiplications in the standard algorithm.
        # The rank of the tensor is the minimum number of rank-1 tensors to sum to the multiplication tensor.
        # For 2x2, the rank is 7? No, it is known to be 7? 
        # Actually, the rank of 2x2 matrix multiplication is 7? 
        # Wait, the schoolbook algorithm performs 8 multiplications. 
        # Strassen performs 7.
        # So the rank is at most 7.
        # Is it 4? No, 4 would imply C00, C01, C10, C11 can be computed with 4 multiplications total? 
        # No, each multiplication produces one scalar. We need 4 scalars. 
        # But the inputs are linear combinations.
        # The rank of the 2x2x2 tensor is 7. (Alon, 1983? No, it's a classic result).
        # Actually, for 2x2, the rank is 7? 
        # Let's check the seed code again. It uses Strassen for 2x2 and says "rank 7".
        # It uses Schoolbook for 2x3 * 3x2 etc.
        # Let's assume the rank of 2x2 * 2x2 is 7 (Strassen).
        # Schoolbook for 2x2 * 2x2 would be rank 8? 
        # Let's calculate schoolbook rank for 2x2.
        # 4 outputs, each sum of 2 products. 
        # If we do it naively:
        # m1 = A00 B00
        # m2 = A01 B10
        # m3 = A00 B01
        # m4 = A01 B11
        # m5 = A10 B00
        # m6 = A11 B10
        # m7 = A10 B01
        # m8 = A11 B11
        # That's 8 rank-1 terms.
        # Strassen reduces this to 7.
        # So for 2x2 blocks, Strassen (7) is better than Schoolbook (8).
        
        # What about 3x3?
        # Schoolbook rank for 3x3 is 27.
        # Winograd/Toom-Cook can do better? 
        # The rank of 3x3 matrix multiplication is not known exactly but is bounded.
        # However, for this task, we likely only have explicit constructions for 2x2 (Strassen) and maybe 3x3 if we implement a specific one.
        # The prompt mentions "Strassen-like decomposition".
        # Let's stick to what is easy to implement correctly: 
        # 1. Schoolbook for any block.
        # 2. Strassen for 2x2 * 2x2.
        
        # Are there other known small ranks?
        # 3x3 * 3x3: Rank is at most 23 (Winograd)? Or 21?
        # Implementing Winograd 3x3 is complex. 
        # Let's first implement a generic search over partitions using Schoolbook and Strassen(2x2).
        
        # If we find that 2+3 split with Strassen(2x2) gives 117, maybe a 3+2 split or 1+4 split is better?
        # Or 1+1+3?
        
        pass

    def schoolbook_rank(r, c, s):
        """Rank for r x c * c x s using schoolbook."""
        return r * c * s

    def strassen_2x2_rank():
        return 7

    def search_best_partition():
        """
        Search over all ways to partition rows/cols into blocks.
        Since n=5, partitions of 5 are:
        5
        4+1
        3+2
        3+1+1
        2+2+1
        2+1+1+1
        1+1+1+1+1
        
        We assume the partition for A (rows/cols), B (rows/cols), and C (rows/cols) is the same.
        i.e., we split the index set {0..4} into contiguous blocks.
        
        For each partition, we calculate the total rank by summing the ranks of all block multiplications.
        For each pair of blocks (A_block, B_block) that contribute to a C_block, we use the best known algorithm for that specific size.
        
        Known algorithms:
        - If A_block is 2x2 and B_block is 2x2 (so C is 2x2), use Strassen (rank 7).
        - Otherwise, use Schoolbook (rank r*c*s).
        
        Note: This assumes we don't mix different algorithms in a way that allows sharing of intermediate results across different block pairs.
        The "block decomposition" strategy in the seed code does exactly this: it treats each block multiplication independently.
        
        So the total rank is the sum of ranks of all block multiplications.
        """
        
        def get_partitions(n):
            """Generate all integer partitions of n into parts >= 1, preserving order (compositions)."""
            # We want compositions, e.g., 2+3 is different from 3+2.
            # Recursive generator
            def gen(rem, max_part):
                if rem == 0:
                    yield []
                    return
                for i in range(1, max_part + 1):
                    if i <= rem:
                        for rest in gen(rem - i, max_part):
                            yield [i] + rest
            
            # Compositions allow any order, so max_part is just rem
            return list(gen(n, n))
        
        best_rank = float('inf')
        best_partition = None
        
        # Partitions of 5
        parts = get_partitions(5)
        
        for part in parts:
            # part is a list like [2, 3]
            # Calculate total rank for this partition
            total_rank = 0
            
            # Identify block boundaries
            # A has blocks of sizes part[i] x part[j] ? No.
            # The partition applies to the dimensions.
            # If we partition the row indices of A and column indices of A into blocks defined by `part`,
            # and similarly for B and C.
            
            # Let's say the partition is `part`.
            # A is divided into blocks A[i][j] of size part[i] x part[j].
            # B is divided into blocks B[j][k] of size part[j] x part[k].
            # C is divided into blocks C[i][k] of size part[i] x part[k].
            
            # C[i][k] = sum_j A[i][j] * B[j][k]
            
            for i in range(len(part)):
                for j in range(len(part)):
                    for k in range(len(part)):
                        # A[i][j] is part[i] x part[j]
                        # B[j][k] is part[j] x part[k]
                        # Product contributes to C[i][k] which is part[i] x part[k]
                        
                        r = part[i]
                        c = part[j]
                        s = part[k]
                        
                        # Check if we can use Strassen
                        # Strassen applies to 2x2 * 2x2 -> 2x2
                        # Here A_block is r x c, B_block is c x s.
                        # For matrix multiplication to be square 2x2, we need r=2, c=2, s=2.
                        if r == 2 and c == 2 and s == 2:
                            total_rank += 7 # Strassen
                        else:
                            total_rank += r * c * s # Schoolbook
            
            if total_rank < best_rank:
                best_rank = total_rank
                best_partition = part
        
        return best_rank, best_partition

    # Run the search
    best_rank, best_part = search_best_partition()
    print(f"Best partition found: {best_part}, Rank: {best_rank}")
    
    # Now, we need to actually construct the U, V, W matrices for this partition.
    # The seed code did this for a specific partition (2+3).
    # We need a generic constructor.
    
    U = []
    V = []
    W = []
    
    rank_so_far = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal rank_so_far
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_indices.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_indices.items():
            v_col[j * p + k] = coeff
        for (k, i), coeff in w_indices.items():
            w_col[k * n + i] = coeff
        
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
        rank_so_far += 1

    def strassen_2x2_terms():
        """
        Returns a list of 7 rank-1 terms for 2x2 matrix multiplication.
        Indices are local (0,1).
        """
        terms = []
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2 = (A10 + A11)B00 -> C10
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
        # M3 = A00(B01 - B11) -> C01 + C11
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
        # M4 = A11(B10 - B00) -> C00 + C10
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
        # M5 = (A00 + A01)B11 -> -C00 + C01
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
        # M6 = (A10 - A11)(B01 + B11) -> C11
        terms.append(({(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(1,1):1}))
        # M7 = (A01 - A11)(B00 + B10) -> C00
        terms.append(({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(0,0):1}))
        return terms

    def add_block_term(u_row_offset, u_col_offset, v_row_offset, v_col_offset, 
                       w_row_offset, w_col_offset, u_local, v_local, w_local):
        """
        Add a term with local indices mapped to global indices.
        u_local: dict {(i,j): coeff} relative to u_row_offset, u_col_offset
        v_local: dict {(j,k): coeff} relative to v_row_offset, v_col_offset
        w_local: dict {(k,i): coeff} relative to w_row_offset, w_col_offset
        """
        u_global = {}
        v_global = {}
        w_global = {}
        
        for (i, j), coeff in u_local.items():
            gi = u_row_offset + i
            gj = u_col_offset + j
            u_global[(gi, gj)] = coeff
        
        for (j, k), coeff in v_local.items():
            gj = v_row_offset + j
            gk = v_col_offset + k
            v_global[(gj, gk)] = coeff
        
        for (k, i), coeff in w_local.items():
            gk = w_row_offset + k
            gi = w_col_offset + i
            w_global[(gk, gi)] = coeff
        
        add_term(u_global, v_global, w_global)

    # Build the decomposition for the best partition
    part = best_part
    
    # Calculate offsets
    # offsets[i] is the starting index for block i
    offsets = [0]
    for i in range(len(part) - 1):
        offsets.append(offsets[-1] + part[i])
    
    # Iterate over all block triplets (i, j, k)
    for i in range(len(part)):
        for j in range(len(part)):
            for k in range(len(part)):
                r = part[i] # Rows of A_block, C_block
                c = part[j] # Cols of A_block, Rows of B_block
                s = part[k] # Cols of B_block, Cols of C_block
                
                # Offsets
                u_r_off = offsets[i]
                u_c_off = offsets[j]
                v_r_off = offsets[j]
                v_c_off = offsets[k]
                w_r_off = offsets[k] # W is indexed by [k*n + i], so row is k, col is i? 
                                     # Wait, W shape is (p*n, rank). Index is k*n + i.
                                     # So "row" in W matrix corresponds to k, "col" to i.
                                     # In our add_block_term, w_local keys are (k, i).
                                     # So w_row_offset should be offsets[k] and w_col_offset offsets[i].
                w_c_off = offsets[i]
                
                if r == 2 and c == 2 and s == 2:
                    # Use Strassen
                    for u_l, v_l, w_l in strassen_2x2_terms():
                        add_block_term(u_r_off, u_c_off, v_r_off, v_c_off, w_r_off, w_c_off, u_l, v_l, w_l)
                else:
                    # Use Schoolbook
                    for i_l in range(r):
                        for j_l in range(c):
                            for k_l in range(s):
                                add_block_term(
                                    u_r_off, u_c_off, 
                                    v_r_off, v_c_off, 
                                    w_r_off, w_c_off,
                                    {(i_l, j_l): 1},
                                    {(j_l, k_l): 1},
                                    {(k_l, i_l): 1}
                                )
    
    rank = rank_so_far
    print(f"Constructed rank: {rank}")
    
    # Verification
    np.random.seed(42)
    valid = True
    for _ in range(5):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        C_calc = np.zeros((n, p))
        for r_idx in range(rank):
            u_col = U[r_idx]
            v_col = V[r_idx]
            w_col = W[r_idx]
            
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += u_col[i * m + j] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += v_col[j * p + k] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_calc[i, k] += alpha[i] * beta[k] * w_col[k * n + i]
        
        if not np.allclose(C_calc, C_expected, atol=1e-10):
            print(f"Verification failed! Rank={rank}")
            valid = False
            break
    
    if valid:
        print(f"Verification passed. Returning rank {rank}.")
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        print("Falling back to schoolbook rank 125.")
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
