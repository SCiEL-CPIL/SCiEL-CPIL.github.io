import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partition indices into blocks of sizes [2, 2, 1]
    # Offsets: 0, 2, 4
    # Sizes:   2, 2, 1
    blocks = [
        (0, 2),
        (2, 2),
        (4, 1)
    ]
    
    # Initialize U, V, W with zeros. We don't know the final rank yet,
    # so we can use a list of tuples or a large array.
    # Max possible rank: 125.
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank), dtype=np.int8)
    V = np.zeros((m * p, max_rank), dtype=np.int8)
    W = np.zeros((p * n, max_rank), dtype=np.int8)
    
    current_rank = 0
    
    # Iterate over all block combinations
    for bi, (off_n, size_n) in enumerate(blocks):
        for bj, (off_m, size_m) in enumerate(blocks):
            for bk, (off_p, size_p) in enumerate(blocks):
                if time.time() > deadline:
                    break
                
                # Determine the type of block
                if size_n == 2 and size_m == 2 and size_p == 2:
                    # Strassen block
                    rank_increment = 7
                    # Strassen coefficients
                    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
                    # M2 = (A10 + A11)B00        -> C10
                    # M3 = A00(B01 - B11)        -> C01
                    # M4 = A11(B10 - B00)        -> C00
                    # M5 = (A00 + A10)B11        -> C01, C00
                    # M6 = (A01 - A00)(B00 + B10)-> C11
                    # M7 = (A10 - A11)(B01 + B11)-> C00, C11
                    
                    # We define the contributions for each of the 7 products
                    # Format: (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w)
                    # u_idx is local (0-1), v_idx local (0-1), w_idx local (0-1)
                    # w_idx corresponds to C row/col? No, C is n x p.
                    # Local C indices: i_c in 0..1, k_c in 0..1.
                    # Global W index: (off_p + k_c)*n + (off_n + i_c)
                    
                    strassen_terms = [
                        # M1
                        [(0,0, 0,0, 0,0, 1, 1, 1), (1,1, 0,0, 0,0, 1, 1, 1), 
                         (0,0, 1,1, 1,1, 1, 1, 1), (1,1, 1,1, 1,1, 1, 1, 1)], # This is getting messy. 
                    ]
                    
                    # Let's use a cleaner representation.
                    # Each term r in 0..6 defines:
                    # U contributions: list of (local_i, local_j, coeff)
                    # V contributions: list of (local_j, local_k, coeff)
                    # W contributions: list of (local_k, local_i, coeff)
                    
                    terms = []
                    
                    # M1: (A00+A11)(B00+B11)
                    terms.append({
                        'U': [(0,0,1), (1,1,1)],
                        'V': [(0,0,1), (1,1,1)],
                        'W': [(0,0,1), (1,1,1)] # C00, C11
                    })
                    
                    # M2: (A10+A11)B00
                    terms.append({
                        'U': [(1,0,1), (1,1,1)],
                        'V': [(0,0,1)],
                        'W': [(0,1,1)] # C10 -> k=0, i=1
                    })
                    
                    # M3: A00(B01-B11)
                    terms.append({
                        'U': [(0,0,1)],
                        'V': [(0,1,1), (1,1,-1)],
                        'W': [(1,0,1)] # C01 -> k=1, i=0
                    })
                    
                    # M4: A11(B10-B00)
                    terms.append({
                        'U': [(1,1,1)],
                        'V': [(1,0,1), (0,0,-1)],
                        'W': [(0,0,1)] # C00
                    })
                    
                    # M5: (A00+A10)B11
                    terms.append({
                        'U': [(0,0,1), (1,0,1)],
                        'V': [(1,1,1)],
                        'W': [(1,0,1), (0,0,1)] # C01, C00
                    })
                    
                    # M6: (A01-A00)(B00+B10)
                    terms.append({
                        'U': [(0,1,1), (0,0,-1)],
                        'V': [(0,0,1), (1,0,1)],
                        'W': [(1,1,1)] # C11
                    })
                    
                    # M7: (A10-A11)(B01+B11)
                    terms.append({
                        'U': [(1,0,1), (1,1,-1)],
                        'V': [(0,1,1), (1,1,1)],
                        'W': [(0,0,1), (1,1,1)] # C00, C11
                    })
                    
                    for term in terms:
                        if current_rank >= max_rank:
                            break
                        r = current_rank
                        for ui, uj, cu in term['U']:
                            U[off_n * m + off_m + ui, r] = cu # Wait, U index is i*m + j
                            # Local i, j. Global i = off_n + ui, Global j = off_m + uj.
                            # U index = (off_n + ui) * m + (off_m + uj)
                            U[(off_n + ui) * m + (off_m + uj), r] = cu
                            
                        for vj, vk, cv in term['V']:
                            # V index = j * p + k
                            # Local j, k. Global j = off_m + vj, Global k = off_p + vk.
                            V[(off_m + vj) * p + (off_p + vk), r] = cv
                            
                        for wk, wi, cw in term['W']:
                            # W index = k * n + i
                            # Local k, i. Global k = off_p + wk, Global i = off_n + wi.
                            W[(off_p + wk) * n + (off_n + wi), r] = cw
                            
                        current_rank += 1

                else:
                    # Schoolbook block
                    rank_increment = size_n * size_m * size_p
                    for ui in range(size_n):
                        for uj in range(size_m):
                            for uk in range(size_p):
                                if time.time() > deadline:
                                    break
                                if current_rank >= max_rank:
                                    break
                                r = current_rank
                                # U
                                U[(off_n + ui) * m + (off_m + uj), r] = 1
                                # V
                                V[(off_m + uj) * p + (off_p + uk), r] = 1
                                # W
                                W[(off_p + uk) * n + (off_n + ui), r] = 1
                                current_rank += 1

    # Trim the matrices to the actual rank
    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
