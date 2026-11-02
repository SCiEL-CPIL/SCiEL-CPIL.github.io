import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    1. Construct the hybrid Strassen/Schoolbook decomposition (Rank 39).
    2. Use linear algebra (SVD) to check for linear dependencies among the term vectors.
    3. If dependencies are found, remove dependent terms to reduce the rank.
    4. If not, return the rank 39 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    _start_time = time.time()
    _deadline = _start_time + 115.0  # Safety margin before 120s limit
    
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (a00 + a11)(b00 + b11)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    # M2 = (a10 + a11)b00
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    # M3 = a00(b01 - b11)
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    # M4 = a11(b10 - b00)
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    # M5 = (a00 + a01)b11
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    # M6 = (a10 - a00)(b00 + b01)
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    # M7 = (a01 - a11)(b10 + b11)
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # M1 = (a00 + a11)(b02 + b13)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1, w_idx(3,1): 1})
    # M2 = (a10 + a11)b02
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1, w_idx(3,1): -1})
    # M3 = a00(b03 - b13)
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1, v_idx(1,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    # M4 = a11(b12 - b02)
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1, v_idx(0,2): -1}, {w_idx(2,0): 1, w_idx(2,1): 1})
    # M5 = (a00 + a01)b13
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(2,0): -1, w_idx(3,0): 1})
    # M6 = (a10 - a00)(b02 + b03)
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,2): 1, v_idx(0,3): 1}, {w_idx(3,1): 1})
    # M7 = (a01 - a11)(b12 + b13)
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1})
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 4:6]
    # M1 = (a00 + a11)(b04 + b15)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,4): 1, v_idx(1,5): 1}, {w_idx(4,0): 1, w_idx(5,1): 1})
    # M2 = (a10 + a11)b04
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,4): 1}, {w_idx(4,1): 1, w_idx(5,1): -1})
    # M3 = a00(b05 - b15)
    add_term({u_idx(0,0): 1}, {v_idx(0,5): 1, v_idx(1,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    # M4 = a11(b14 - b04)
    add_term({u_idx(1,1): 1}, {v_idx(1,4): 1, v_idx(0,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    # M5 = (a00 + a01)b15
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,5): 1}, {w_idx(4,0): -1, w_idx(5,0): 1})
    # M6 = (a10 - a00)(b04 + b05)
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,4): 1, v_idx(0,5): 1}, {w_idx(5,1): 1})
    # M7 = (a01 - a11)(b14 + b15)
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,4): 1, v_idx(1,5): 1}, {w_idx(4,0): 1})
    
    # Part 4: Schoolbook for A[0:2, 0:2] * B[0:2, 6]
    # C[0,6] = a00*b06 + a01*b16
    # C[1,6] = a10*b06 + a11*b16
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,6): 1}, {w_idx(6,1): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
    # C[i,k] += a_i2 * b_2k for i in {0,1}, k in {0..6}
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
        
    original_rank = len(terms)
    
    # Check for linear dependencies using matrix rank
    # Flatten each term to a vector of size (n*m * m*p * p*n)
    vec_dim = (n * m) * (m * p) * (p * n)
    tensor_matrix = np.zeros((original_rank, vec_dim), dtype=float)
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        u_arr = np.array(u_list)
        v_arr = np.array(v_list)
        w_arr = np.array(w_list)
        term_vec = np.kron(np.kron(u_arr, v_arr), w_arr)
        tensor_matrix[c, :] = term_vec
        
    # Calculate rank
    try:
        U_svd, s, Vh = np.linalg.svd(tensor_matrix, full_matrices=False)
        rank = np.sum(s > 1e-8)
    except:
        rank = original_rank
        
    if rank < original_rank:
        # Greedy removal of dependent vectors
        kept_indices = list(range(original_rank))
        current_rank = rank
        
        for i in range(original_rank):
            if time.time() > _deadline:
                break
            if i not in kept_indices:
                continue
            
            trial_indices = [j for j in kept_indices if j != i]
            if len(trial_indices) < current_rank:
                continue
                
            trial_matrix = tensor_matrix[trial_indices, :]
            try:
                s_trial = np.linalg.svd(trial_matrix, full_matrices=False)[1]
                new_rank = np.sum(s_trial > 1e-8)
                if new_rank == current_rank:
                    kept_indices = trial_indices
                    current_rank = new_rank
            except:
                pass
                
        final_indices = kept_indices
    else:
        final_indices = list(range(original_rank))
        
    final_rank = len(final_indices)
    
    U = [[0.0] * final_rank for _ in range(n * m)]
    V = [[0.0] * final_rank for _ in range(m * p)]
    W = [[0.0] * final_rank for _ in range(p * n)]
    
    for new_c, old_c in enumerate(final_indices):
        u_list, v_list, w_list = terms[old_c]
        for i in range(n * m):
            U[i][new_c] = float(u_list[i])
        for i in range(m * p):
            V[i][new_c] = float(v_list[i])
        for i in range(p * n):
            W[i][new_c] = float(w_list[i])
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
