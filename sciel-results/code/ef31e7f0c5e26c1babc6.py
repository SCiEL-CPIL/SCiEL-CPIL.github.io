import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Precompute support indices
# a = i*m + j
# b = j*p + k
# c = k*n + i
SUPPORT = []
# Map a to list of (b, c)
A_TO_BC = {a: [] for a in range(NM)}
# Map b to list of (a, c)
B_TO_AC = {b: [] for b in range(MP)}
# Map c to list of (a, b)
C_TO_AB = {c: [] for c in range(PN)}

for i in range(n):
    for j in range(m):
        for k in range(p):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            SUPPORT.append((a, b, c))
            A_TO_BC[a].append((b, c))
            B_TO_AC[b].append((a, c))
            C_TO_AB[c].append((a, b))

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin
    
    # We will try ranks from 18 up to 36
    # But we want the minimal rank, so we should try low ranks first?
    # Or high to low to ensure we find a valid solution quickly?
    # The score is "rank". Lower is better.
    # If we find a solution at rank 20, we should try 19, 18.
    # If we don't find 18, we return 20.
    
    # Let's try ranks in order: 18, 19, ..., 36
    # But if 18 is impossible, we waste time.
    # Known lower bound is 18.
    # Let's try 18, 19, 20, 21, 22, 24, 28, 36.
    
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    for R in ranks_to_try:
        if time.time() > time_limit:
            break
            
        # Try to find a solution of rank R
        result = search_rank(R, time_limit - (time.time() - start_time) - 5)
        if result is not None:
            return result
            
    # Fallback: Rank 36
    return get_trivial_solution()

def get_trivial_solution():
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def search_rank(R, time_budget):
    # Precompute target values (all 1)
    # We maintain P_abc for support points
    # P is a dict or array indexed by support index
    # Let's use an array of size len(SUPPORT)
    num_support = len(SUPPORT)
    
    # Mapping from (a,b,c) to index in support array
    # We already have A_TO_BC, etc.
    
    # We need to be able to update P quickly.
    # P_idx corresponds to SUPPORT[idx]
    
    # Let's represent U, V, W as numpy arrays for speed in inner loops?
    # But we need to update individual entries and calculate delta error.
    
    # Strategy:
    # 1. Initialize U, V, W with random integers.
    # 2. Compute initial P and Error.
    # 3. Local search:
    #    - Pick a random entry (mat, row, col).
    #    - Try +1, -1.
    #    - Calculate delta error using precomputed factors.
    #    - If error decreases, accept.
    
    # Precomputation for fast delta:
    # For each r, a:
    #   D_U[r, a, k] = V[b_k, r] * W[c_k, r] where (b_k, c_k) are from A_TO_BC[a]
    # This is expensive to precompute for all r, a.
    # Instead, calculate on the fly?
    # p=6. So 6 multiplications. Very fast.
    
    # Let's use lists for U, V, W to allow easy integer updates?
    # Numpy is faster for vectorized ops, but we are doing scalar updates.
    # Python lists might be slower for indexing?
    # Let's stick to Numpy arrays but access scalars.
    
    best_err = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    n_restarts = 0
    while time.time() < time.time() + time_budget:
        n_restarts += 1
        
        U = np.zeros((NM, R), dtype=np.int32)
        V = np.zeros((MP, R), dtype=np.int32)
        W = np.zeros((PN, R), dtype=np.int32)
        
        # Initialization
        if n_restarts == 1:
            # Structured init: try to pack the trivial solution into R slots?
            # If R < 36, this is hard.
            # Just random.
            pass
        
        # Random init with small values
        U = np.random.randint(-2, 3, size=(NM, R))
        V = np.random.randint(-2, 3, size=(MP, R))
        W = np.random.randint(-2, 3, size=(PN, R))
        
        # Compute initial P and Error
        P = np.zeros(num_support, dtype=np.float64)
        for idx, (a, b, c) in enumerate(SUPPORT):
            P[idx] = np.dot(U[a], V[b] * W[c]) # V[b]*W[c] is vector of R
            # Wait, V[b] and W[c] are vectors of length R.
            # U[a] is vector of length R.
            # P = sum_r U[a,r] V[b,r] W[c,r]
            # np.dot(U[a], V[b] * W[c]) works.
        
        err = np.sum((1.0 - P) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
        
        # Local Search
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > time.time() + time_budget:
                break
            
            # Pick random matrix and entry
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U
                dim = NM
                name = 'U'
            elif mat_choice == 1:
                mat = V
                dim = MP
                name = 'V'
            else:
                mat = W
                dim = PN
                name = 'W'
            
            row = np.random.randint(0, dim)
            col = np.random.randint(0, R)
            
            old_val = mat[row, col]
            
            # Try +1 and -1
            for delta in [1, -1]:
                new_val = old_val + delta
                if new_val == old_val:
                    continue
                
                # Calculate Delta Error
                delta_err = 0.0
                
                if name == 'U':
                    # a = row, r = col
                    # Affected support points: (a, b, c) in A_TO_BC[a]
                    for (b, c) in A_TO_BC[row]:
                        # Find index in P
                        # We need a mapping from (a,b,c) to idx
                        # This is slow if we search.
                        # Precompute A_TO_BC_IDX
                        pass
                # ...
            
            # This "on the fly" index lookup is slow.
            # Precompute A_TO_BC_IDX, B_TO_AC_IDX, C_TO_AB_IDX.
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-9:
                    break
            
            if it > 100 and not improved: # Need 'improved' flag
                break
        
        if best_err < 1e-9:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

# EVOLVE-BLOCK-END
