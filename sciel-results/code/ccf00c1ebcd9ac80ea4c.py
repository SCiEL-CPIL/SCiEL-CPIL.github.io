import time
import random
import numpy as np
from scipy.optimize import least_squares

DEADLINE = 150.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    """
    Constructs a low-rank exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    Strategy: Partition the 3x9 matrix into three 3x3 blocks.
    Find a low-rank decomposition for 2x3 by 3x3 (rank 5 or 6).
    Replicate it 3 times.
    """
    n, m, p = 2, 3, 9
    
    # We want to solve for the 2x3x3 tensor decomposition.
    # Indices: i in 0..1 (rows of A), j in 0..2 (cols of A / rows of B), k in 0..2 (cols of B)
    # U: 6 x rank (index i*3+j)
    # V: 9 x rank (index j*3+k)
    # W: 6 x rank (index k*2+i)
    
    # Target: sum_r U[i*3+j, r] V[j'*3+k, r] W[k*2+i, r] = delta_{j,j'}
    
    def solve_rank(r, max_iters=20, seed=42):
        # Variables: U (6x r), V (9x r), W (6x r)
        # Flatten to vector of length 6r + 9r + 6r = 21r
        
        def residuals(params):
            U = params[0:6*r].reshape(6, r)
            V = params[6*r:15*r].reshape(9, r)
            W = params[15*r:21*r].reshape(6, r)
            
            res = []
            for i in range(2):
                for k in range(3):
                    for j in range(3):
                        for jp in range(3):
                            val = 0.0
                            for rr in range(r):
                                val += U[i*3+j, rr] * V[jp*3+k, rr] * W[k*2+i, rr]
                            target = 1.0 if j == jp else 0.0
                            res.append(val - target)
            return res
        
        best_sol = None
        best_err = float('inf')
        
        random.seed(seed)
        np.random.seed(seed)
        
        for attempt in range(max_iters):
            if time_remaining() < 10:
                break
                
            # Random initialization
            # Scale matters. Let's use small random values.
            p0 = np.random.uniform(-1, 1, size=21*r)
            
            # Use least squares
            try:
                result = least_squares(residuals, p0, method='lm', max_nfev=1000)
                err = result.cost
                if err < best_err:
                    best_err = err
                    best_sol = result.x
            except Exception:
                pass
        
        if best_err < 1e-6 and best_sol is not None:
            U = best_sol[0:6*r].reshape(6, r)
            V = best_sol[6*r:15*r].reshape(9, r)
            W = best_sol[15*r:21*r].reshape(6, r)
            return U, V, W
        return None

    # Try rank 5
    U5, V5, W5 = solve_rank(5, max_iters=10, seed=1)
    if U5 is not None:
        rank_block = 5
        U_block, V_block, W_block = U5, V5, W5
    else:
        # Try rank 6
        U6, V6, W6 = solve_rank(6, max_iters=10, seed=2)
        if U6 is not None:
            rank_block = 6
            U_block, V_block, W_block = U6, V6, W6
        else:
            # Fallback to schoolbook for 2x3x3 (rank 18)
            # This is safe but not optimal.
            # Actually, we can just use the full schoolbook for 2x3x9 directly.
            # But let's construct the block schoolbook.
            rank_block = 18
            U_block = np.zeros((6, 18))
            V_block = np.zeros((9, 18))
            W_block = np.zeros((6, 18))
            col = 0
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        U_block[i*3+j, col] = 1
                        V_block[j*3+k, col] = 1
                        W_block[k*2+i, col] = 1
                        col += 1

    # Now, replicate the block solution 3 times for the 9 columns.
    # The 3x9 matrix B is split into B1, B2, B3 (each 3x3).
    # C = [A B1, A B2, A B3].
    # The decomposition for each block is independent.
    # We just need to map the indices correctly.
    
    total_rank = rank_block * 3
    
    U = np.zeros((6, total_rank))
    V = np.zeros((27, total_rank))
    W = np.zeros((18, total_rank))
    
    for block in range(3):
        # Block b corresponds to columns 3*b, 3*b+1, 3*b+2 of B and C.
        # U is the same for all blocks (it only depends on A).
        # V indices: j*9 + k. For block b, k is 3*b + k_local.
        # W indices: k*2 + i. For block b, k is 3*b + k_local.
        
        for r in range(rank_block):
            col_idx = block * rank_block + r
            
            # U
            for i in range(2):
                for j in range(3):
                    U[i*3+j, col_idx] = U_block[i*3+j, r]
                    
            # V
            for j in range(3):
                for k_local in range(3):
                    k_global = 3 * block + k_local
                    V[j*9 + k_global, col_idx] = V_block[j*3 + k_local, r]
                    
            # W
            for k_local in range(3):
                for i in range(2):
                    k_global = 3 * block + k_local
                    W[k_global*2 + i, col_idx] = W_block[k_local*2 + i, r]

    # Convert to lists of lists (integers or floats)
    # The evaluator accepts integer/half-integer. The solver produces floats.
    # We should check if they are close to integers or halves.
    # If not, we might need to round or use a different method.
    # However, the tensor decomposition for matmul usually has integer solutions.
    # The numerical solver might produce 0.999999.
    
    def clean_matrix(M):
        M_clean = []
        for row in M:
            new_row = []
            for val in row:
                if abs(val) < 1e-9:
                    new_row.append(0)
                else:
                    # Try to round to nearest 0.5
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        new_row.append(rounded)
                    else:
                        new_row.append(val)
            M_clean.append(new_row)
        return M_clean

    U_list = clean_matrix(U)
    V_list = clean_matrix(V)
    W_list = clean_matrix(W)

    return {"U": U_list, "V": V_list, "W": W_list}
