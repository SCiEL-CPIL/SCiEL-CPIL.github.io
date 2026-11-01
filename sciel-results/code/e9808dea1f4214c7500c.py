import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def check_solution(U, V, W):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > 1e-9:
                            return False
        return True

    # Schoolbook baseline
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 110.0

    # Structural approach: Use Strassen-style 2x2 blocks for the n=2 dimension
    # The 2x4x7 tensor can be viewed as two 1x4x7 blocks (one for each i).
    # For each i, we need to multiply a 4-vector by a 4x7 matrix.
    # A 1x4x7 multiplication has rank at least 7 (each output column needs one term).
    # But we can use Strassen-style tricks on the 4x4 sub-blocks.
    
    # Approach: Split the 4x7 into 2x(4x7) blocks using Strassen 2x2
    # For a 2x2 x 2x2 block, Strassen uses 7 mults instead of 8.
    # We have 4x7 = 2*(2x7) where each 2x7 can be seen as 2x(2+2+3)
    # Let's try: 4x7 as (2x2 + 2x2 + 2x3) blocks along the column dimension
    
    # Actually, let's use a more systematic approach:
    # Split 4 rows into 2+2, split 7 cols into 2+2+3
    # This gives 2x2, 2x2, 2x3, 2x2, 2x2, 2x3 blocks
    # For 2x2 blocks: Strassen (7 mults each)
    # For 2x3 blocks: need at least 6 mults (2*3), but can we do better?
    # 2x3 rank is 6 (no known improvement for 2x3)
    
    # Total: 2*(2x2) + 2*(2x3) per i, times 2 i's
    # = 2*(7 + 6 + 7 + 6) = 2*26 = 52 for the full 2x4x7
    
    # Let's build this structured decomposition:
    # For each i in {0,1}:
    #   Split 4 rows as [0,1] and [2,3]
    #   Split 7 cols as [0,1], [2,3], [4,5,6]
    #   Block (row_group, col_group):
    #     (0,0): 2x2 -> Strassen 7
    #     (0,1): 2x2 -> Strassen 7  
    #     (0,2): 2x3 -> schoolbook 6
    #     (1,0): 2x2 -> Strassen 7
    #     (1,1): 2x2 -> Strassen 7
    #     (1,2): 2x3 -> schoolbook 6
    # Total per i: 7+7+6+7+7+6 = 40
    # Total for 2 i's: 80. That's worse than schoolbook 56!
    
    # Let me reconsider. Schoolbook is 2*4*7 = 56.
    # The 1x4x7 subproblem has rank at least 4*7/4 = 7? No, it's exactly 4*7=28 for schoolbook.
    # But 1x4x7 is just a linear map from R^4 to R^7, which has rank 7 (the matrix rank).
    # Wait, the tensor T[i,j,j',k] for fixed i is T[0,j,j',k] = delta[j,j'] for all k.
    # That's not right. Let me re-read the problem.
    
    # The matrix multiplication tensor for 2x4 * 4x7:
    # Output C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,j',k] where we need sum_j T[i,j,j',k] = delta[j,j'] when contracted properly.
    # Actually the standard form: we want sum_c U[i,j,c] V[j',k,c] W[k,i,c] = delta[j,j']
    # where U has indices (i,j), V has (j',k), W has (k,i).
    
    # For the subtensor with i=0: we need sum_c U[0,j,c] V[j',k,c] W[k,0,c] = delta[j,j']
    # This is a tensor of shape (j, j', k) with the constraint that it equals delta[j,j'] for all k.
    # This means for each k, the matrix M_k[j,j'] = sum_c U[0,j,c] V[j',k,c] W[k,0,c] must be I_4.
    # The rank of this subtensor is at least... well, it's a 4x4x7 tensor that's "parallel" in k.
    # Each k-slice is a 4x4 identity, so the tensor is I_4 ⊗ e_k (in some sense).
    # The rank of I_m ⊗ (1...1) for m=4, length=7 is at least 4 (since each of the 4 diagonal entries must be 1).
    # Actually it's exactly 4 if we can make each term contribute to one diagonal entry for all k.
    # But the constraint is that the SAME c-terms must work for all k simultaneously.
    
    # Let me just try to find good structured decompositions by search.
    
    # Strategy: Try ranks from 50 down to 40 with structured random initialization
    # Use small integer values {-1, 0, 1} and try to find valid solutions
    
    np.random.seed(42)
    
    for rank in range(rank_school - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Use structured initialization: start from schoolbook and perturb
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Initialize with schoolbook pattern for first min(rank, n*m*p) columns
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m * p + j * p + k
                        if col < rank:
                            U[i*m + j, col] = 1.0
                            V[j*p + k, col] = 1.0
                            W[k*n + i, col] = 1.0
            
            # Add random perturbations to remaining columns
            for c in range(min(rank, n*m*p), rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
                V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                W[:, c] = np.random.choice([-1, 0, 1], size=p*n)
            
            if check_solution(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break
        
        # Also try pure random
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            if check_solution(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break

    # Try half-integer values too
    if time.time() < deadline:
        for rank in range(min(best_rank, rank_school), 39, -1):
            if time.time() > deadline:
                break
            
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
                
                if check_solution(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
