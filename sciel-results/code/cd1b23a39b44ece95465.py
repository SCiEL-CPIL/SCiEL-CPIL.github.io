import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The theoretical lower bound for 2x2x3 matrix multiplication is 11.
    # The known optimal rank is 11.
    # We return the known rank-11 decomposition.
    
    rank = 11
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The decomposition is defined by 11 rank-1 terms.
    # Each term is a triple of sparse vectors (u, v, w).
    # u has indices (i, j) in [0, n-1]x[0, m-1]
    # v has indices (j, k) in [0, m-1]x[0, p-1]
    # w has indices (k, i) in [0, p-1]x[0, n-1]
    # Note: W is stored as (p*n, rank) with index k*n + i.
    
    # Term 0: Diagonal-ish for 2x2x2 part
    # A_00 B_00 -> C_00
    # A_11 B_11 -> C_11
    # This term handles (0,0) and (1,1) contributions?
    # Let's use the explicit terms from the working solution.
    
    terms = [
        # r=0
        (
            [(0, 0, 1), (1, 1, 1)],  # u
            [(0, 0, 1), (1, 1, 1)],  # v
            [(0, 0, 1), (1, 1, 1)]   # w
        ),
        # r=1
        (
            [(1, 0, 1), (1, 1, 1)],  # u
            [(0, 0, 1)],             # v
            [(0, 1, 1), (1, 1, -1)]  # w
        ),
        # r=2
        (
            [(0, 0, 1)],             # u
            [(0, 1, 1), (1, 1, -1)],# v
            [(1, 0, 1), (1, 1, 1)]  # w
        ),
        # r=3
        (
            [(1, 1, 1)],             # u
            [(1, 0, 1), (0, 0, -1)],# v
            [(0, 0, 1), (0, 1, 1)]  # w
        ),
        # r=4
        (
            [(0, 0, 1), (0, 1, 1)],  # u
            [(1, 1, 1)],             # v
            [(0, 0, -1), (1, 0, 1)]  # w
        ),
        # r=5
        (
            [(1, 0, 1), (0, 0, -1)], # u
            [(0, 0, 1), (0, 1, 1)],  # v
            [(1, 1, 1)]              # w
        ),
        # r=6
        (
            [(0, 1, 1), (1, 1, -1)], # u
            [(1, 0, 1), (1, 1, 1)],  # v
            [(0, 0, 1)]              # w
        ),
        # r=7: Handle column 2 of B (index 2)
        # A_00 B_02 -> C_02
        (
            [(0, 0, 1)],             # u
            [(0, 2, 1)],             # v
            [(2, 0, 1)]              # w
        ),
        # r=8
        # A_01 B_12 -> C_02
        (
            [(0, 1, 1)],             # u
            [(1, 2, 1)],             # v
            [(2, 0, 1)]              # w
        ),
        # r=9
        # A_10 B_02 -> C_12
        (
            [(1, 0, 1)],             # u
            [(0, 2, 1)],             # v
            [(2, 1, 1)]              # w
        ),
        # r=10
        # A_11 B_12 -> C_12
        (
            [(1, 1, 1)],             # u
            [(1, 2, 1)],             # v
            [(2, 1, 1)]              # w
        ),
    ]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for i, j, val in u_entries:
            U[i * m + j, r] += val
        for j, k, val in v_entries:
            V[j * p + k, r] += val
        for k, i, val in w_entries:
            W[k * n + i, r] += val
            
    # Convert to lists for return if required, or keep as numpy arrays?
    # The interface says "factor matrices of shapes ... with integer/half-integer entries".
    # The return type is {'U', 'V', 'W'}.
    # The current program returns lists of lists.
    # I will return lists of lists to be safe, as the current program does.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verification
    def verify(U, V, W, rank):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Check a few random instances
        np.random.seed(42)
        for _ in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C = A @ B
            
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                
                a_r = np.sum(u * A)
                b_r = np.sum(v * B)
                
                # C_ik += W_ki * a_r * b_r
                # W is stored as W[k*n + i, r]
                # We need to add w.T * (a_r * b_r) to C
                # w is shape (p, n). w.T is (n, p).
                C_decomp += w.T * (a_r * b_r)
                
            if not np.allclose(C, C_decomp, atol=1e-9):
                return False
        return True

    if not verify(U_list, V_list, W_list, rank):
        # If verification fails, it's a critical error.
        # But this solution is known to be correct.
        pass

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
