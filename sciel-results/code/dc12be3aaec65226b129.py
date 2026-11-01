"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Uses the schoolbook decomposition which is guaranteed to be valid.
    Rank = n * m * p = 3 * 3 * 8 = 72.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i, j)
                U[i * m + j, idx] = 1
                # V is indexed by (j, k)
                V[j * p + k, idx] = 1
                # W is indexed by (k, i) following the trace convention W[k*n+i]
                W[k * n + i, idx] = 1
                idx += 1
    
    # Convert to lists of lists with integer entries
    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Constructed rank {len(result['U'][0])}")
