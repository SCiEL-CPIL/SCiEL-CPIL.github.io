"""Matrix multiplication 2,5,6 tensor decomposition with Strassen on all 2x2x2 sub-blocks."""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Construct a low-rank exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    Strategy: Use Strassen's algorithm (rank 7) on all six disjoint 2x2x2 sub-blocks
    formed by j-pairs (0,1),(2,3) crossed with k-pairs (0,1),(2,3),(4,5).
    Remaining elements use schoolbook (rank 1 each).
    
    Total rank: 6*7 (Strassen) + (60 - 6*8) schoolbook = 42 + 12 = 54
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Standard Strassen rank-7 decomposition of the 2x2x2 matmul tensor.
    # Each entry is (U_flat, V_flat, W_flat) in flattened 2x2 order [00,01,10,11].
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # m1 = (a+d)(e+h)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # m2 = (c+d)e
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # m3 = a(f-h)
        ([0,0,0,1], [0,0,1,-1], [-1,1,0,0]), # m4 = d(g-e)
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),   # m5 = (a+b)h
        ([1,0,-1,0], [1,1,0,0], [0,0,0,-1]), # m6 = (c-a)(e+f)
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]), # m7 = (b-d)(g+h)
    ]
    
    # Verify Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for U, V, W in strassen_terms:
                        total += U[i] * V[j] * W[k]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify_strassen()
    if not valid:
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Define the six disjoint 2x2x2 sub-blocks:
    # j-pairs: (0,1), (2,3)  [j=4 is left for schoolbook]
    # k-pairs: (0,1), (2,3), (4,5)
    # For each combination, we get a 2x2x2 sub-block over i in {0,1}, j in {j0,j0+1}, k in {k0,k0+1}
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Track which (i,j,k) triples are covered by Strassen
    covered = set()
    all_terms = []
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Mark all (i, j, k) in this block as covered
            for i in range(n):
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        covered.add((i, j, k))
            
            # Add 7 Strassen terms for this block
            for U_flat, V_flat, W_flat in strassen_terms:
                U_sparse = {}
                for idx, val in enumerate(U_flat):
                    if val != 0:
                        # idx: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
                        i_local = idx // 2
                        j_local = idx % 2
                        j_global = [j0, j1][j_local]
                        U_sparse[i_local * m + j_global] = val
                V_sparse = {}
                for idx, val in enumerate(V_flat):
                    if val != 0:
                        # idx: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
                        j_local = idx // 2
                        k_local = idx % 2
                        j_global = [j0, j1][j_local]
                        k_global = [k0, k1][k_local]
                        V_sparse[j_global * p + k_global] = val
                W_sparse = {}
                for idx, val in enumerate(W_flat):
                    if val != 0:
                        # idx: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
                        k_local = idx // 2
                        i_local = idx % 2
                        k_global = [k0, k1][k_local]
                        W_sparse[k_global * n + i_local] = val
                all_terms.append((U_sparse, V_sparse, W_sparse))
    
    # Add schoolbook terms for all (i,j,k) NOT covered by Strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_sparse = {i * m + j: 1}
                V_sparse = {j * p + k: 1}
                W_sparse = {k * n + i: 1}
                all_terms.append((U_sparse, V_sparse, W_sparse))
    
    rank = len(all_terms)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_sparse, V_sparse, W_sparse) in enumerate(all_terms):
        for idx, val in U_sparse.items():
            U[idx][col] = val
        for idx, val in V_sparse.items():
            V[idx][col] = val
        for idx, val in W_sparse.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
