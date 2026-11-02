import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    # The problem decomposes into p=9 independent 2x3x1 multiplications.
    # We try to find a rank 4 decomposition for one 2x3x1 block.
    # If successful, total rank is 9 * 4 = 36.
    
    def solve_2x3x1(rank_target=4, n_restarts=50, max_iters=100):
        # Variables for one block:
        # A is 2x3 (indices 0,1 for i; 0,1,2 for j)
        # B is 3x1 (indices 0,1,2 for j)
        # C is 2x1 (indices 0,1 for i)
        # Tensor T_{i,j,k} = 1 if k=0.
        # We need U (2x3 x rank), V (3x1 x rank), W (2x1 x rank)
        # Actually, the indices are:
        # A indices: (i, j) -> 2*3 = 6
        # B indices: (j, k) -> 3*1 = 3 (k is fixed to 0 in this block)
        # C indices: (k, i) -> 1*2 = 2 (k is fixed to 0)
        
        # Wait, the global indices are:
        # A: i*m + j
        # B: j*p + k
        # C: k*n + i
        
        # For block k=0:
        # A indices: i*3 + j
        # B indices: j*9 + 0
        # C indices: 0*2 + i = i
        
        # We optimize U_block (6 x rank), V_block (3 x rank), W_block (2 x rank)
        
        best_err = float('inf')
        best = None
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            U = np.random.randn(6, rank_target) * 0.5
            V = np.random.randn(3, rank_target) * 0.5
            W = np.random.randn(2, rank_target) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                # Update U
                for a in range(6):
                    i = a // 3
                    j = a % 3
                    # Constraint: sum_r U[a,r] V[j,r] W[i,r] = 1
                    # A = V[j] * W[i]
                    A = np.outer(V[j], W[i]) # (rank, rank) -> need to be careful
                    # Actually, we need U[a] such that dot(U[a], V[j] * W[i]) = 1
                    # V[j] is (rank,), W[i] is (rank,).
                    # The term is sum_r U[a,r] V[j,r] W[i,r].
                    # This is U[a] dot (V[j] * W[i]).
                    target = 1.0
                    vec = V[j] * W[i]
                    if np.linalg.norm(vec) > 1e-9:
                        U[a] = vec / np.dot(vec, vec)
                    
                # Update V
                for b in range(3):
                    j = b
                    # Constraints for each i in 0,1
                    # sum_r U[i*3+j, r] V[j, r] W[i, r] = 1
                    # We have 2 equations for V[j]
                    # A = [U[i*3+j] * W[i] for i in 0,1]
                    # bvec = [1, 1]
                    A = np.array([U[i*3+j] * W[i] for i in range(2)])
                    bvec = np.ones(2)
                    V[j] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                    
                # Update W
                for c in range(2):
                    i = c
                    # Constraints for each j in 0,1,2
                    # sum_r U[i*3+j, r] V[j, r] W[i, r] = 1
                    A = np.array([U[i*3+j] * V[j] for j in range(3)])
                    bvec = np.ones(3)
                    W[i] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                
                # Compute error
                err = 0.0
                for i in range(2):
                    for j in range(3):
                        val = np.dot(U[i*3+j], V[j] * W[i])
                        err += (val - 1.0)**2
                        # Also check non-triples?
                        # In this block, only (i,j,0) are triples.
                        # Other combinations should be 0.
                        # But since we are solving for exact 1s, and the structure is sparse,
                        # we should check if other terms are 0.
                        # In the global tensor, other k are separate blocks.
                        # Within this block, are there other terms?
                        # The tensor is T_{i,j,k}.
                        # For k=0, T_{i,j,0} = 1.
                        # For k!=0, T_{i,j,k} = 0.
                        # In the block decomposition, we only care about k=0.
                        # The "other terms" in the global sense are handled by other blocks.
                        # Within this block, we only have 6 triples.
                        # Are there any (a,b,c) in this block that are NOT triples?
                        # a in 0..5, b in 0..2, c in 0..1.
                        # Triple if c=0 and a=i*3+j and b=j.
                        # If c=1, it's not a triple in this block?
                        # Wait, c is the index for W. W has size 2.
                        # W[0] corresponds to i=0, W[1] to i=1?
                        # No, W index is k*n + i.
                        # For k=0, W index is i.
                        # So c=0 -> i=0, c=1 -> i=1.
                        # So for a given a (i,j) and b (j), c is determined by i.
                        # So (a,b,c) is a triple if c == i.
                        # So for a=i*3+j, b=j, c must be i.
                        # If c != i, the term should be 0.
                        # So we need to check those too.
                        if c != i:
                            val = np.dot(U[a], V[b] * W[c])
                            err += val**2
                            
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                if err < 1e-12:
                    break
        
        if best and best_err < 1e-9:
            return best
        return None

    # Try to find rank 4 solution for one block
    res = solve_2x3x1(4, 100, 200)
    
    if res:
        U_block, V_block, W_block = res
        # Construct global matrices
        # Global U: 6 x 36
        # Global V: 27 x 36
        # Global W: 18 x 36
        
        rank = 36
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        
        for k in range(9):
            offset = k * 4
            # U is same for all k?
            # A indices are i*m+j. Independent of k.
            # So U columns for block k are U_block.
            U[:, offset:offset+4] = U_block
            
            # V indices are j*p+k.
            # For block k, V index is j*9+k.
            # V_block has 3 rows (j=0,1,2).
            for j in range(3):
                V[j*9+k, offset:offset+4] = V_block[j]
                
            # W indices are k*n+i.
            # For block k, W index is k*2+i.
            # W_block has 2 rows (i=0,1).
            for i in range(2):
                W[k*2+i, offset:offset+4] = W_block[i]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 54
    rank = 54
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(27)]
    W = [[0]*rank for _ in range(18)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(9):
                U[i*3+j][col] = 1
                V[j*9+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
