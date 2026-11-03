"""Program for matrix multiplication tensor decomposition task."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: Split into two independent 3x3x3 multiplications.
    Search for a low-rank decomposition of the 3x3x3 multiplication tensor.
    If a rank-R decomposition is found for 3x3x3, the total rank is 2R.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 115  # Safety margin
    
    def snap_to_half(x):
        """Snap value to nearest 0.5 multiple."""
        return round(x * 2) / 2.0

    def snap_matrix(M):
        """Snap all entries in matrix to nearest 0.5 multiple."""
        return np.array([[snap_to_half(x) for x in row] for row in M])

    def verify_3x3x3(U, V, W, rank, tol=1e-9):
        """
        Verify if U, V, W (snapped) form an exact decomposition of 3x3x3 multiplication.
        U: (9, rank), V: (9, rank), W: (9, rank)
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check all 729 entries of the tensor
        # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # Target: 1 if (u,v,w) corresponds to (i,j,k) with u=3i+j, v=3j+k, w=3k+i
        # 0 otherwise
        
        T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = i*3 + j
                    v = j*3 + k
                    w = k*3 + i
                    if abs(T_current[u, v, w] - 1.0) > tol:
                        return False
        
        # Check zeros
        # We can just check all entries
        for u in range(9):
            for v in range(9):
                for w in range(9):
                    # Check if this is a valid (i,j,k)
                    # u = 3i + j => i = u//3, j = u%3
                    # v = 3j + k => j = v//3, k = v%3
                    # w = 3k + i => k = w//3, i = w%3
                    i1 = u // 3
                    j1 = u % 3
                    j2 = v // 3
                    k1 = v % 3
                    k2 = w // 3
                    i2 = w % 3
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        # This is a 1 position, already checked
                        continue
                    else:
                        if abs(T_current[u, v, w]) > tol:
                            return False
        return True

    def find_3x3x3_decomposition(rank, max_time=10):
        """
        Attempt to find a rank-R decomposition for 3x3x3 multiplication.
        Returns (U, V, W) if found, else None.
        """
        start_time = time.time()
        
        # Target tensor
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        rng = np.random.RandomState(rank * 100 + 42)
        
        # Try multiple restarts
        max_restarts = 100
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with small random values
            # Try integer/half-integer initialization
            scale = 1.0
            U = rng.randint(-1, 2, size=(9, rank)) * 0.5
            V = rng.randint(-1, 2, size=(9, rank)) * 0.5
            W = rng.randint(-1, 2, size=(9, rank)) * 0.5
            
            # Refine with gradient descent
            for step in range(50):
                if time.time() - start_time > max_time:
                    break
                
                T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
                Residual = T_current - T_target
                
                if np.max(np.abs(Residual)) < 1e-9:
                    break
                
                grad_U = np.einsum('uvw,vr,wr->ur', Residual, V, W)
                grad_V = np.einsum('uvw,ur,wr->vr', Residual, U, W)
                grad_W = np.einsum('uvw,ur,vr->wr', Residual, U, V)
                
                lr = 0.05
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Snap periodically
                if step % 5 == 4:
                    U = snap_matrix(U)
                    V = snap_matrix(V)
                    W = snap_matrix(W)
                    
                    if verify_3x3x3(U, V, W, rank):
                        return U.tolist(), V.tolist(), W.tolist()

            # Final check
            U = snap_matrix(U)
            V = snap_matrix(V)
            W = snap_matrix(W)
            if verify_3x3x3(U, V, W, rank):
                return U.tolist(), V.tolist(), W.tolist()
                
        return None

    def construct_full(U3, V3, W3, rank_3x3):
        """
        Construct full 3x3x6 decomposition from two 3x3x3 decompositions.
        """
        total_rank = 2 * rank_3x3
        
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(18)]
        W_full = [[0] * total_rank for _ in range(18)]
        
        # First 3x3 block (k=0,1,2)
        # U_full[i*3+j, r] = U3[i*3+j, r]
        # V_full[j*3+k, r] = V3[j*3+k, r] for k=0,1,2
        # W_full[k*3+i, r] = W3[k*3+i, r] for k=0,1,2
        
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r] = U3[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V_full[j*3+k][r] = V3[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_full[k*3+i][r] = W3[k*3+i][r]
                    
        # Second 3x3 block (k=3,4,5)
        # U_full[i*3+j, r+rank_3x3] = U3[i*3+j, r]
        # V_full[j*6+k, r+rank_3x3] = V3[j*3+(k-3), r] for k=3,4,5
        # W_full[k*3+i, r+rank_3x3] = W3[(k-3)*3+i, r] for k=3,4,5
        
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+rank_3x3] = U3[i*3+j][r]
            for j in range(3):
                for k in range(3, 6):
                    V_full[j*6+k][r+rank_3x3] = V3[j*3+(k-3)][r]
            for k in range(3, 6):
                for i in range(3):
                    W_full[k*3+i][r+rank_3x3] = W3[(k-3)*3+i][r]
                    
        return {"U": U_full, "V": V_full, "W": W_full}

    # Try ranks from low to high for 3x3x3
    # Best known is 23. Try 23, 24, 25.
    best_decomp = None
    best_rank = None
    
    for rank in [23, 24, 25]:
        if time.time() > deadline - 10:
            break
        decomp = find_3x3x3_decomposition(rank, max_time=5)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break # Found the lowest rank, stop
            
    if best_decomp is not None:
        U3, V3, W3 = best_decomp
        return construct_full(U3, V3, W3, best_rank)
    else:
        # Fallback: Schoolbook
        rank = 54
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        column = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i * 3 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
