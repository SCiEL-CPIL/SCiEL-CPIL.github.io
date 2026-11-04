"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find the minimal rank r such that this holds for all i,j,k.
    
    # The schoolbook rank is n*m*p = 2*3*9 = 54.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition.
        U, V, W are numpy arrays or lists.
        Returns True if valid, False otherwise.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def try_kronecker_decomposition():
        """
        Strategy: Model the 2x3 by 3x9 multiplication as a Kronecker product 
        of the 2x3 by 3x3 tensor and the identity tensor on the 3-dimensional 
        column-block space.
        
        We know that 2x3x3 matrix multiplication has a rank-5 decomposition 
        (derived from 3x3x3 rank 5).
        If we can express the 2x3x9 tensor as a sum of 3 blocks, each being 
        a 2x3x3 tensor tensored with a unit vector in the 3rd dimension, 
        we might get rank 5*3 = 15.
        
        Specifically:
        T_{i,j,k} = 1 for all i in [0,1], j in [0,2], k in [0,8].
        
        Let k = 3*q + r where q in [0,2] and r in [0,2].
        We want to construct U, V, W of rank 15.
        Let the rank index c be 5*q' + r' where q' in [0,2] and r' in [0,4].
        
        Hypothesis:
        U[i*m+j, 5*q' + r'] = U_small[i*m+j, r'] * delta_{q', ?}
        This is tricky because the "block" structure of the 2x3x9 product 
        is not simply a Kronecker product of the 2x3x3 product with I_3 
        in the standard sense of tensor rank. 
        The tensor T is just all 1s.
        T_{i,j,k} = 1.
        
        Let's try a direct construction for rank 15.
        We have 15 columns.
        Let's assign 5 columns to each "third" of k? No, k is the output index of B.
        
        Actually, let's look at the structure:
        C_{i,k} = sum_j A_{i,j} B_{j,k}.
        
        If we split k into 3 groups of 3: k_1, k_2, k_3.
        C_{i, 3*q+r} = sum_j A_{i,j} B_{j, 3*q+r}.
        
        This looks like 3 separate 2x3x3 multiplications if B was block-diagonal, 
        but B is a full 3x9 matrix.
        
        However, the tensor is T_{i,j,k} = 1.
        We need sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1.
        
        Let's try to find a rank 15 solution directly using the structure 
        that might exist.
        
        Alternative: Use the known rank 5 for 2x3x3.
        Let U5, V5, W5 be a rank 5 decomposition of 2x3x3.
        i.e., sum_{c=0}^4 U5_{i,j,c} V5_{j,k,c} W5_{k,i,c} = 1 for i in 0..1, j in 0..2, k in 0..2.
        
        Can we use this to build a 2x3x9 decomposition?
        We need to handle k in 0..8.
        Let's map k -> (q, r) where q = k//3, r = k%3.
        
        If we set:
        U_{i,j, 5q+c} = U5_{i,j,c}
        V_{j,k, 5q+c} = V5_{j,r,c} * delta_{q, q_k}? No, V depends on j,k.
        W_{k,i, 5q+c} = W5_{r,i,c} * delta_{q, q_k}?
        
        If we do this, for a fixed k (with q_k, r_k):
        sum_{c=0}^4 U_{i,j, 5q_k+c} V_{j,k, 5q_k+c} W_{k,i, 5q_k+c}
        = sum_{c=0}^4 U5_{i,j,c} V5_{j,r_k,c} W5_{r_k,i,c}
        = 1.
        
        This works IF we can define V and W such that they are zero for the "wrong" q.
        i.e., V_{j, 3q+r, 5q'+c} = V5_{j,r,c} if q=q', else 0.
        And W_{3q+r, i, 5q'+c} = W5_{r,i,c} if q=q', else 0.
        And U_{i,j, 5q'+c} = U5_{i,j,c} for all q'?
        Wait, U is summed over ALL c.
        sum_{c=0}^{14} U_{i,j,c} V_{j,k,c} W_{k,i,c}
        = sum_{q'=0}^2 sum_{c=0}^4 U_{i,j, 5q'+c} V_{j,k, 5q'+c} W_{k,i, 5q'+c}
        
        If U_{i,j, 5q'+c} = U5_{i,j,c} for all q',
        and V_{j,k, 5q'+c} = V5_{j,r,c} if q'=q_k else 0,
        and W_{k,i, 5q'+c} = W5_{r,i,c} if q'=q_k else 0,
        
        Then the term is non-zero only when q'=q_k.
        Sum = sum_{c=0}^4 U5_{i,j,c} V5_{j,r_k,c} W5_{r_k,i,c} = 1.
        
        So this constructs a valid rank 15 decomposition!
        We just need a rank 5 decomposition of 2x3x3.
        """
        if time.time() > deadline - 5:
            return None

        # Find a rank 5 decomposition of 2x3x3.
        # We can try to find one using random search + snap, or use a known one.
        # Since 2x3x3 is small, let's try to find one quickly.
        
        # Try random search for 2x3x3 rank 5
        # U: (6, 5), V: (9, 5), W: (6, 5) -- wait, dims for 2x3x3 are n=2, m=3, p=3.
        # U: (2*3, 5) = (6,5)
        # V: (3*3, 5) = (9,5)
        # W: (3*2, 5) = (6,5)
        
        n_s, m_s, p_s = 2, 3, 3
        rank_s = 5
        
        found_small = None
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            U_s = np.random.randn(n_s*m_s, rank_s)
            V_s = np.random.randn(m_s*p_s, rank_s)
            W_s = np.random.randn(p_s*n_s, rank_s)
            
            # Optimize? Or just check if random works? No, random won't work.
            # We need to solve it.
            # Let's use the alternating least squares method for 2x3x3.
            
            for it in range(20):
                # Solve for U
                U_new = np.zeros((n_s*m_s, rank_s))
                for i in range(n_s):
                    for j in range(m_s):
                        A = np.zeros(rank_s)
                        for k in range(p_s):
                            A += V_s[j*p_s+k, :] * W_s[k*n_s+i, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            U_new[i*m_s+j, :] = A / norm_sq
                U_s = U_new
                
                # Solve for V
                V_new = np.zeros((m_s*p_s, rank_s))
                for j in range(m_s):
                    for k in range(p_s):
                        A = np.zeros(rank_s)
                        for i in range(n_s):
                            A += U_s[i*m_s+j, :] * W_s[k*n_s+i, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            V_new[j*p_s+k, :] = A / norm_sq
                V_s = V_new
                
                # Solve for W
                W_new = np.zeros((p_s*n_s, rank_s))
                for k in range(p_s):
                    for i in range(n_s):
                        A = np.zeros(rank_s)
                        for j in range(m_s):
                            A += U_s[i*m_s+j, :] * V_s[j*p_s+k, :]
                        norm_sq = np.sum(A*A)
                        if norm_sq > 1e-9:
                            W_new[k*n_s+i, :] = A / norm_sq
                W_s = W_new
                
                # Check
                valid = True
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            val = np.dot(U_s[i*m_s+j, :] * V_s[j*p_s+k, :], W_s[k*n_s+i, :])
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Snap to half-integers
                    U_snap = snap_to_half_int(U_s.tolist())
                    V_snap = snap_to_half_int(V_s.tolist())
                    W_snap = snap_to_half_int(W_s.tolist())
                    
                    # Verify snapped
                    valid_snap = True
                    for i in range(n_s):
                        for j in range(m_s):
                            for k in range(p_s):
                                val = np.dot(np.array(U_snap[i*m_s+j]) * np.array(V_snap[j*p_s+k]), np.array(W_snap[k*n_s+i]))
                                if abs(val - 1.0) > 1e-9:
                                    valid_snap = False
                                    break
                            if not valid_snap: break
                        if not valid_snap: break
                    
                    if valid_snap:
                        found_small = (U_snap, V_snap, W_snap)
                        break
            if found_small:
                break
        
        if found_small is None:
            # If we couldn't find a snapped rank 5 for 2x3x3, we can't easily build rank 15 for 2x3x9.
            # Return None to let other methods try.
            return None
        
        U_s, V_s, W_s = found_small
        
        # Construct rank 15 decomposition for 2x3x9
        rank_big = 15
        U_big = [[0.0] * rank_big for _ in range(n*m)]
        V_big = [[0.0] * rank_big for _ in range(m*p)]
        W_big = [[0.0] * rank_big for _ in range(p*n)]
        
        # U_big[i*m+j, 5q+c] = U_s[i*m+j, c] for all q
        for i in range(n):
            for j in range(m):
                for q in range(3):
                    for c in range(5):
                        U_big[i*m+j][5*q + c] = U_s[i*m_s+j][c]
        
        # V_big[j*(3q+r), 5q'+c] = V_s[j*3+r, c] if q'=q else 0
        # Note: V_s index is j*p_s+k. Here p_s=3. So V_s[j*3+r, c].
        # V_big index is j*p+k. Here p=9. So k = 3q+r.
        for j in range(m):
            for q in range(3):
                for r in range(3):
                    k = 3*q + r
                    for c in range(5):
                        V_big[j*p + k][5*q + c] = V_s[j*p_s + r][c]
        
        # W_big[(3q+r)*n+i, 5q'+c] = W_s[r*n+i, c] if q'=q else 0
        # W_s index is k*n_s+i. Here k=r. So W_s[r*n_s+i, c].
        # W_big index is k*n+i. Here k=3q+r.
        for q in range(3):
            for r in range(3):
                for i in range(n):
                    k = 3*q + r
                    for c in range(5):
                        W_big[k*n + i][5*q + c] = W_s[r*p_s + i][c] # Wait, W_s is (p_s*n_s, rank). Index is k*n_s+i.
                        # W_s[r*2 + i, c]
        
        # Verify the big one
        if verify(U_big, V_big, W_big):
            return {"U": U_big, "V": V_big, "W": W_big}
        
        return None

    def try_random_search(rank, attempts=100):
        """
        Try to find a decomposition of given rank using random initialization
        and local optimization (ALS).
        """
        if time.time() > deadline - 5:
            return None
        
        best_loss = float('inf')
        best_params = None
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            U_mat = np.random.randn(n * m, rank)
            V_mat = np.random.randn(m * p, rank)
            W_mat = np.random.randn(p * n, rank)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for U
                U_mat = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros(rank)
                        for k in range(p):
                            A += V_mat[j * p + k, :] * W_mat[k * n + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            U_mat[i * m + j, :] = A / norm_sq
                
                # Solve for V
                V_mat_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros(rank)
                        for i in range(n):
                            A += U_mat[i * m + j, :] * W_mat[k * n + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            V_mat_new[j * p + k, :] = A / norm_sq
                V_mat = V_mat_new
                
                # Solve for W
                W_mat_new = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros(rank)
                        for j in range(m):
                            A += U_mat[i * m + j, :] * V_mat[j * p + k, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            W_mat_new[k * n + i, :] = A / norm_sq
                W_mat = W_mat_new
                
                # Check convergence
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U_mat[i * m + j, :]
                            v = V_mat[j * p + k, :]
                            w = W_mat[k * n + i, :]
                            val = np.dot(u * v, w)
                            loss += (val - 1.0) ** 2
                
                if loss < 1e-6:
                    U_snap = snap_to_half_int(U_mat.tolist())
                    V_snap = snap_to_half_int(V_mat.tolist())
                    W_snap = snap_to_half_int(W_mat.tolist())
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        return None

    # First, try the Kronecker-inspired rank 15 decomposition
    res = try_kronecker_decomposition()
    if res is not None:
        return res

    # Try ranks from lower bound upwards
    for rank in range(27, 55):
        if time.time() > deadline - 10:
            break
        
        attempts = max(5, 50 - (rank - 27) * 2)
        
        res = try_random_search(rank, attempts)
        if res is not None:
            return res

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
