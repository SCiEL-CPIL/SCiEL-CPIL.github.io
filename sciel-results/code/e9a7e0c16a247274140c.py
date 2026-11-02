import random
import numpy as np
import time

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Tensor T_{a,b,c,d,i,k} is 1 if a==i, b==d, c==k else 0
    # Decomposition: sum_l U_{a*m+b, l} V_{c*p+d, l} W_{k*n+i, l}
    
    # Vectorized check for speed
    # U: (nm, r), V: (mp, r), W: (pn, r)
    # We need to check for all a,b,c,d,i,k:
    # sum_l U[a*m+b, l] V[c*p+d, l] W[k*n+i, l] == (a==i and c==k and b==d)
    
    # We can check this by iterating over the non-zero targets (which are just the diagonal-like terms)
    # and checking that all other terms are 0.
    
    # Actually, for a small tensor, a simple loop is fast enough and less error-prone for verification logic.
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 6
    
    # Strategy:
    # M_{2,3,6} can be viewed as two independent M_{2,3,3} multiplications if we split the output columns.
    # However, we suspect M_{2,3,3} has rank 17 (from seed).
    # Total rank would be 34.
    # 
    # New Idea: Can we do better than 2 * Rank(M_{2,3,3})?
    # Let's look at M_{2,3,2}.
    # If we can find a rank-5 decomposition for M_{2,3,2}, we can potentially combine them?
    # No, M_{2,3,6} is not a direct sum of M_{2,3,2}s in a way that reduces rank below sum of ranks.
    # 
    # Wait, let's look at the structure of M_{2,3,2}.
    # n=2, m=3, p=2.
    # U: 6xR, V: 6xR, W: 4xR.
    # Schoolbook rank is 12.
    # Is there a rank 5?
    # Strassen for 2x2x2 is 7.
    # For 2x3x2, it's like multiplying a 2x3 by a 3x2.
    # This is equivalent to computing 6 dot products of length 3.
    # 
    # Let's try to construct a rank 17 solution for M_{2,3,6} directly.
    # How?
    # If we use the rank 17 solution for M_{2,3,3} (from seed), we get 34.
    # Is there a rank 17 solution for M_{2,3,6}?
    # That would imply Rank(M_{2,3,3}) <= 17 is not the bottleneck, but maybe we can share terms?
    # No, the blocks are disjoint in the output dimension W.
    # W has shape (12, R).
    # If the solution for block 1 (cols 0-2) uses W rows 0-5, and block 2 (cols 3-5) uses W rows 6-11,
    # they are disjoint.
    # However, U and V can potentially share columns if the patterns allow?
    # No, U and V are multiplied by W.
    # Term l contributes to U[:,l] * V[:,l] * W[:,l].
    # If W[:,l] is non-zero only in the first 6 rows, it only affects the first block of the output.
    # If W[:,l] is non-zero only in the last 6 rows, it only affects the second block.
    # If W[:,l] has non-zeros in both, it mixes the blocks.
    # 
    # So, a rank 17 solution for M_{2,3,6} would require mixing the blocks.
    # This is equivalent to finding a rank R decomposition of the tensor T such that T = T1 + T2 (disjoint supports in W)
    # is decomposed with rank R < Rank(T1) + Rank(T2).
    # This is possible if the "cross terms" can be cancelled or if the structure allows a more efficient decomposition.
    #
    # Let's try to find a rank 17 solution for M_{2,3,6} via a direct search,
    # initializing with the rank 34 solution and trying to merge/cancel terms?
    # Or, better: use the known rank 5 solution for M_{2,3,2} if it exists?
    # 
    # Let's check if M_{2,3,2} has rank 5.
    # If Rank(M_{2,3,2}) = 5, then we can compute M_{2,3,6} as three M_{2,3,2} blocks?
    # No, 6 is not divisible by 2 in a way that helps unless we overlap?
    # M_{2,3,6} = M_{2,3,2} \oplus M_{2,3,2} \oplus M_{2,3,2}?
    # If we split the 6 columns into three groups of 2.
    # Rank would be 3 * 5 = 15.
    # This is better than 17!
    # 
    # So the key is: Does M_{2,3,2} have a rank 5 decomposition?
    # Let's try to find one.
    # n=2, m=3, p=2.
    # U: (6, R), V: (6, R), W: (4, R).
    # 
    # We will perform a quick search for a rank 5 decomposition of M_{2,3,2}.
    # If found, we construct the rank 15 solution for M_{2,3,6}.
    # If not, we fall back to the rank 34 (or 17 if we find a better M_{2,3,3}) solution.
    
    def find_m232_rank(target_rank, time_budget):
        """
        Attempts to find a rank 'target_rank' decomposition for 2x3x2.
        """
        if time.time() > start_time + time_budget:
            return None
            
        n2, m2, p2 = 2, 3, 2
        U_dim = n2 * m2 # 6
        V_dim = m2 * p2 # 6
        W_dim = p2 * n2 # 4
        
        # We will try to find a solution with integer/half-integer entries.
        # Search space is small for R=5.
        # We can use a randomized local search with snapping.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Initialize with schoolbook (rank 12) and try to reduce?
        # Or just random restarts.
        
        for restart in range(200):
            if time.time() > start_time + time_budget:
                break
                
            # Random initialization
            U = np.random.randn(U_dim, target_rank) * 0.5
            V = np.random.randn(V_dim, target_rank) * 0.5
            W = np.random.randn(W_dim, target_rank) * 0.5
            
            # Simple gradient descent / SA
            lr = 0.1
            for step in range(100):
                # Calculate error
                err = 0.0
                for a in range(n2):
                    for b in range(m2):
                        for c in range(m2):
                            for d in range(p2):
                                for i in range(n2):
                                    for k in range(p2):
                                        target = 1.0 if (a == i and c == k and b == d) else 0.0
                                        s = np.dot(U[a * m2 + b, :], np.multiply(V[c * p2 + d, :], W[k * n2 + i, :]))
                                        err += (s - target) ** 2
                                        
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    if best_error < 1e-9:
                        # Try to snap to integers/half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        if verify_decomposition(U_snap, V_snap, W_snap, n2, m2, p2):
                            return U_snap, V_snap, W_snap
                            
                        # Try integers
                        U_snap = np.round(U)
                        V_snap = np.round(V)
                        W_snap = np.round(W)
                        if verify_decomposition(U_snap, V_snap, W_snap, n2, m2, p2):
                            return U_snap, V_snap, W_snap
                        
                # Perturb
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    dim = U_dim
                elif mat_idx == 1:
                    mat = V
                    dim = V_dim
                else:
                    mat = W
                    dim = W_dim
                    
                row = random.randint(0, dim - 1)
                col = random.randint(0, target_rank - 1)
                mat[row, col] += np.random.randn() * lr
                
        return None

    # Try to find rank 5 for M_{2,3,2}
    # Budget: 30 seconds
    m232_sol = find_m232_rank(5, 30)
    
    if m232_sol is not None:
        U232, V232, W232 = m232_sol
        r232 = U232.shape[1] # 5
        
        # Construct M_{2,3,6} from three copies of M_{2,3,2}
        # Split the 6 columns into three groups of 2: (0,1), (2,3), (4,5)
        # U shape: (6, 3*r232)
        # V shape: (18, 3*r232)
        # W shape: (12, 3*r232)
        
        total_rank = 3 * r232
        U_comb = np.zeros((6, total_rank))
        V_comb = np.zeros((18, total_rank))
        W_comb = np.zeros((12, total_rank))
        
        for block in range(3):
            # Block 0: cols 0,1. Block 1: cols 2,3. Block 2: cols 4,5.
            # V indices: c*p + d.
            # For block 0, d in {0,1}. V index = c*6 + d.
            # For block 1, d in {2,3}. V index = c*6 + d.
            # For block 2, d in {4,5}. V index = c*6 + d.
            
            # W indices: k*n + i.
            # For block 0, k in {0,1}. W index = k*2 + i.
            # For block 1, k in {2,3}. W index = k*2 + i.
            # For block 2, k in {4,5}. W index = k*2 + i.
            
            start_col = block * r232
            end_col = (block + 1) * r232
            
            # U is the same for all blocks
            U_comb[:, start_col:end_col] = U232
            
            # V mapping
            for c in range(3):
                for d in range(2):
                    # In M_{2,3,2}, d is 0 or 1.
                    # In M_{2,3,6}, d is block*2 + d_local.
                    v_idx_m232 = c * 2 + d
                    v_idx_m236 = c * 6 + (block * 2 + d)
                    V_comb[v_idx_m236, start_col:end_col] = V232[v_idx_m232]
                    
            # W mapping
            for k in range(2):
                for i in range(2):
                    # In M_{2,3,2}, k is 0 or 1.
                    # In M_{2,3,6}, k is block*2 + k_local.
                    w_idx_m232 = k * 2 + i
                    w_idx_m236 = (block * 2 + k) * 2 + i
                    W_comb[w_idx_m236, start_col:end_col] = W232[w_idx_m232]
                    
        # Verify full solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Fallback: Rank 34 solution (two copies of Rank 17 M_{2,3,3})
    # This is the verified solution from the current program.
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
