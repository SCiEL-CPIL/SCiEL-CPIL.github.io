import time
import numpy as np
from itertools import product, combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Vectorized check
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T to be 1 for all valid (i,j,k)
        
        # Build target tensor T of shape (n, m, p)
        # But we can check via matrix multiplication logic
        
        # Efficient check:
        # For each r, compute the contribution
        # Let's just do the full sum for all i, j, k
        
        # U: (n*m, rank) -> reshape to (n, m, rank)
        U_r = U_arr.reshape(n, m, rank)
        # V: (m*p, rank) -> reshape to (m, p, rank)
        V_r = V_arr.reshape(m, p, rank)
        # W: (p*n, rank) -> reshape to (p, n, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        # We want sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r] == 1 for all i,j,k
        
        # Let's iterate over r and accumulate
        # Or use einsum
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        
        # This is a bit tricky with standard einsum because of the index mixing.
        # Let's stick to a slightly optimized loop or batched operations
        
        # For each i, j, k:
        # s = sum_r U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:]
        
        # Let's use broadcasting
        # U_r[i,j,:] shape (rank,)
        # V_r[j,k,:] shape (rank,)
        # W_r[k,i,:] shape (rank,)
        
        # Create full tensor?
        # T[i,j,k] = np.einsum('r,r,r->', ...)
        
        # Let's try a batched approach for speed
        # Reshape U_r to (n, m, rank)
        # Reshape V_r to (m, p, rank)
        # Reshape W_r to (p, n, rank)
        
        # We can compute the tensor T of shape (n, m, p)
        # T[i,j,k] = sum_r U_r[i,j,r] V_r[j,k,r] W_r[k,i,r]
        
        # Let's iterate over r? No, rank is up to 60.
        # Let's iterate over i, j, k? 2*5*6 = 60 iterations.
        
        for i in range(n):
            for j in range(m):
                u_vec = U_r[i, j, :]
                for k in range(p):
                    v_vec = V_r[j, k, :]
                    w_vec = W_r[k, i, :]
                    s = np.dot(u_vec, v_vec * w_vec)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for _ in range(10):  # Multiple passes
            if time_left() < 5:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Check if term is redundant
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once by adding/subtracting them."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 10:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10:
                    return U_list, V_list, W_list, rank
                
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][idx1] = 0
                    U_new[i][idx2] = 0
                for j in range(m*p):
                    V_new[j][idx1] = 0
                    V_new[j][idx2] = 0
                for k in range(p*n):
                    W_new[k][idx1] = 0
                    W_new[k][idx2] = 0
                    
                if verify(U_new, V_new, W_new, rank):
                    # Remove both columns
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    # Restart loop to find more
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def find_rank_11_232():
        """
        Attempts to find a rank 11 decomposition for 2x3x2.
        Returns (U, V, W) lists or None.
        """
        # Dimensions for sub-problem
        # n_sub=2, m_sub=3, p_sub=2
        # U: (2*3, 11) = (6, 11)
        # V: (3*2, 11) = (6, 11)
        # W: (2*2, 11) = (4, 11)
        
        # Target tensor T[i,j,k] = 1
        # i in {0,1}, j in {0,1,2}, k in {0,1}
        
        # We will try to solve for a specific structure or use random search.
        # Given time constraints, we will try a few random restarts with a simple local search.
        
        # Actually, let's try to construct one manually or use a known one if available.
        # Since we don't have a known one, let's try to optimize.
        
        # We'll define the objective: sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
        # Plus penalty for non-integer/half-integer? No, just find any real solution first, then round?
        # The problem requires integer/half-integer.
        
        # Let's try to find a solution with entries in {0, 0.5, 1, -0.5, -1}.
        
        # This is a discrete search.
        # Let's try random search with limited domain.
        
        domain = [0, 0.5, 1, -0.5, -1]
        
        # Too large for random search over 6*11 + 6*11 + 4*11 = 154 variables.
        
        # Alternative: Use the fact that 2x3x2 is small.
        # We can set up a linear system if we fix some variables?
        
        # Let's just return None for now and rely on the main search.
        # Or try a very short random search.
        
        if time_left() < 5:
            return None
            
        # Try 10 random restarts
        for _ in range(10):
            if time_left() < 5:
                break
                
            U = np.random.choice(domain, size=(6, 11))
            V = np.random.choice(domain, size=(6, 11))
            W = np.random.choice(domain, size=(4, 11))
            
            # Simple check
            if verify(U.tolist(), V.tolist(), W.tolist(), 11):
                return U.tolist(), V.tolist(), W.tolist()
                
        return None

    # Main logic
    best_result = None
    best_rank = float('inf')
    
    # Try standard Strassen configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
    ]
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        if time_left() > 20:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
            
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to find a better decomposition using the 2x3x2 sub-block idea
    # If we can find a rank 11 for 2x3x2, we can potentially reduce the total rank.
    # Current best is 54.
    # If we use 2x2x2 Strassen (rank 7) for j={0,1}, k={0,1}
    # And 2x3x2 (rank 11?) for j={0,1,2}, k={2,3}? No, dimensions don't match.
    
    # Let's stick to the current best if we can't find a better one quickly.
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
