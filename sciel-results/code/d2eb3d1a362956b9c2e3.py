import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition exactly using integer arithmetic."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, j_singles, k_pairs):
        """
        Build decomposition using Strassen 2x2x2 blocks for paired j,k
        and standard terms for single j with paired k.
        """
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        
        all_j = set(range(m))
        covered_j = used_j | set(j_singles)
        assert covered_j == all_j, f"j coverage mismatch: {covered_j} vs {all_j}"
        
        num_j_pairs = len(j_pairs)
        num_k_pairs = len(k_pairs)
        num_j_singles = len(j_singles)
        
        rank = 7 * num_j_pairs * num_k_pairs + 4 * num_j_singles * num_k_pairs
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        for j in j_singles:
            for k0, k1 in k_pairs:
                U[0*m+j, col] += 1
                V[j*p+k0, col] += 1
                W[k0*n+0, col] += 1
                col += 1
                
                U[0*m+j, col] += 1
                V[j*p+k1, col] += 1
                W[k1*n+0, col] += 1
                col += 1
                
                U[1*m+j, col] += 1
                V[j*p+k0, col] += 1
                W[k0*n+1, col] += 1
                col += 1
                
                U[1*m+j, col] += 1
                V[j*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank, max_iterations=500):
        """Try to eliminate redundant terms by checking if removing a column still verifies."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        iterations = 0
        while iterations < max_iterations:
            if time_left() < 10:
                break
            iterations += 1
            found = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
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
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    found = True
                    break
            
            if not found:
                break
        
        return U_list, V_list, W_list, rank

    def try_merge_proportional(U, V, W, rank, max_time=30):
        """
        Try to merge terms where U and V columns are proportional.
        If U[:,j] = c*U[:,i], V[:,j] = c*V[:,i], then we can absorb j into i
        by setting W[:,i] += W[:,j] and removing column j.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        start_ms = time.time()
        merged = True
        while merged:
            if time.time() - start_ms > max_time or time_left() < 10:
                break
            merged = False
            for i in range(rank - 1):
                if time_left() < 5:
                    break
                for j in range(i + 1, rank):
                    if time_left() < 5:
                        break
                    
                    U_i = U_arr[:, i]
                    U_j = U_arr[:, j]
                    V_i = V_arr[:, i]
                    V_j = V_arr[:, j]
                    
                    # Check if U_j is proportional to U_i
                    if np.linalg.norm(U_i) < 1e-12:
                        continue
                    c_u = np.dot(U_i, U_j) / np.dot(U_i, U_i)
                    if not np.allclose(U_j, c_u * U_i, atol=1e-9):
                        continue
                    
                    if np.linalg.norm(V_i) < 1e-12:
                        continue
                    c_v = np.dot(V_i, V_j) / np.dot(V_i, V_i)
                    if not np.allclose(V_j, c_v * V_i, atol=1e-9):
                        continue
                    
                    if abs(c_u - c_v) > 1e-9:
                        continue
                    
                    # Merge: W[:,i] += W[:,j], remove column j
                    W_arr[:, i] = W_arr[:, i] + W_arr[:, j]
                    U_arr = np.delete(U_arr, j, axis=1)
                    V_arr = np.delete(V_arr, j, axis=1)
                    W_arr = np.delete(W_arr, j, axis=1)
                    rank -= 1
                    merged = True
                    break
                if merged:
                    break
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

    def try_pairwise_merge(U, V, W, rank, max_time=20):
        """
        Try to merge pairs of terms into single terms using more general conditions.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        start_ms = time.time()
        merged = True
        while merged:
            if time.time() - start_ms > max_time or time_left() < 10:
                break
            merged = False
            for i in range(rank - 1):
                if time_left() < 5:
                    break
                for j in range(i + 1, rank):
                    if time_left() < 5:
                        break
                    
                    U_i = U_arr[:, i]
                    U_j = U_arr[:, j]
                    V_i = V_arr[:, i]
                    V_j = V_arr[:, j]
                    W_i = W_arr[:, i]
                    W_j = W_arr[:, j]
                    
                    # Check if U_j = c*U_i and V_j = c*V_i
                    if np.linalg.norm(U_i) > 1e-9:
                        c_u = np.dot(U_i, U_j) / np.dot(U_i, U_i)
                        if np.allclose(U_j, c_u * U_i, atol=1e-9):
                            if np.linalg.norm(V_i) > 1e-9:
                                c_v = np.dot(V_i, V_j) / np.dot(V_i, V_i)
                                if np.allclose(V_j, c_v * V_i, atol=1e-9):
                                    if abs(c_u - c_v) < 1e-9:
                                        W_arr[:, i] = W_i + W_j
                                        U_arr = np.delete(U_arr, j, axis=1)
                                        V_arr = np.delete(V_arr, j, axis=1)
                                        W_arr = np.delete(W_arr, j, axis=1)
                                        rank -= 1
                                        merged = True
                                        break
                    if merged:
                        break
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

    # Try different configurations
    configurations = [
        # 2 j-pairs, 1 j-single; 3 k-pairs (rank 54)
        ([(0,1),(2,3)], [4], [(0,1),(2,3),(4,5)]),
        ([(0,1),(3,4)], [2], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [0], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [4], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,4)], [2], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [2], [(0,1),(2,3),(4,5)]),
        # Different k pairings
        ([(0,1),(2,3)], [4], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [4], [(0,2),(1,3),(4,5)]),
        ([(0,1),(2,3)], [4], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [4], [(0,5),(1,2),(3,4)]),
        # More diverse j pairings
        ([(0,1),(2,4)], [3], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [1], [(0,1),(2,3),(4,5)]),
        ([(1,3),(2,4)], [0], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, j_singles, k_pairs in configurations:
        if time_left() < 15:
            break
        
        U, V, W, rank = build_strassen_decomp(j_pairs, j_singles, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Step 1: Eliminate redundant terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Step 2: Merge proportional terms
        U_m, V_m, W_m, rank_m = try_merge_proportional(U_list, V_list, W_list, new_rank, max_time=15)
        if rank_m < new_rank:
            new_rank = rank_m
            U_list, V_list, W_list = U_m, V_m, W_m
        
        # Step 3: Another elimination pass after merging
        U_e2, V_e2, W_e2, rank_e2 = try_eliminate_terms(U_list, V_list, W_list, new_rank)
        if rank_e2 < new_rank:
            new_rank = rank_e2
            U_list, V_list, W_list = U_e2, V_e2, W_e2
        
        # Step 4: Final merge pass
        U_f, V_f, W_f, rank_f = try_pairwise_merge(U_list, V_list, W_list, new_rank, max_time=10)
        if rank_f < new_rank:
            new_rank = rank_f
            U_list, V_list, W_list = U_f, V_f, W_f
        
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
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
