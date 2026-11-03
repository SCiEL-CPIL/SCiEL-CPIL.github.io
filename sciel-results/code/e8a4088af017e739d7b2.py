import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_vectorized(U, V, W, rank):
        """Fast vectorized verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_rows = U_arr[i_idx * m + j_idx, :]
        v_rows = V_arr[j_idx * p + k_idx, :]
        w_rows = W_arr[k_idx * n + i_idx, :]
        
        # Sum over rank dimension
        result = np.einsum('tr,tr,tr->t', u_rows, v_rows, w_rows)
        
        return np.all(np.abs(result - 1.0) < 1e-9)

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
        """Try to eliminate redundant terms using vectorized verification."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for _ in range(8):  # Multiple passes
            if time_left() < 10:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 10:
                    break
                # Zero out the term
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify_vectorized(U_new, V_new, W_new, rank):
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
                    break  # Restart from beginning after removal
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_linear_combination_elimination(U_list, V_list, W_list, rank):
        """Try to express one term as a linear combination of others."""
        if time_left() < 20:
            return U_list, V_list, W_list, rank
            
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        for target in range(rank):
            if time_left() < 20:
                break
            if rank <= 1:
                break
            
            others = [i for i in range(rank) if i != target]
            
            # Check if target is proportional to any other term in all three matrices
            for other in others:
                if time_left() < 20:
                    break
                if np.allclose(U_arr[:, target], 0.0) and np.allclose(U_arr[:, other], 0.0):
                    continue
                if np.allclose(U_arr[:, target], 0.0) or np.allclose(U_arr[:, other], 0.0):
                    continue
                alpha_u = U_arr[np.abs(U_arr[:, other]) > 1e-12, target] / U_arr[np.abs(U_arr[:, other]) > 1e-12, other]
                if len(alpha_u) == 0 or np.std(alpha_u) > 1e-10:
                    continue
                alpha = alpha_u[0]
                
                if np.allclose(V_arr[:, target], alpha * V_arr[:, other], atol=1e-10):
                    if np.allclose(W_arr[:, target], alpha * W_arr[:, other], atol=1e-10):
                        keep = [i for i in range(rank) if i != other]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 1
                        U_arr = np.array(U_list, dtype=float)
                        V_arr = np.array(V_list, dtype=float)
                        W_arr = np.array(W_list, dtype=float)
                        target = min(target, rank-1)
                        break
        
        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once."""
        if time_left() < 15:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 15:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 15:
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
                    
                if verify_vectorized(U_new, V_new, W_new, rank):
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def try_triple_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate three terms at once."""
        if time_left() < 12:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 12:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 12:
                    return U_list, V_list, W_list, rank
                for idx3 in range(idx2 + 1, rank):
                    if time_left() < 12:
                        return U_list, V_list, W_list, rank
                    
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][idx1] = 0
                        U_new[i][idx2] = 0
                        U_new[i][idx3] = 0
                    for j in range(m*p):
                        V_new[j][idx1] = 0
                        V_new[j][idx2] = 0
                        V_new[j][idx3] = 0
                    for k in range(p*n):
                        W_new[k][idx1] = 0
                        W_new[k][idx2] = 0
                        W_new[k][idx3] = 0
                        
                    if verify_vectorized(U_new, V_new, W_new, rank):
                        keep = [j for j in range(rank) if j != idx1 and j != idx2 and j != idx3]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 3
                        return try_triple_elimination(U_list, V_list, W_list, rank)
                        
        return U_list, V_list, W_list, rank

    # Try different configurations - more diverse
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(1,2),(3,4),(0,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(2,3)], [(1,4),(0,5),(2,3)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(2,4)], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify_vectorized(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try linear combination elimination
        if time_left() > 25:
            U_list, V_list, W_list, new_rank = try_linear_combination_elimination(U_list, V_list, W_list, new_rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 20:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
            
        # Try triple elimination if time permits
        if time_left() > 18:
            U_list, V_list, W_list, new_rank = try_triple_elimination(U_list, V_list, W_list, new_rank)
            
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
