import time
import numpy as np
from itertools import product, combinations, permutations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify_fast(U, V, W, rank, tol=1e-9):
        """Fast vectorized verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i, j, k), check sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs, remaining_j):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
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
        """Try to eliminate redundant terms by checking if zeroing them still works."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for _ in range(3):  # Multiple passes
            if time_left() < 5:
                break
            progress = False
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
                
                if verify_fast(U_new, V_new, W_new, rank):
                    keep = [j for j in range(rank) if j != term_idx]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once."""
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
                    
                if verify_fast(U_new, V_new, W_new, rank):
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    # Generate all possible j-pairings (2 pairs covering 4 of 5 j's)
    all_j_pairs = []
    for j0 in range(m):
        for j1 in range(j0+1, m):
            for j2 in range(j0+1, m):
                if j2 == j1:
                    continue
                for j3 in range(j2+1, m):
                    # pairs are (j0,j1) and (j2,j3), remaining is the one not used
                    used = {j0, j1, j2, j3}
                    remaining = [j for j in range(m) if j not in used]
                    if len(remaining) == 1:
                        # Check we haven't already added this exact set of pairs
                        key = (frozenset((j0,j1)), frozenset((j2,j3)), remaining[0])
                        all_j_pairs.append(((j0,j1),(j2,j3), remaining[0]))
    
    # Deduplicate j-pair configurations
    seen_j = set()
    unique_j_configs = []
    for jp in all_j_pairs:
        key = (frozenset(jp[0]), frozenset(jp[1]), jp[2])
        if key not in seen_j:
            seen_j.add(key)
            unique_j_configs.append(jp)
    
    # Generate all possible k-pairings (3 pairs covering all 6 k's)
    all_k_pairs = []
    for k0 in range(p):
        for k1 in range(k0+1, p):
            remaining_k = [k for k in range(p) if k != k0 and k != k1]
            for k2 in range(len(remaining_k)):
                for k3 in range(k2+1, len(remaining_k)):
                    k2v = remaining_k[k2]
                    k3v = remaining_k[k3]
                    remaining_k2 = [k for k in remaining_k if k != k2v and k != k3v]
                    if len(remaining_k2) == 2:
                        k4v, k5v = remaining_k2
                        all_k_pairs.append(((k0,k1),(k2v,k3v),(k4v,k5v)))
    
    # Deduplicate k-pair configurations
    seen_k = set()
    unique_k_configs = []
    for kp in all_k_pairs:
        key = tuple(sorted([frozenset(kp[0]), frozenset(kp[1]), frozenset(kp[2])]))
        if key not in seen_k:
            seen_k.add(key)
            unique_k_configs.append(kp)
    
    best_result = None
    best_rank = float('inf')
    
    # Try combinations of j and k pairings
    # Limit to avoid combinatorial explosion
    max_j_configs = min(len(unique_j_configs), 30)
    max_k_configs = min(len(unique_k_configs), 30)
    
    for j_idx, (j0_j1, j2_j3, rem_j) in enumerate(unique_j_configs[:max_j_configs]):
        if time_left() < 5:
            break
        for k_idx, k_triplet in enumerate(unique_k_configs[:max_k_configs]):
            if time_left() < 5:
                break
            
            j_pairs = [j0_j1, j2_j3]
            k_pairs = list(k_triplet)
            
            U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs, [rem_j])
            
            if not verify_fast(U, V, W, rank):
                continue
            
            # Try to eliminate terms
            U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
            
            # Try pairwise elimination if time permits
            if time_left() > 15:
                U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
                
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
