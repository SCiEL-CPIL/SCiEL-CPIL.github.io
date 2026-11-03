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
        """Vectorized verification: check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1 for all (i,j,k)."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build tensors of shape (n, m, p, rank)
        # U_t[i,j,k,t] = U[i*m+j, t] (independent of k)
        # V_t[i,j,k,t] = V[j*p+k, t] (independent of i)
        # W_t[i,j,k,t] = W[k*n+i, t] (independent of j)
        
        # Reshape U: (n, m, rank) -> (n, m, 1, rank)
        U_t = U_arr.reshape(n, m, 1, rank)
        # Reshape V: (m, p, rank) -> (1, m, p, rank)
        V_t = V_arr.reshape(1, m, p, rank)
        # Reshape W: (p, n, rank) -> (n, 1, p, rank)
        W_t = W_arr.reshape(n, 1, p, rank)
        
        # Product: (n, m, p, rank)
        prod = U_t * V_t * W_t
        # Sum over rank axis: (n, m, p)
        result = prod.sum(axis=3)
        
        return np.allclose(result, 1.0, atol=tol)

    def verify_fast_list(U_list, V_list, W_list, rank, tol=1e-9):
        """Vectorized verification for list inputs."""
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        U_t = U_arr.reshape(n, m, 1, rank)
        V_t = V_arr.reshape(1, m, p, rank)
        W_t = W_arr.reshape(n, 1, p, rank)
        
        prod = U_t * V_t * W_t
        result = prod.sum(axis=3)
        
        return np.allclose(result, 1.0, atol=tol)

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
        U_arr = U.copy()
        V_arr = V.copy()
        W_arr = W.copy()
        
        for _ in range(5):  # Multiple passes
            if time_left() < 3:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 3:
                    break
                
                # Zero out term_idx
                U_arr[:, term_idx] = 0
                V_arr[:, term_idx] = 0
                W_arr[:, term_idx] = 0
                
                if verify_fast(U_arr, V_arr, W_arr, rank):
                    # Remove this column
                    keep = list(range(term_idx)) + list(range(term_idx+1, rank))
                    U_arr = U_arr[:, keep]
                    V_arr = V_arr[:, keep]
                    W_arr = W_arr[:, keep]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once."""
        if time_left() < 5:
            return U_list, V_list, W_list, rank
            
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        for idx1 in range(rank):
            if time_left() < 5:
                break
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 5:
                    break
                
                U_save = U_arr[:, idx1].copy()
                V_save = V_arr[:, idx1].copy()
                W_save = W_arr[:, idx1].copy()
                U_save2 = U_arr[:, idx2].copy()
                V_save2 = V_arr[:, idx2].copy()
                W_save2 = W_arr[:, idx2].copy()
                
                U_arr[:, idx1] = 0
                U_arr[:, idx2] = 0
                V_arr[:, idx1] = 0
                V_arr[:, idx2] = 0
                W_arr[:, idx1] = 0
                W_arr[:, idx2] = 0
                    
                if verify_fast(U_arr, V_arr, W_arr, rank):
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_arr = U_arr[:, keep]
                    V_arr = V_arr[:, keep]
                    W_arr = W_arr[:, keep]
                    rank -= 2
                    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank
                else:
                    # Restore
                    U_arr[:, idx1] = U_save
                    U_arr[:, idx2] = U_save2
                    V_arr[:, idx1] = V_save
                    V_arr[:, idx2] = V_save2
                    W_arr[:, idx1] = W_save
                    W_arr[:, idx2] = W_save2
                    
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), rank

    # Direct rank-54 construction: 2 Strassen blocks (j-pairs) × 1 k-pair + 3 schoolbook j's
    # Actually: 2 j-pairs × 3 k-pairs = 6 blocks × 7 = 42, plus 1 remaining j × 6 k's × 2 = 12, total 54
    # Let's use a simpler direct construction:
    # j-pairs: [(0,1), (2,3)], remaining: [4]
    # k-pairs: [(0,1), (2,3), (4,5)]
    # Rank = 7 * 2 * 3 + 2 * 1 * 6 = 42 + 12 = 54
    
    best_result = None
    best_rank = float('inf')
    
    # Quick direct construction for rank 54
    j_pairs_direct = [(0, 1), (2, 3)]
    k_pairs_direct = [(0, 1), (2, 3), (4, 5)]
    remaining_j_direct = [4]
    
    U, V, W, rank = build_strassen_decomp(j_pairs_direct, k_pairs_direct, remaining_j_direct)
    
    if verify_fast(U, V, W, rank):
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try elimination on the direct construction
    if best_rank == 54:
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
        
        if time_left() > 10:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Generate all possible j-pairings (2 pairs covering 4 of 5 j's)
    all_j_pairs = []
    for j0 in range(m):
        for j1 in range(j0+1, m):
            for j2 in range(j0+1, m):
                if j2 == j1:
                    continue
                for j3 in range(j2+1, m):
                    used = {j0, j1, j2, j3}
                    remaining = [j for j in range(m) if j not in used]
                    if len(remaining) == 1:
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
    
    # Try combinations of j and k pairings (skip the one we already tried)
    for j_idx, (j0_j1, j2_j3, rem_j) in enumerate(unique_j_configs):
        if time_left() < 3:
            break
        for k_idx, k_triplet in enumerate(unique_k_configs):
            if time_left() < 3:
                break
            
            # Skip the direct construction we already tried
            if (j0_j1, j2_j3, rem_j) == (j_pairs_direct[0], j_pairs_direct[1], remaining_j_direct[0]):
                continue
            if tuple(sorted([frozenset(t) for t in k_triplet])) == tuple(sorted([frozenset(t) for t in k_pairs_direct])):
                if (j0_j1, j2_j3, rem_j) == (j_pairs_direct[0], j_pairs_direct[1], remaining_j_direct[0]):
                    continue
            
            j_pairs = [j0_j1, j2_j3]
            k_pairs = list(k_triplet)
            
            U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs, [rem_j])
            
            if not verify_fast(U, V, W, rank):
                continue
            
            # Try to eliminate terms
            U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
            
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                
            # Try pairwise elimination if time permits and we found something good
            if new_rank < 54 and time_left() > 10:
                U_list, V_list, W_list, pw_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
                if pw_rank < best_rank:
                    best_rank = pw_rank
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
