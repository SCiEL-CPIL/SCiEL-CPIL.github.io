import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 3, 5, 7
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Strategy: greedily pack Strassen blocks, then fill rest with schoolbook
    # A Strassen block covers (i0,i1) x (j0,j1) x (k0,k1) = 8 triples
    # We want to maximize the number of disjoint Strassen blocks
    
    # All possible Strassen blocks: choose 2 from each dimension
    i_pairs = [(0,1), (0,2), (1,2)]
    j_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    k_pairs = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
               (1,2), (1,3), (1,4), (1,5), (1,6),
               (2,3), (2,4), (2,5), (2,6),
               (3,4), (3,5), (3,6),
               (4,5), (4,6),
               (5,6)]
    
    # Each block is identified by (i_pair, j_pair, k_pair)
    # A block covers the set of (i,j,k) triples in the Cartesian product
    all_blocks = []
    for ip in i_pairs:
        for jp in j_pairs:
            for kp in k_pairs:
                # The set of covered triples
                covered = set()
                for i in ip:
                    for j in jp:
                        for k in kp:
                            covered.add((i, j, k))
                all_blocks.append((ip, jp, kp, covered))
    
    # Greedy: try to pack as many non-overlapping blocks as possible
    # Use multiple random orderings and take the best
    best_blocks = []
    best_count = 0
    
    rng = np.random.RandomState(42)
    
    for trial in range(200):
        if time.time() > deadline - 20:
            break
        
        # Shuffle blocks
        indices = list(range(len(all_blocks)))
        rng.shuffle(indices)
        
        used = set()  # set of (i,j,k) triples already covered
        selected = []
        
        for idx in indices:
            ip, jp, kp, covered = all_blocks[idx]
            if not covered & used:
                selected.append((ip, jp, kp))
                used |= covered
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_blocks = selected[:]
        
        # Early termination if we've found a good number
        # Maximum possible: floor(3/2)*floor(5/2)*floor(7/2) = 1*2*3 = 6
        # But with different i-pairs we might do better
        if best_count >= 8:
            break
    
    # Also try a systematic approach: fix which i's are used in Strassen
    # For each choice of i-subset of size 2, try to pack j and k pairs
    for i_pair in i_pairs:
        if time.time() > deadline - 20:
            break
        # For this i_pair, we can use Strassen blocks that use exactly this i_pair
        # The j and k dimensions are independent
        # Greedily pack j-pairs and k-pairs
        used_j = set()
        selected_j = []
        for jp in j_pairs:
            if not set(jp) & used_j:
                selected_j.append(jp)
                used_j |= set(jp)
        
        used_k = set()
        selected_k = []
        for kp in k_pairs:
            if not set(kp) & used_k:
                selected_k.append(kp)
                used_k |= set(kp)
        
        # All combinations
        blocks_for_this_i = [(i_pair, jp, kp) for jp in selected_j for kp in selected_k]
        if len(blocks_for_this_i) > best_count:
            best_count = len(blocks_for_this_i)
            best_blocks = blocks_for_this_i[:]
    
    # Now construct the decomposition
    # First, determine which triples are covered by Strassen blocks
    strassen_covered = set()
    for (ip, jp, kp) in best_blocks:
        for i in ip:
            for j in jp:
                for k in kp:
                    strassen_covered.add((i, j, k))
    
    # Schoolbook triples: all triples not covered by Strassen
    schoolbook_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    schoolbook_triples.append((i, j, k))
    
    rank = len(best_blocks) * 7 + len(schoolbook_triples)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Add Strassen blocks
    for (ip, jp, kp) in best_blocks:
        i0, i1 = ip
        j0, j1 = jp
        k0, k1 = kp
        
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0, col] += 1
        U[i1 * m + j1, col] += 1
        V[j0 * p + k0, col] += 1
        V[j1 * p + k1, col] += 1
        W[k0 * n + i0, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
        # M2: (A[i1,j0]+A[i1,j1])*(B[j0,k0])
        U[i1 * m + j0, col] += 1
        U[i1 * m + j1, col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + i1, col] += 1
        W[k1 * n + i1, col] -= 1
        col += 1
        
        # M3: (A[i0,j0])*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0, col] += 1
        V[j0 * p + k1, col] += 1
        V[j1 * p + k1, col] -= 1
        W[k1 * n + i0, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
        # M4: (A[i1,j1])*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1, col] += 1
        V[j1 * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + i1, col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j1])*(B[j1,k1])
        U[i0 * m + j0, col] += 1
        U[i0 * m + j1, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i0, col] += 1
        W[k0 * n + i0, col] -= 1
        col += 1
        
        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + k1, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
        
        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1, col] += 1
        U[i1 * m + j1, col] -= 1
        V[j1 * p + k0, col] += 1
        V[j1 * p + k1, col] += 1
        W[k0 * n + i0, col] += 1
        col += 1
    
    # Add schoolbook terms
    for (i, j, k) in schoolbook_triples:
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
    
    # Verify
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    try:
        computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if not np.allclose(computed, T, atol=1e-9):
            # Fallback to full schoolbook
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col_fb] = 1
                        V_fb[j * p + k, col_fb] = 1
                        W_fb[k * n + i, col_fb] = 1
                        col_fb += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
