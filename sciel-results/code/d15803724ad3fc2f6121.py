import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0
    
    total_words = q ** n
    
    # Generate all words
    words = []
    for i in range(total_words):
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (Hamming distance <= radius)
    # For radius=1, this is the word itself plus all words differing in exactly 1 position
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(neigh)
    
    # Build sparse incidence matrix A
    # A[j, i] = 1 if word j is covered by word i (dist <= radius)
    rows = []
    cols = []
    data = []
    for i in range(total_words):
        for j in neighbors[i]:
            rows.append(j)
            cols.append(i)
            data.append(1)
    
    A_sparse = csr_matrix((data, (rows, cols)), shape=(total_words, total_words), dtype=np.int8)
    
    # MILP formulation
    c_obj = np.ones(total_words)
    constraints = LinearConstraint(A_sparse, lb=np.ones(total_words), ub=np.inf)
    integrality = np.ones(total_words)
    bounds = Bounds(0, 1)
    
    best_C = []
    
    # Try MILP first
    try:
        remaining = deadline - (time.time() - start_time) - 5
        if remaining > 10:
            res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={"time_limit": remaining})
            if res.status in [0, 1] and res.x is not None:
                x_bin = (res.x > 0.5).astype(int)
                best_C = [i for i in range(total_words) if x_bin[i] == 1]
    except Exception:
        pass
    
    # Fallback or initial: Greedy Set Cover
    if not best_C:
        best_C = _greedy_cover(total_words, neighbors)
    
    # Post-processing: Local search to reduce size
    # 1. Remove redundant codewords
    best_C = _remove_redundant(best_C, total_words, neighbors)
    
    # 2. Try swaps to enable further removals
    if time.time() - start_time < deadline - 10:
        best_C = _local_search_reduce(best_C, total_words, neighbors, deadline - 15)
    
    # Final verification
    if best_C:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_C:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        if np.min(cover_count) < 1:
            # Invalid, fallback to greedy
            best_C = _greedy_cover(total_words, neighbors)
            best_C = _remove_redundant(best_C, total_words, neighbors)
    
    result = [list(words[idx]) for idx in best_C]
    return result

def _greedy_cover(total_words, neighbors):
    uncovered = set(range(total_words))
    C = []
    # Precompute coverage counts for speed? Or just iterate.
    # Since total_words=729, O(N^2 * radius) is fine.
    while uncovered:
        best_c = -1
        max_covered = -1
        for c in range(total_words):
            covered = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    covered += 1
            if covered > max_covered:
                max_covered = covered
                best_c = c
        if best_c == -1:
            break
        C.append(best_c)
        for nb in neighbors[best_c]:
            uncovered.discard(nb)
    return C

def _remove_redundant(C, total_words, neighbors):
    C = list(C)
    in_code = [False] * total_words
    for c in C:
        in_code[c] = True
        
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in C:
            # Check if c is essential
            essential = False
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    essential = True
                    break
            if not essential:
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                in_code[c] = False
                changed = True
            else:
                new_C.append(c)
        C = new_C
    return C

def _local_search_reduce(C, total_words, neighbors, deadline):
    C = list(C)
    in_code = [False] * total_words
    for c in C:
        in_code[c] = True
        
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove any codeword
        for i in range(len(C)):
            c_i = C[i]
            # Check if c_i is essential
            essential = False
            for nb in neighbors[c_i]:
                if cover_count[nb] == 1:
                    essential = True
                    break
            if not essential:
                # Remove it
                for nb in neighbors[c_i]:
                    cover_count[nb] -= 1
                in_code[c_i] = False
                C.pop(i)
                improved = True
                break
        
        if improved:
            continue
            
        # Try swaps: replace c_i with c_j
        for i in range(len(C)):
            c_i = C[i]
            # Find words covered exclusively by c_i
            exclusive = []
            for nb in neighbors[c_i]:
                if cover_count[nb] == 1:
                    exclusive.append(nb)
            
            if not exclusive:
                continue
                
            # Find a candidate c_j not in C that covers all exclusive words
            found_swap = False
            for c_j in range(total_words):
                if in_code[c_j]:
                    continue
                can_cover = True
                for nb in exclusive:
                    if nb not in neighbors[c_j]:
                        can_cover = False
                        break
                if can_cover:
                    # Perform swap
                    # Remove c_i
                    for nb in neighbors[c_i]:
                        cover_count[nb] -= 1
                    in_code[c_i] = False
                    
                    # Add c_j
                    for nb in neighbors[c_j]:
                        cover_count[nb] += 1
                    in_code[c_j] = True
                    
                    C[i] = c_j
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break
    return C
