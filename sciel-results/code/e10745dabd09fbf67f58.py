import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
        
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords

def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for return/cleanup
    
    # Precompute neighbors for all 6561 words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    neighbors_np = [None] * total
    neighbors_list = [None] * total
    neighbor_sets = [None] * total
    
    for idx in range(total):
        w = all_words[idx]
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = w.copy()
                    w_new[pos] = val
                    new_idx = 0
                    for v in w_new:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors_list[idx] = nbrs
        neighbors_np[idx] = np.array(nbrs, dtype=np.int32)
        neighbor_sets[idx] = set(nbrs)

    # Greedy Phase with NumPy
    covered = np.zeros(total, dtype=bool)
    code_list = []
    code_set = set()
    
    # Limit greedy phase to 10 seconds to allow more time for local search
    greedy_deadline = start_time + 10
    
    while time.time() < greedy_deadline:
        # Check if all covered
        if np.all(covered):
            break
            
        # Find best candidate
        best_idx = -1
        best_gain = -1
        
        for i in range(total):
            if i in code_set:
                continue
            gain = np.sum(~covered[neighbors_np[i]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        covered[neighbors_np[best_idx]] = True

    # Local Search Phase
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_np[c]] += 1
        
    best_size = len(code_list)
    best_code = list(code_list)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        # Phase 1: Try to remove words
        removed_any = False
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            c = code_list[i]
            # Check if removable: all neighbors have cover_count > 1
            if np.all(cover_count[neighbors_np[c]] > 1):
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors_np[c]] -= 1
                removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
        
        if removed_any:
            no_improvement_count = 0
            continue
        
        # Phase 2: Try swaps (1-for-1)
        if len(code_list) > 450:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            cover_count[neighbors_np[c]] -= 1
            
            # Find uncovered words
            nbrs_c = neighbors_np[c]
            uncovered_mask = cover_count[nbrs_c] == 0
            uncovered_words = nbrs_c[uncovered_mask]
            
            if len(uncovered_words) == 0:
                # Safe removal
                code_list.pop(idx_c)
                code_set.discard(c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                no_improvement_count = 0
                continue
            
            # Find replacement
            # Candidate must cover all uncovered_words
            candidate_set = neighbor_sets[uncovered_words[0]].copy()
            for u in uncovered_words[1:]:
                candidate_set &= neighbor_sets[u]
            
            candidate_set.discard(c)
            candidate_set -= code_set
            
            found_swap = False
            for idx_cp in candidate_set:
                # Perform swap
                code_list[idx_c] = idx_cp
                code_set.discard(c)
                code_set.add(idx_cp)
                cover_count[neighbors_np[idx_cp]] += 1
                found_swap = True
                no_improvement_count = 0
                break
            
            if not found_swap:
                # Revert
                cover_count[neighbors_np[c]] += 1
                no_improvement_count += 1
        
        # Phase 3: 2-for-1 escape (remove two, add one)
        if no_improvement_count > max_no_improvement:
            if time.time() < deadline - 10 and len(code_list) >= 2:
                i1, i2 = random.sample(range(len(code_list)), 2)
                c1 = code_list[i1]
                c2 = code_list[i2]
                
                # Temporarily remove both
                cover_count[neighbors_np[c1]] -= 1
                cover_count[neighbors_np[c2]] -= 1
                
                # Find uncovered words
                all_uncovered = set()
                nbrs1 = neighbors_np[c1]
                nbrs2 = neighbors_np[c2]
                for nbr in nbrs1:
                    if cover_count[nbr] == 0:
                        all_uncovered.add(nbr)
                for nbr in nbrs2:
                    if cover_count[nbr] == 0:
                        all_uncovered.add(nbr)
                
                if not all_uncovered:
                    # Safe removal of both
                    code_list.remove(c2)
                    code_list.remove(c1)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement_count = 0
                    continue
                
                # Find replacement
                uncovered_list = list(all_uncovered)
                candidate_set = neighbor_sets[uncovered_list[0]].copy()
                for u in uncovered_list[1:]:
                    candidate_set &= neighbor_sets[u]
                
                candidate_set.discard(c1)
                candidate_set.discard(c2)
                candidate_set -= code_set
                
                found = False
                for idx_cp in candidate_set:
                    # Swap
                    code_list.remove(c2)
                    code_list.remove(c1)
                    code_list.append(idx_cp)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    code_set.add(idx_cp)
                    cover_count[neighbors_np[idx_cp]] += 1
                    found = True
                    no_improvement_count = 0
                    break
                
                if not found:
                    # Revert
                    cover_count[neighbors_np[c1]] += 1
                    cover_count[neighbors_np[c2]] += 1
                    no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement * 2:
            # Aggressive restart: remove 10% and rebuild
            if time.time() < deadline - 15:
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, min(num_remove, len(code_list)))
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        cover_count[neighbors_np[c]] -= 1
                        code_list.remove(c)
                
                # Rebuild greedily
                while time.time() < deadline - 5:
                    # Check if all covered
                    if np.all(cover_count > 0):
                        break
                    
                    best_idx = -1
                    best_count = -1
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        # Count uncovered in neighbors
                        cnt = np.sum(cover_count[neighbors_np[idx]] == 0)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx
                    
                    if best_idx == -1 or best_count == 0:
                        break
                    
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    cover_count[neighbors_np[best_idx]] += 1
                
                no_improvement_count = 0
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
