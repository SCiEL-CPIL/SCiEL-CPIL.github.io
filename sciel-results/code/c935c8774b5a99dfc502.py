import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Precompute neighbors efficiently using index arithmetic
    neighbors = [None] * N
    for i in range(N):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
    
    # Precompute neighbor sets for fast intersection in local search
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Phase 1: Fast Greedy with count-array approach
    covered = np.zeros(N, dtype=bool)
    code_list = []
    count = np.full(N, 17, dtype=np.int32)  # Each word covers itself + 16 neighbors = 17
    
    while not covered.all() and time.time() < deadline - 40.0:
        c = int(np.argmax(count))
        if count[c] == 0:
            break
        code_list.append(c)
        for x in neighbors[c]:
            if not covered[x]:
                covered[x] = True
                for y in neighbors[x]:
                    count[y] -= 1
    
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # Initialize coverage counts
    current_cov_count = np.zeros(N, dtype=np.int32)
    for c in current_code:
        for x in neighbors[c]:
            current_cov_count[x] += 1
    
    random.seed(42)
    
    # Helper: try to remove a codeword if all its neighbors have count >= 2
    def try_remove(idx):
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        c = current_code[idx]
        removable = True
        for x in neighbors[c]:
            if current_cov_count[x] <= 1:
                removable = False
                break
        if removable:
            for x in neighbors[c]:
                current_cov_count[x] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            return True
        return False
    
    # Helper: try a swap - remove c, add c' that covers all unique points of c
    def try_swap(c_idx):
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        c = current_code[c_idx]
        risky_points = []
        for x in neighbors[c]:
            if current_cov_count[x] == 1:
                risky_points.append(x)
        
        if not risky_points:
            # Can just remove
            for x in neighbors[c]:
                current_cov_count[x] -= 1
            current_code.pop(c_idx)
            current_size -= 1
            stuck_counter = 0
            return True
        
        if len(risky_points) > 3:
            return False
        
        # Find candidates that cover all risky points
        candidate_set = neighbor_sets[risky_points[0]].copy()
        for x in risky_points[1:]:
            candidate_set &= neighbor_sets[x]
            if not candidate_set:
                break
        
        if not candidate_set:
            return False
        
        # Remove current codewords from candidates
        for c_code in current_code:
            if c_code != c:
                candidate_set.discard(c_code)
        candidate_set.discard(c)
        
        if not candidate_set:
            return False
        
        # Pick a candidate with good overlap
        candidates = list(candidate_set)[:20]
        best_cand = None
        best_score = -1
        for cand in candidates:
            overlap = sum(1 for x in neighbors[cand] if current_cov_count[x] > 0)
            score = overlap
            if score > best_score:
                best_score = score
                best_cand = cand
        
        if best_cand is None:
            return False
        
        # Apply swap
        for x in neighbors[c]:
            current_cov_count[x] -= 1
        for x in neighbors[best_cand]:
            current_cov_count[x] += 1
        current_code[c_idx] = best_cand
        stuck_counter = 0
        return True
    
    # Helper: try two-swap - remove two codewords, add one
    def try_two_swap():
        nonlocal current_code, current_size, current_cov_count, stuck_counter
        if current_size < 2:
            return False
        
        i1 = random.randint(0, current_size - 1)
        i2 = random.randint(0, current_size - 1)
        if i2 == i1:
            i2 = (i2 + 1) % current_size
        
        c1 = current_code[i1]
        c2 = current_code[i2]
        
        # Find points that would become uncovered if we remove both c1 and c2
        risky = set()
        for x in neighbors[c1]:
            if current_cov_count[x] == 1:
                risky.add(x)
        for x in neighbors[c2]:
            if current_cov_count[x] == 1:
                risky.add(x)
        
        # Also consider points covered by both c1 and c2 with count == 2
        # If we remove both, they become uncovered
        for x in neighbors[c1]:
            if current_cov_count[x] == 2 and x in neighbors[c2]:
                risky.add(x)
        
        if len(risky) > 4:
            return False
        
        if not risky:
            # Can just remove both
            for x in neighbors[c1]:
                current_cov_count[x] -= 1
            for x in neighbors[c2]:
                current_cov_count[x] -= 1
            # Remove from code (handle index carefully)
            idxs = sorted([i1, i2], reverse=True)
            for idx in idxs:
                current_code.pop(idx)
            current_size -= 2
            stuck_counter = 0
            return True
        
        # Find candidates that cover all risky points
        risky_list = list(risky)
        candidate_set = neighbor_sets[risky_list[0]].copy()
        for x in risky_list[1:]:
            candidate_set &= neighbor_sets[x]
            if not candidate_set:
                break
        
        if not candidate_set:
            return False
        
        # Remove current codewords from candidates
        for c_code in current_code:
            candidate_set.discard(c_code)
        
        if not candidate_set:
            return False
        
        # Pick a candidate with good overlap
        candidates = list(candidate_set)[:20]
        best_cand = None
        best_score = -1
        for cand in candidates:
            overlap = sum(1 for x in neighbors[cand] if current_cov_count[x] > 0)
            score = overlap
            if score > best_score:
                best_score = score
                best_cand = cand
        
        if best_cand is None:
            return False
        
        # Apply: remove c1 and c2, add best_cand
        for x in neighbors[c1]:
            current_cov_count[x] -= 1
        for x in neighbors[c2]:
            current_cov_count[x] -= 1
        for x in neighbors[best_cand]:
            current_cov_count[x] += 1
        
        idxs = sorted([i1, i2], reverse=True)
        for idx in idxs:
            current_code.pop(idx)
        current_code.append(best_cand)
        current_size -= 1
        stuck_counter = 0
        return True
    
    # Helper: full reduction pass
    def full_reduction():
        nonlocal current_code, current_size, current_cov_count
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in current_code:
                if np.min(current_cov_count[neighbors[c]]) >= 2:
                    current_cov_count[neighbors[c]] -= 1
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                current_code = new_code
                current_size = len(current_code)
    
    # Phase 2: Local Search
    stuck_counter = 0
    
    while time.time() < deadline - 5.0:
        if current_size == 0:
            break
        
        # First, try a full reduction
        old_size = current_size
        full_reduction()
        
        if current_size < best_size:
            best_size = current_size
            best_code = current_code[:]
        
        if current_size < old_size:
            stuck_counter = 0
            continue
        
        # Try random removals
        removed = False
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            if current_size == 0:
                break
            idx = random.randint(0, current_size - 1)
            if try_remove(idx):
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                removed = True
                break
        
        if removed:
            continue
        
        stuck_counter += 1
        
        # Try swaps if stuck
        if stuck_counter > 2 and time.time() < deadline - 10.0:
            swapped = False
            for _ in range(5):
                if time.time() > deadline - 10.0:
                    break
                if current_size == 0:
                    break
                c_idx = random.randint(0, current_size - 1)
                if try_swap(c_idx):
                    # Try reduction after swap
                    old_sz = current_size
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    swapped = True
                    break
            
            if swapped:
                continue
        
        # Try two-swaps if more stuck
        if stuck_counter > 10 and current_size >= 2 and time.time() < deadline - 15.0:
            two_swapped = False
            for _ in range(3):
                if time.time() > deadline - 15.0:
                    break
                if try_two_swap():
                    # Try reduction after two-swap
                    old_sz = current_size
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    two_swapped = True
                    break
            
            if two_swapped:
                continue
        
        # If completely stuck, try a random kick to escape local minimum
        if stuck_counter > 20 and time.time() < deadline - 10.0 and current_size > 0:
            c_idx = random.randint(0, current_size - 1)
            c = current_code[c_idx]
            cand = neighbors[c][random.randint(1, len(neighbors[c])-1)]  # Skip self
            if cand not in current_code:
                old_code = current_code[:]
                old_counts = current_cov_count.copy()
                
                # Apply kick
                for x in neighbors[c]:
                    current_cov_count[x] -= 1
                for x in neighbors[cand]:
                    current_cov_count[x] += 1
                current_code[c_idx] = cand
                
                # Check validity
                if np.any(current_cov_count < 1):
                    # Revert
                    current_code = old_code
                    current_cov_count = old_counts
                else:
                    # Try reduction
                    full_reduction()
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    stuck_counter = 0
    
    # Final reduction
    current_code = best_code[:]
    current_size = len(current_code)
    current_cov_count = np.zeros(N, dtype=np.int32)
    for c in current_code:
        for x in neighbors[c]:
            current_cov_count[x] += 1
    full_reduction()
    
    all_words = list(itertools.product(range(3), repeat=8))
    return [list(all_words[i]) for i in current_code]
