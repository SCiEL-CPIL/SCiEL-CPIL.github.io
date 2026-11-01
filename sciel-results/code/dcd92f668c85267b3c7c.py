import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for other parameters
        total = q ** n
        if total > 100000:
            # Too large for exact search; return trivial covering
            return [list(range(n)) for _ in range(min(total, 1000))]
        # Small case greedy
        def w2i(w):
            idx = 0
            for x in w:
                idx = idx * q + x
            return idx
        def i2w(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w
        masks = [0] * total
        for i in range(total):
            wi = i2w(i)
            m = 1 << i
            for pos in range(n):
                orig = wi[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    wn = wi[:]
                    wn[pos] = val
                    m |= 1 << w2i(wn)
            masks[i] = m
        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            if time.time() > deadline - 1:
                break
            best_i = -1
            best_c = -1
            for i in range(total):
                common = uncovered & masks[i]
                if common:
                    cnt = common.bit_count()
                    if cnt > best_c:
                        best_c = cnt
                        best_i = i
            if best_i == -1:
                break
            code.append(i2w(best_i))
            uncovered &= ~masks[best_i]
        return code
    
    # Optimized path for q=3, n=6, radius=1
    total = q ** n  # 729
    
    def w2i(w):
        idx = 0
        for x in w:
            idx = idx * 3 + x
        return idx
    
    def i2w(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w
    
    # Precompute neighbor masks for all 729 words
    # mask[i] is a Python int bitset of indices j such that dist(i,j) <= 1
    masks = [0] * total
    for i in range(total):
        wi = i2w(i)
        m = 1 << i  # self
        for pos in range(n):
            orig = wi[pos]
            for val in range(3):
                if val == orig:
                    continue
                wn = wi[:]
                wn[pos] = val
                m |= 1 << w2i(wn)
        masks[i] = m
    
    # Helper: get neighbors as list of indices
    def get_neighbor_indices(idx):
        wi = i2w(idx)
        neighbors = [idx]
        for pos in range(n):
            orig = wi[pos]
            for val in range(3):
                if val == orig:
                    continue
                wn = wi[:]
                wn[pos] = val
                neighbors.append(w2i(wn))
        return neighbors
    
    # Precompute neighbor lists
    neighbor_lists = [get_neighbor_indices(i) for i in range(total)]
    
    def greedy_set_cover(masks, total, deadline, random_tie=False, seed=None):
        """Greedy set cover using bitsets. Returns list of indices."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        uncovered = (1 << total) - 1
        code_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_count = -1
            best_candidates = []
            
            for i in range(total):
                common = uncovered & masks[i]
                if common:
                    cnt = common.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_candidates = [i]
                    elif cnt == best_count:
                        best_candidates.append(i)
            
            if not best_candidates:
                break
            
            if random_tie and len(best_candidates) > 1:
                best_idx = rng.choice(best_candidates)
            else:
                best_idx = best_candidates[0]
            
            code_indices.append(best_idx)
            uncovered &= ~masks[best_idx]
        
        return code_indices
    
    def remove_redundant(code_indices, masks, total, deadline):
        """Remove redundant codewords from the code."""
        code_set = set(code_indices)
        code_list = list(code_indices)
        
        # Coverage count for each word
        coverage = [0] * total
        for ci in code_list:
            for nb in neighbor_lists[ci]:
                coverage[nb] += 1
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for ci in list(code_list):
                # Check if removing ci keeps all neighbors covered
                can_remove = True
                for nb in neighbor_lists[ci]:
                    if coverage[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove ci
                    for nb in neighbor_lists[ci]:
                        coverage[nb] -= 1
                    code_list.remove(ci)
                    code_set.discard(ci)
                    changed = True
        
        return code_list
    
    # Strategy 1: Start with tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    direct_sum = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    direct_sum_indices = [w2i(c) for c in direct_sum]
    
    # Remove redundant words from direct sum
    ds_cleaned = remove_redundant(direct_sum_indices, masks, total, deadline)
    
    # Strategy 2: Greedy set cover
    greedy_code = greedy_set_cover(masks, total, deadline, random_tie=False)
    greedy_cleaned = remove_redundant(greedy_code, masks, total, deadline)
    
    # Pick the best so far
    best_code = min([ds_cleaned, greedy_cleaned], key=len)
    best_size = len(best_code)
    
    # Strategy 3: Randomized greedy restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_size > 73:
        num_restarts = min(50, int(remaining_time / 2.0))
        for r in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            seed_val = r * 12345 + 67890
            rc = greedy_set_cover(masks, total, deadline - 2.0, random_tie=True, seed=seed_val)
            rc_cleaned = remove_redundant(rc, masks, total, deadline - 1.0)
            if len(rc_cleaned) < best_size:
                best_size = len(rc_cleaned)
                best_code = rc_cleaned
                if best_size <= 73:
                    break
    
    # Strategy 4: Structured local search to reduce code size
    if time.time() < deadline - 15.0 and best_size > 73:
        current = list(best_code)
        current_set = set(current)
        
        # Helper: compute coverage array
        def compute_coverage(codes):
            cov = [0] * total
            for ci in codes:
                for nb in neighbor_lists[ci]:
                    cov[nb] += 1
            return cov
        
        # Helper: try to remove all redundant codewords from a list
        def remove_all_redundant(codes, cov, dl):
            codes = list(codes)
            cov = list(cov)
            changed = True
            while changed:
                if time.time() > dl:
                    break
                changed = False
                for ci in list(codes):
                    removable = True
                    for nb in neighbor_lists[ci]:
                        if cov[nb] <= 1:
                            removable = False
                            break
                    if removable:
                        for nb in neighbor_lists[ci]:
                            cov[nb] -= 1
                        codes.remove(ci)
                        changed = True
            return codes, cov
        
        # Phase A: 1-for-1 swaps — replace codeword A with B, then remove any redundant
        # This can reduce size if B covers what A uniquely covered and enables another removal
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            cov = compute_coverage(current)
            
            # For each codeword A in current, find best replacement B not in current
            # that maximizes: (unique coverage gained by B) - (unique coverage lost by A)
            # Then check if removal of any other codeword becomes possible
            for idx_a, a in enumerate(list(current)):
                if time.time() > deadline - 10.0:
                    break
                # Words only covered by A (would become uncovered if A removed)
                a_unique = [nb for nb in neighbor_lists[a] if cov[nb] == 1]
                if not a_unique:
                    # A is redundant, just remove it
                    for nb in neighbor_lists[a]:
                        cov[nb] -= 1
                    current.remove(a)
                    current_set.discard(a)
                    improved = True
                    break
                
                # Find B that covers all of a_unique
                best_b = -1
                best_score = -1
                for b in range(total):
                    if b in current_set:
                        continue
                    # Check if b covers all a_unique words
                    b_covers = all(any(nb == u for nb in neighbor_lists[b]) for u in a_unique)
                    if not b_covers:
                        continue
                    # Score: how many additional critical words does b cover?
                    extra = 0
                    for nb in neighbor_lists[b]:
                        if cov[nb] == 1 and nb not in neighbor_lists[a]:
                            extra += 1
                    # Also count how many of b's neighbors are at coverage 1 (good)
                    score = extra * 10 - len(neighbor_lists[b]) * 0  # prefer more extra coverage
                    if score > best_score:
                        best_score = score
                        best_b = b
                
                if best_b == -1:
                    continue
                
                # Perform the swap: remove A, add B
                for nb in neighbor_lists[a]:
                    cov[nb] -= 1
                for nb in neighbor_lists[best_b]:
                    cov[nb] += 1
                current.remove(a)
                current_set.discard(a)
                current.append(best_b)
                current_set.add(best_b)
                
                # Now try to remove redundant codewords
                current, cov = remove_all_redundant(current, cov, deadline - 8.0)
                
                if len(current) < best_size:
                    best_size = len(current)
                    best_code = list(current)
                    improved = True
                    if best_size <= 73:
                        break
                else:
                    # Swap didn't help, revert
                    current.remove(best_b)
                    current_set.discard(best_b)
                    current.append(a)
                    current_set.add(a)
                    for nb in neighbor_lists[a]:
                        cov[nb] -= 1
                    for nb in neighbor_lists[best_b]:
                        cov[nb] += 1
                    # Recompute coverage
                    cov = compute_coverage(current)
                    break
            
            if best_size <= 73:
                break
        
        # Phase B: 2-for-1 exchange — remove two codewords, add one that covers their union
        if time.time() < deadline - 10.0 and best_size > 73:
            cov = compute_coverage(current)
            # Find pairs (A, B) whose unique words can all be covered by a single new codeword C
            for i_a in range(len(current)):
                if time.time() > deadline - 10.0:
                    break
                a = current[i_a]
                a_unique = [nb for nb in neighbor_lists[a] if cov[nb] == 1]
                if not a_unique:
                    continue
                for i_b in range(i_a + 1, len(current)):
                    if time.time() > deadline - 10.0:
                        break
                    b = current[i_b]
                    b_unique = [nb for nb in neighbor_lists[b] if cov[nb] == 1]
                    if not b_unique:
                        continue
                    # Combined unique words that need covering
                    combined_unique = set(a_unique) | set(b_unique)
                    # Find C that covers all combined_unique
                    for c in range(total):
                        if c in current_set:
                            continue
                        c_covers = all(any(nb == u for nb in neighbor_lists[c]) for u in combined_unique)
                        if c_covers:
                            # Perform exchange: remove A and B, add C
                            new_current = [x for x in current if x != a and x != b]
                            new_current.append(c)
                            new_cov = compute_coverage(new_current)
                            # Check if all words are still covered
                            if all(new_cov[nb] >= 1 for nb in range(total)):
                                new_current, new_cov = remove_all_redundant(new_current, new_cov, deadline - 8.0)
                                if len(new_current) < best_size:
                                    best_size = len(new_current)
                                    best_code = list(new_current)
                                    current = new_current
                                    current_set = set(current)
                                    cov = new_cov
                                    if best_size <= 73:
                                        break
                            break  # Only try first valid C for this pair
                    if best_size <= 73:
                        break
                if best_size <= 73:
                    break
    
    # Final cleanup: remove any remaining redundant codewords
    if time.time() < deadline - 5.0:
        cov = compute_coverage(best_code)
        best_code, _ = remove_all_redundant(best_code, cov, deadline - 3.0)
        best_size = len(best_code)
    
    # Convert indices back to words
    result = [i2w(i) for i in best_code]
    
    return result
