import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
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
    
    # Strategy 4: Local search - try swapping codewords
    if time.time() < deadline - 10.0 and best_size > 73:
        # Try to replace one codeword with another to enable removal of another
        current = list(best_code)
        for _ in range(20):  # limited iterations
            if time.time() > deadline - 5.0:
                break
            
            # Compute coverage
            coverage = [0] * total
            for ci in current:
                for nb in neighbor_lists[ci]:
                    coverage[nb] += 1
            
            # Find critical words (coverage == 1)
            critical = [i for i in range(total) if coverage[i] == 1]
            
            if not critical:
                # No critical words, try to remove any codeword
                for ci in list(current):
                    can_remove = True
                    for nb in neighbor_lists[ci]:
                        if coverage[nb] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for nb in neighbor_lists[ci]:
                            coverage[nb] -= 1
                        current.remove(ci)
                        break
                continue
            
            # Try to add a codeword that covers critical words, then remove redundant
            # Find codeword that covers most critical words
            best_add = -1
            best_crit_covered = 0
            for ci in range(total):
                if ci in set(current):
                    continue
                crit_covered = sum(1 for c in critical if c in neighbor_lists[ci])
                if crit_covered > best_crit_covered:
                    best_crit_covered = crit_covered
                    best_add = ci
            
            if best_add == -1 or best_crit_covered == 0:
                break
            
            # Add best_add
            current.append(best_add)
            for nb in neighbor_lists[best_add]:
                coverage[nb] += 1
            
            # Now try to remove redundant codewords
            removed_any = True
            while removed_any:
                removed_any = False
                for ci in list(current):
                    can_remove = True
                    for nb in neighbor_lists[ci]:
                        if coverage[nb] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for nb in neighbor_lists[ci]:
                            coverage[nb] -= 1
                        current.remove(ci)
                        removed_any = True
                        break
            
            if len(current) < best_size:
                best_size = len(current)
                best_code = list(current)
                if best_size <= 73:
                    break
    
    # Convert indices back to words
    result = [i2w(i) for i in best_code]
    
    return result
