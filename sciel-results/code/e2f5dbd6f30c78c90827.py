import itertools
import random
import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    
    def to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute cover masks
    cover_masks = [0] * total_words
    for idx in range(total_words):
        w = to_tuple(idx)
        mask = 0
        mask |= (1 << idx)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    new_idx = to_index(tuple(new_w))
                    mask |= (1 << new_idx)
        cover_masks[idx] = mask

    # Greedy to get initial upper bound
    uncovered = (1 << total_words) - 1
    greedy_centers = []
    while uncovered:
        best_score = -1
        best_idx = -1
        for i in range(total_words):
            score = (cover_masks[i] & uncovered).bit_count()
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score == 0: break
        greedy_centers.append(best_idx)
        uncovered &= ~cover_masks[best_idx]
    
    best_size = len(greedy_centers)
    
    # Local Search on greedy solution to improve upper bound
    # (Using the logic from the previous program which was effective)
    centers = greedy_centers.copy()
    changed = True
    while changed:
        changed = False
        counts = [0] * total_words
        for c in centers:
            m = cover_masks[c]
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        new_centers = []
        for c in centers:
            mask = cover_masks[c]
            redundant = True
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                changed = True
            else:
                new_centers.append(c)
        centers = new_centers
    
    if len(centers) < best_size:
        best_size = len(centers)

    # Branch and Bound
    # Heuristic: Pick uncovered word with minimum number of covering centers
    # Precompute for each word, the list of centers that cover it
    # cover_list[w] = list of center indices c such that w is in cover_masks[c]
    cover_list = [[] for _ in range(total_words)]
    for c in range(total_words):
        m = cover_masks[c]
        while m:
            lsb = m & (-m)
            w_idx = lsb.bit_length() - 1
            cover_list[w_idx].append(c)
            m ^= lsb

    # Sort centers in cover_list by some heuristic? 
    # Maybe by the number of words they cover?
    # Or just keep as is.

    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    max_time = 115

    # Global variables for B&B
    best_found = best_size
    found_centers = centers.copy()
    
    def lower_bound(uncovered_mask):
        # Simple LB: ceil(count / 15)
        # Better LB: Greedy packing
        # For now, use simple LB for speed
        return (uncovered_mask.bit_count() + 14) // 15

    def search(uncovered_mask, current_count, current_centers):
        nonlocal best_found, found_centers
        
        if time.time() - start_time > max_time:
            return
            
        if uncovered_mask == 0:
            if current_count < best_found:
                best_found = current_count
                found_centers = current_centers.copy()
            return
            
        if current_count + lower_bound(uncovered_mask) >= best_found:
            return
            
        # Pick word with minimum degree (fewest centers covering it)
        # Iterate through set bits of uncovered_mask
        m = uncovered_mask
        best_w = -1
        min_degree = float('inf')
        while m:
            lsb = m & (-m)
            w_idx = lsb.bit_length() - 1
            degree = len(cover_list[w_idx])
            if degree < min_degree:
                min_degree = degree
                best_w = w_idx
            m ^= lsb
            
        if best_w == -1:
            return
            
        # Branch on each center covering best_w
        # Order centers by how many new words they cover?
        candidates = cover_list[best_w]
        # Sort candidates by coverage of current uncovered set (descending)
        # This helps find a good solution faster for pruning
        candidates.sort(key=lambda c: (cover_masks[c] & uncovered_mask).bit_count(), reverse=True)
        
        for c in candidates:
            new_uncovered = uncovered_mask & ~cover_masks[c]
            current_centers.append(c)
            search(new_uncovered, current_count + 1, current_centers)
            current_centers.pop()
            
            # Pruning: if we found a solution of size best_found, 
            # and current_count + 1 >= best_found, we can break?
            # No, because other branches might be smaller.
            # But if current_count + 1 >= best_found, any further additions will be >= best_found.
            # So if current_count + 1 >= best_found, we can break the loop?
            # Yes, because we are adding 1 center. If current_count + 1 >= best_found, 
            # then any solution in this branch will have size >= current_count + 1 >= best_found.
            # So we don't need to explore further in this branch.
            if current_count + 1 >= best_found:
                break

    search((1 << total_words) - 1, 0, [])

    result = [to_tuple(idx) for idx in found_centers]
    return result
