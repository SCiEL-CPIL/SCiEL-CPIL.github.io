import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute neighbors: neighbors[w] = list of 17 indices (w itself + 16 neighbors)
    neighbors = np.zeros((total, 17), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors[w, idx] = new_w
                idx += 1
    
    # Build membership: for each word, which codeword indices cover it
    # We'll use a sparse representation: cover_map[word] = set of selected indices that cover it
    # But for efficiency, we'll track cover_count and use neighbors to check coverage
    
    # Tetracode: optimal (3,4,1) covering code with 9 words
    tetracode = []
    for a in range(3):
        for b in range(3):
            word = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            tetracode.append(word)
    
    # Convert tetracode to indices (4-digit ternary)
    tetracode_indices = []
    for word in tetracode:
        idx = 0
        for d in word:
            idx = idx * 3 + d
        tetracode_indices.append(idx)
    
    # Precompute: for each 4-digit prefix, the list of full indices covered by tetracode words
    # with that prefix in positions 4-7
    # Actually, let's think differently.
    
    # Partition approach:
    # Split 8 positions into two groups of 4: positions 0-3 (prefix) and positions 4-7 (suffix)
    # For each of the 3^4 = 81 possible prefixes, we can place codewords.
    # A codeword with prefix P and suffix S covers all words (P', S') where:
    #   - P' differs from P in at most 1 position AND S' = S, OR
    #   - P' = P AND S' differs from S in at most 1 position
    # Wait, that's not right. A codeword (P, S) covers words within Hamming distance 1,
    # which means words that differ in at most 1 position total.
    
    # Better approach: Use the tetracode structure.
    # The tetracode is a (3,4,1) covering code: 9 words cover all 3^4 = 81 words within distance 1.
    # We can use a direct product-like construction:
    #   Codewords: for each pair (t1, t2) where t1, t2 are tetracode words,
    #   form codeword (t1, t2) in 8 dimensions.
    #   This gives 9*9 = 81 codewords.
    #   But does this cover all 3^8 words within distance 1?
    #   A word (x, y) where x, y are 4-digit ternary is covered if there exist t1, t2 such that
    #   d(x, t1) + d(y, t2) <= 1.
    #   This means either d(x,t1)=0 and d(y,t2)<=1, or d(x,t1)<=1 and d(y,t2)=0.
    #   Since tetracode covers all 4-digit words within distance 1, for any x there exists t1 with d(x,t1)<=1,
    #   and for any y there exists t2 with d(y,t2)<=1.
    #   But we need d(x,t1)+d(y,t2)<=1, which is stronger.
    #   So the direct product doesn't work directly.
    
    # Instead, let's use a smarter partition:
    # For each prefix P (4 digits), select a set of suffixes S such that:
    #   1. All words with prefix P are covered by codewords with prefix P (suffix within distance 1)
    #      OR by codewords with prefix P' where d(P,P')=1 and same suffix.
    #   2. This is complex. Let's use a different strategy.
    
    # Strategy: Start with the tetracode product (81 codewords), then greedily add/remove
    # to achieve full coverage, then local search.
    
    def create_tetracode_product_init():
        """Create initial set using tetracode product structure."""
        init = []
        for t1_idx in tetracode_indices:
            for t2_idx in tetracode_indices:
                # Convert to full 8-digit index
                full_idx = t1_idx * (3 ** 4) + t2_idx
                init.append(full_idx)
        return init
    
    # Helper: greedy cover from a starting set
    def greedy_cover_from(start_selected, cover_count, uncovered_mask, uncovered_count):
        selected = list(start_selected)
        while uncovered_count > 0:
            # Count how many uncovered words each candidate covers
            counts = uncovered_mask[neighbors].sum(axis=1)
            
            max_count = np.max(counts)
            
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            ball = neighbors[best_idx]
            was_uncovered = uncovered_mask[ball]
            num_newly_covered = np.sum(was_uncovered)
            uncovered_count -= num_newly_covered
            
            cover_count[ball] += 1
            uncovered_mask[ball] = False
            
            if time.time() > deadline - 5.0:
                break
        
        return selected, cover_count, uncovered_mask, uncovered_count
    
    # Helper: local search to improve
    def local_search(selected, cover_count):
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            
            # Pass 1: Removal - remove codewords that are redundant
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            for i in range(len(selected)):
                cw = selected[i]
                if min_counts[cw] > 1:
                    selected.pop(i)
                    cover_count[neighbors[cw]] -= 1
                    changed = True
                    break
            
            if not changed:
                # Try swaps
                if time.time() < deadline - 10.0 and len(selected) > 486:
                    for _ in range(10):
                        if time.time() > deadline - 5.0:
                            break
                        
                        idx_to_swap = np.random.randint(len(selected))
                        cw_removed = selected[idx_to_swap]
                        
                        # Find words that would become uncovered
                        U = []
                        for w in neighbors[cw_removed]:
                            if cover_count[w] == 1:
                                U.append(w)
                        
                        U = np.array(U, dtype=np.int32)
                        
                        if len(U) == 0:
                            selected.pop(idx_to_swap)
                            cover_count[neighbors[cw_removed]] -= 1
                            changed = True
                            break
                        
                        selected_set = set(selected)
                        replacement_found = False
                        
                        # Find a replacement codeword that covers all words in U
                        for cw2 in neighbors[U[0]]:
                            if cw2 == cw_removed:
                                continue
                            if cw2 in selected_set:
                                continue
                            
                            covers_all = True
                            for w in U:
                                if w not in neighbors[cw2]:
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                selected[idx_to_swap] = cw2
                                cover_count[neighbors[cw_removed]] -= 1
                                cover_count[neighbors[cw2]] += 1
                                replacement_found = True
                                changed = True
                                break
                        
                        if replacement_found:
                            break
        
        return selected, cover_count
    
    # Cross-subcube pruning: given a selected set, remove codewords whose ball is
    # fully covered by other codewords in the same or adjacent subcubes
    def cross_subcube_prune(selected, cover_count):
        """Remove redundant codewords using partition-aware pruning."""
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            
            # For each codeword, check if its ball is fully covered by others
            min_counts = np.min(cover_count[neighbors], axis=1)
            
            # Find codewords that are redundant (all words in their ball have count > 1)
            redundant_candidates = [i for i, cw in enumerate(selected) if min_counts[cw] > 1]
            
            if redundant_candidates:
                # Remove one at a time, preferring those with highest min count
                redundant_candidates.sort(key=lambda i: min_counts[selected[i]], reverse=True)
                for i in redundant_candidates:
                    if i >= len(selected):
                        continue
                    cw = selected[i]
                    if min_counts[cw] > 1:
                        selected.pop(i)
                        cover_count[neighbors[cw]] -= 1
                        changed = True
                        break
        
        return selected, cover_count
    
    best_size = total
    best_set = None
    
    # Main loop
    while time.time() < deadline - 2.0:
        # 1. Tetracode product initialization (81 codewords)
        init_selected = create_tetracode_product_init()
        
        # Compute initial coverage
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        for cw in init_selected:
            ball = neighbors[cw]
            cover_count[ball] += 1
            uncovered_mask[ball] = False
        
        uncovered_count = int(np.sum(uncovered_mask))
        
        # 2. Greedy completion
        selected, cover_count, uncovered_mask, uncovered_count = greedy_cover_from(
            init_selected, cover_count, uncovered_mask, uncovered_count
        )
        
        if uncovered_count > 0:
            # Greedy didn't finish, try plain greedy from scratch
            cover_count = np.zeros(total, dtype=np.int32)
            uncovered_mask = np.ones(total, dtype=bool)
            selected = []
            uncovered_count = total
            
            while uncovered_count > 0 and time.time() < deadline - 5.0:
                counts = uncovered_mask[neighbors].sum(axis=1)
                max_count = np.max(counts)
                if max_count == 0:
                    break
                candidates = np.where(counts == max_count)[0]
                best_idx = candidates[np.random.randint(len(candidates))]
                selected.append(best_idx)
                ball = neighbors[best_idx]
                was_uncovered = uncovered_mask[ball]
                num_newly_covered = np.sum(was_uncovered)
                uncovered_count -= num_newly_covered
                cover_count[ball] += 1
                uncovered_mask[ball] = False
            
            if uncovered_count > 0:
                continue
        
        # 3. Cross-subcube pruning
        selected, cover_count = cross_subcube_prune(selected, cover_count)
        
        # 4. Local search
        selected, cover_count = local_search(selected, cover_count)
        
        # 5. Final pruning pass
        selected, cover_count = cross_subcube_prune(selected, cover_count)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break
    
    # Convert best_set to list of digit lists
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n).ljust(n, '0')) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
