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
    deadline = start_time + 115  # Leave 35s margin for return/cleanup
    
    # Precompute neighbors for all 6561 words using numpy for speed
    # Build all words as a numpy array of shape (6561, 8)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Build a mapping from word tuple to index
    # Using base-3 encoding for O(1) lookup instead of dict of tuples
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
    
    # Precompute neighbors as lists of ints
    neighbors = [None] * total
    for idx in range(total):
        w = idx_to_word(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = list(w)
                    w_new[pos] = val
                    nbrs.append(word_to_idx(tuple(w_new)))
        neighbors[idx] = nbrs
    
    # Precompute neighbor sets for O(1) membership and intersection
    neighbor_sets = [set(nbrs) for nbrs in neighbors]
    
    # Bit-parallel representation for fast gain calculation
    # Each word's neighborhood is a set of 17 indices (0..6560)
    # We need 6561 bits = 103 uint64 words
    num_blocks = (total + 63) // 64  # 103
    # neighbor_bits[i] is a list of num_blocks uint64s representing the mask for neighbors[i]
    neighbor_bits = [None] * total
    for i in range(total):
        blocks = [0] * num_blocks
        for nbr in neighbors[i]:
            blocks[nbr // 64] |= (1 << (nbr % 64))
        neighbor_bits[i] = blocks
    
    # Greedy construction with bit-parallel gain updates
    # covered_mask: list of uint64s, bit set if covered
    covered_mask = [0] * num_blocks
    
    def is_covered(idx):
        return (covered_mask[idx // 64] >> (idx % 64)) & 1
    
    def mark_covered(idx):
        covered_mask[idx // 64] |= (1 << (idx % 64))
    
    def count_uncovered_in_mask(blocks):
        # Count bits set in blocks that are NOT in covered_mask
        cnt = 0
        for b in range(num_blocks):
            # bits we want: blocks[b] AND NOT covered_mask[b]
            bits = blocks[b] & (~covered_mask[b] & ((1 << 64) - 1))
            cnt += bin(bits).count('1')
        return cnt
    
    code_list = []
    code_set = set()
    
    # Greedy phase: pick word with max uncovered coverage
    # We maintain gains[i] = number of uncovered words in neighbors[i]
    gains = [0] * total
    for i in range(total):
        gains[i] = len(neighbors[i])  # initially all 17
    
    while time.time() < deadline - 60:
        # Check if all covered
        all_covered = True
        for b in range(num_blocks):
            if covered_mask[b] != ((1 << 64) - 1):
                # Check if any bit is 0
                if b == num_blocks - 1:
                    # Last block might not be full
                    valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                    if covered_mask[b] != valid_mask:
                        all_covered = False
                        break
                else:
                    all_covered = False
                    break
        
        if all_covered:
            break
        
        # Find best candidate
        best_idx = -1
        best_gain = -1
        for i in range(total):
            if i in code_set:
                continue
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        # Add best_idx
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        # Update covered_mask and gains
        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
        
        # Update gains for all words in dist2 ball of best_idx
        # For efficiency, we update gains for all words whose neighborhood intersects neighbors[best_idx]
        # This is equivalent to words within distance 2 of best_idx
        # But simpler: for each word w, gains[w] -= |neighbors[w] intersect neighbors[best_idx]|
        # We can iterate over all w and compute intersection size using neighbor_sets
        # But that's O(6561) per step, which is fine for ~500 steps
        
        # Actually, let's just recompute gains for candidates that might have changed
        # The gain of word w changes only if neighbors[w] intersects neighbors[best_idx]
        # i.e., dist(w, best_idx) <= 2
        # We can precompute dist2_balls for efficiency, but for now let's do a simpler update
        
        # Simpler approach: just recompute gains for all non-code words every few steps
        # Or update incrementally. Let's do incremental update.
        
        # For each word w in the code's "influence" (dist <= 2), update gain
        # Precompute dist2_balls? That's 6561 * ~129 = 846k ints, fine.
        # But let's just iterate over all words and check intersection with neighbor_sets[best_idx]
        
        # Optimization: only update gains for words not in code
        # And we can use the fact that gains[w] = count of uncovered in neighbors[w]
        # When we cover neighbors[best_idx], for each w, gains[w] decreases by 
        # |neighbors[w] intersect neighbors[best_idx] intersect uncovered_before|
        # Since we mark covered immediately, we can compute:
        # decrease = |neighbors[w] intersect neighbors[best_idx]|  (but only for those just covered)
        # This is tricky. Let's just recompute gains for all words periodically.
        
        # For now, recompute gains for all non-code words
        # This is O(6561 * 17) = 111k ops per step, acceptable for 500 steps
        
        for w in range(total):
            if w in code_set:
                continue
            # Count uncovered in neighbors[w]
            cnt = 0
            for nbr in neighbors[w]:
                if not is_covered(nbr):
                    cnt += 1
            gains[w] = cnt
    
    # Check if we've covered everything
    # If not, continue greedy until covered or time runs out
    while time.time() < deadline - 30:
        all_covered = True
        for b in range(num_blocks):
            if b < num_blocks - 1:
                if covered_mask[b] != ((1 << 64) - 1):
                    all_covered = False
                    break
            else:
                valid_mask = (1 << (total % 64)) - 1 if total % 64 != 0 else ((1 << 64) - 1)
                if covered_mask[b] != valid_mask:
                    all_covered = False
                    break
        
        if all_covered:
            break
        
        # Find best candidate
        best_idx = -1
        best_gain = -1
        for i in range(total):
            if i in code_set:
                continue
            cnt = count_uncovered_in_mask(neighbor_bits[i])
            if cnt > best_gain:
                best_gain = cnt
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            mark_covered(nbr)
        
        # Update gains
        for w in range(total):
            if w in code_set:
                continue
            gains[w] = count_uncovered_in_mask(neighbor_bits[w])
    
    # Now do local search to minimize size
    # cover_count: how many codewords cover each word
    cover_count = [0] * total
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 5:
        # Phase 1: Try to remove words
        removed_any = False
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            c = code_list[i]
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
        
        if removed_any:
            no_improvement_count = 0
            continue
        
        # Phase 2: Try swaps using intersection-based candidate search
        if len(code_list) > 450:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered words (those with cover_count == 0)
            uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_words:
                # Safe removal
                code_list.pop(idx_c)
                code_set.discard(c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                no_improvement_count = 0
                continue
            
            # Find replacement: must be in intersection of neighbor_sets of all uncovered_words
            # Start with neighbors of first uncovered word, intersect with others
            candidate_set = neighbor_sets[uncovered_words[0]].copy()
            for u in uncovered_words[1:]:
                candidate_set &= neighbor_sets[u]
            
            # Remove c and existing code from candidates
            candidate_set.discard(c)
            candidate_set -= code_set
            
            found_swap = False
            for idx_cp in candidate_set:
                # Check if idx_cp covers all uncovered words (it should by construction)
                # Perform swap
                code_list[idx_c] = idx_cp
                code_set.discard(c)
                code_set.add(idx_cp)
                for nbr in neighbors[idx_cp]:
                    cover_count[nbr] += 1
                found_swap = True
                no_improvement_count = 0
                break
            
            if not found_swap:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                no_improvement_count += 1
        
        # Phase 3: 2-opt escape (remove two, add one) when stuck
        if no_improvement_count > max_no_improvement:
            if time.time() < deadline - 15:
                # Pick two random codewords to remove
                if len(code_list) >= 2:
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1 = code_list[i1]
                    c2 = code_list[i2]
                    
                    # Temporarily remove both
                    for nbr in neighbors[c1]:
                        cover_count[nbr] -= 1
                    for nbr in neighbors[c2]:
                        cover_count[nbr] -= 1
                    
                    # Find uncovered words
                    all_uncovered = set()
                    for nbr in neighbors[c1]:
                        if cover_count[nbr] == 0:
                            all_uncovered.add(nbr)
                    for nbr in neighbors[c2]:
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
                    
                    # Find replacement that covers all uncovered words
                    candidate_set = neighbor_sets[list(all_uncovered)[0]].copy()
                    for u in list(all_uncovered)[1:]:
                        candidate_set &= neighbor_sets[u]
                    
                    candidate_set.discard(c1)
                    candidate_set.discard(c2)
                    candidate_set -= code_set
                    
                    found = False
                    for idx_cp in candidate_set:
                        # Swap: remove c1, c2, add idx_cp
                        code_list.remove(c2)
                        code_list.remove(c1)
                        code_list.append(idx_cp)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        code_set.add(idx_cp)
                        for nbr in neighbors[idx_cp]:
                            cover_count[nbr] += 1
                        found = True
                        no_improvement_count = 0
                        break
                    
                    if not found:
                        # Revert
                        for nbr in neighbors[c1]:
                            cover_count[nbr] += 1
                        for nbr in neighbors[c2]:
                            cover_count[nbr] += 1
                        no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement * 2:
            # Aggressive restart: remove 10% and rebuild
            if time.time() < deadline - 20:
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, min(num_remove, len(code_list)))
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        code_list.remove(c)
                
                # Rebuild greedily
                while time.time() < deadline - 10:
                    # Check if all covered
                    all_covered = True
                    for idx in range(total):
                        if cover_count[idx] == 0:
                            all_covered = False
                            break
                    if all_covered:
                        break
                    
                    best_idx = -1
                    best_count = -1
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        cnt = sum(1 for nbr in neighbors[idx] if cover_count[nbr] == 0)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx
                    
                    if best_idx == -1 or best_count == 0:
                        break
                    
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    for nbr in neighbors[best_idx]:
                        cover_count[nbr] += 1
                
                no_improvement_count = 0
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
