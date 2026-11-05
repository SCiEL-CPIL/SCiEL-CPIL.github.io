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
    deadline = start_time + 118  # Leave 32s margin for return/cleanup
    
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
        
        # Update gains for all non-code words
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
    
    # Precompute for each codeword the "redundancy score": sum of cover_counts of its neighbors
    # This helps identify codewords that are "most redundant" (their neighbors are well-covered by others)
    
    while time.time() < deadline - 5:
        # Phase 1: Peeling - remove codeword with highest redundancy
        # A codeword c is "redundant" if for all nbr in neighbors[c], cover_count[nbr] > 1
        # We want to remove the one where the neighbors are MOST covered (highest min cover_count or sum)
        
        # Efficiently find removable codewords and pick the "best" one
        # "Best" = one whose removal leaves the most robust coverage (maximize min cover_count of affected neighbors)
        # Or simpler: just remove any removable one, but prioritize those with high overlap
        
        removed_any = False
        
        # Score each codeword: sum of cover_count[nbr] for nbr in neighbors[c]
        # Higher score = more redundant (neighbors are heavily covered by others)
        
        best_c = -1
        best_score = -1
        best_min_cover = -1
        
        for c in code_list:
            # Check if removable
            removable = True
            min_cover = 100
            score = 0
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
                score += cover_count[nbr]
                if cover_count[nbr] < min_cover:
                    min_cover = cover_count[nbr]
            
            if not removable:
                continue
            
            # Prefer higher min_cover (more robust), then higher score
            if min_cover > best_min_cover or (min_cover == best_min_cover and score > best_score):
                best_min_cover = min_cover
                best_score = score
                best_c = c
        
        if best_c != -1:
            # Remove best_c
            code_list.remove(best_c)
            code_set.discard(best_c)
            for nbr in neighbors[best_c]:
                cover_count[nbr] -= 1
            removed_any = True
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
        
        if removed_any:
            continue
        
        # Phase 2: If no single removal possible, try swap
        # Pick a random codeword to try removing
        if len(code_list) > 0:
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
                continue
            
            # Find replacement: must be in intersection of neighbor_sets of all uncovered_words
            candidate_set = neighbor_sets[uncovered_words[0]].copy()
            for u in uncovered_words[1:]:
                candidate_set &= neighbor_sets[u]
            
            candidate_set.discard(c)
            candidate_set -= code_set
            
            found_swap = False
            for idx_cp in candidate_set:
                code_list[idx_c] = idx_cp
                code_set.discard(c)
                code_set.add(idx_cp)
                for nbr in neighbors[idx_cp]:
                    cover_count[nbr] += 1
                found_swap = True
                break
            
            if not found_swap:
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
        
        # Phase 3: 2-opt escape when stuck
        if time.time() < deadline - 15:
            if len(code_list) >= 2:
                i1, i2 = random.sample(range(len(code_list)), 2)
                c1 = code_list[i1]
                c2 = code_list[i2]
                
                for nbr in neighbors[c1]:
                    cover_count[nbr] -= 1
                for nbr in neighbors[c2]:
                    cover_count[nbr] -= 1
                
                all_uncovered = set()
                for nbr in neighbors[c1]:
                    if cover_count[nbr] == 0:
                        all_uncovered.add(nbr)
                for nbr in neighbors[c2]:
                    if cover_count[nbr] == 0:
                        all_uncovered.add(nbr)
                
                if not all_uncovered:
                    code_list.remove(c2)
                    code_list.remove(c1)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    continue
                
                candidate_set = neighbor_sets[list(all_uncovered)[0]].copy()
                for u in list(all_uncovered)[1:]:
                    candidate_set &= neighbor_sets[u]
                
                candidate_set.discard(c1)
                candidate_set.discard(c2)
                candidate_set -= code_set
                
                found = False
                for idx_cp in candidate_set:
                    code_list.remove(c2)
                    code_list.remove(c1)
                    code_list.append(idx_cp)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    code_set.add(idx_cp)
                    for nbr in neighbors[idx_cp]:
                        cover_count[nbr] += 1
                    found = True
                    break
                
                if not found:
                    for nbr in neighbors[c1]:
                        cover_count[nbr] += 1
                    for nbr in neighbors[c2]:
                        cover_count[nbr] += 1
    
    # Final polish: multiple passes of peeling
    if time.time() < deadline - 3:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            improved = False
            while True:
                best_c = -1
                best_min_cover = -1
                best_score = -1
                
                for c in code_list:
                    removable = True
                    min_cover = 100
                    score = 0
                    for nbr in neighbors[c]:
                        if cover_count[nbr] <= 1:
                            removable = False
                            break
                        score += cover_count[nbr]
                        if cover_count[nbr] < min_cover:
                            min_cover = cover_count[nbr]
                    
                    if not removable:
                        continue
                    
                    if min_cover > best_min_cover or (min_cover == best_min_cover and score > best_score):
                        best_min_cover = min_cover
                        best_score = score
                        best_c = c
                
                if best_c == -1:
                    break
                
                code_list.remove(best_c)
                code_set.discard(best_c)
                for nbr in neighbors[best_c]:
                    cover_count[nbr] -= 1
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
            
            if not improved:
                break
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
