import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Iterated Local Search (Greedy + Perturbation + Repair) to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors for all words as lists
    # For radius=1, neighbors are the word itself + words at Hamming distance 1
    # Size is 1 + n*(q-1)
    neighbors = [[] for _ in range(total_words)]
    if radius == 1:
        for i in range(total_words):
            neighbors[i].append(i)
            w = int_to_word(i, n, q)
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
    else:
        # Generic case (slower, but handles other params)
        for i in range(total_words):
            w = int_to_word(i, n, q)
            neighbors[i].append(i)
            # For simplicity in this specific task context, we assume radius is small.
            # If radius >= 2, we need to add neighbors.
            if radius >= 2:
                # This is expensive. For the specific task q=3, n=6, r=1, this block is skipped.
                # We implement a basic expansion for correctness on other inputs.
                seen = set(neighbors[i])
                current_level = list(neighbors[i])
                for _ in range(radius - 1):
                    next_level = []
                    for c in current_level:
                        wc = int_to_word(c, n, q)
                        for pos in range(n):
                            ov = wc[pos]
                            for nv in range(q):
                                if nv != ov:
                                    wc[pos] = nv
                                    n_idx = word_to_int(wc, q)
                                    if n_idx not in seen:
                                        seen.add(n_idx)
                                        next_level.append(n_idx)
                            wc[pos] = ov
                    current_level = next_level
                    neighbors[i].extend(next_level)
                # Deduplicate
                neighbors[i] = list(seen)

    # Build neighbor bitmasks for fast coverage checks
    # Use Python integers as bitmasks (arbitrary precision)
    def build_mask(indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        return mask

    neighbor_masks = [build_mask(neighbors[i]) for i in range(total_words)]
    
    # Initialize with a known good construction: Direct sum of tetracode
    initial_codewords = set()
    if q == 3 and n == 6 and radius == 1:
        # Tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    initial_codewords.add(idx)
    else:
        # Fallback: Greedy construction from scratch
        uncovered_mask = (1 << total_words) - 1
        while uncovered_mask:
            best_center = -1
            best_count = -1
            best_weight = float('inf')
            for candidate in range(total_words):
                count = (uncovered_mask & neighbor_masks[candidate]).bit_count()
                if count > best_count or (count == best_count and candidate < best_weight):
                    best_count = count
                    best_center = candidate
                    best_weight = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            uncovered_mask &= ~neighbor_masks[best_center]

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)
    
    def get_coverage_mask(codewords):
        """Calculate the coverage bitmask given a set of codewords."""
        mask = 0
        for c in codewords:
            mask |= neighbor_masks[c]
        return mask

    def compute_coverage_counts(codewords):
        """For each word, count how many codewords cover it. Returns array of length total_words."""
        counts = np.zeros(total_words, dtype=np.int32)
        for c in codewords:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    def is_valid(codewords):
        return get_coverage_mask(codewords) == (1 << total_words) - 1

    def local_search(codewords):
        """
        Aggressive local search:
        1. Remove redundant codewords.
        2. Try 2-removals/1-additions (remove 2, add 1) to reduce size by 1.
        3. Try 1-swaps (remove 1, add 1) to escape local minima.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant codewords
            counts = compute_coverage_counts(codewords)
            removed = False
            for c in list(codewords):
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    codewords.remove(c)
                    removed = True
                    improved = True
            if removed:
                continue # Restart loop with new counts

            # Phase 2: Try 2-removals / 1-additions (Reduce size by 1)
            codeword_list = list(codewords)
            random.shuffle(codeword_list)
            found_swap = False
            for i in range(len(codeword_list)):
                if time.time() > deadline:
                    return codewords
                c1 = codeword_list[i]
                if c1 not in codewords: continue
                for j in range(i + 1, len(codeword_list)):
                    c2 = codeword_list[j]
                    if c2 not in codewords: continue
                    
                    # Tentatively remove c1 and c2
                    test_set = set(codewords)
                    test_set.remove(c1)
                    test_set.remove(c2)
                    
                    covered = get_coverage_mask(test_set)
                    uncovered = ((1 << total_words) - 1) & ~covered
                    if not uncovered:
                        # We can remove both? That would be a reduction by 2.
                        codewords = test_set
                        improved = True
                        found_swap = True
                        break
                    
                    # Try to add one codeword to cover 'uncovered'
                    best_add = -1
                    best_gain = -1
                    # Iterate over all candidates
                    for candidate in range(total_words):
                        if candidate in test_set:
                            continue
                        gain = (uncovered & neighbor_masks[candidate]).bit_count()
                        if gain == uncovered.bit_count():
                            best_add = candidate
                            best_gain = gain
                            break
                        elif gain > best_gain:
                            best_gain = gain
                            best_add = candidate
                    
                    if best_add != -1 and best_gain == uncovered.bit_count():
                        test_set.add(best_add)
                        if is_valid(test_set):
                            codewords = test_set
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            
            if found_swap:
                continue # Restart loop

            # Phase 3: Try 1-swaps (Remove 1, Add 1)
            if random.random() < 0.5:
                 c_to_remove = random.choice(list(codewords))
                 test_set = set(codewords)
                 test_set.remove(c_to_remove)
                 covered = get_coverage_mask(test_set)
                 uncovered = ((1 << total_words) - 1) & ~covered
                 
                 if not uncovered:
                     pass
                 else:
                     best_add = -1
                     for candidate in range(total_words):
                         if candidate in test_set:
                             continue
                         gain = (uncovered & neighbor_masks[candidate]).bit_count()
                         if gain == uncovered.bit_count():
                             best_add = candidate
                             break
                     
                     if best_add != -1:
                         test_set.add(best_add)
                         if is_valid(test_set):
                             codewords = test_set
                             improved = True
                             break # Restart loop

        return codewords

    def greedy_repair(codewords):
        """Add codewords greedily to cover all uncovered words. Returns modified set."""
        covered = get_coverage_mask(codewords)
        uncovered = ((1 << total_words) - 1) & ~covered
        while uncovered and time.time() < deadline:
            best_center = -1
            best_count = -1
            best_weight = float('inf')
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                count = (uncovered & neighbor_masks[candidate]).bit_count()
                if count > best_count or (count == best_count and candidate < best_weight):
                    best_count = count
                    best_center = candidate
                    best_weight = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    def find_least_critical_codewords(codewords, k=5):
        """Find k codewords that are 'least critical'."""
        if len(codewords) <= k:
            return list(codewords)
        counts = compute_coverage_counts(codewords)
        criticality = []
        for c in codewords:
            min_count = min(counts[nb] for nb in neighbors[c])
            criticality.append((min_count, c))
        criticality.sort(key=lambda x: x[0])
        return [c for _, c in criticality[:k]]

    # Main ILS Loop
    current_codewords = set(best_codewords)
    
    while time.time() < deadline:
        # 1. Local Search (Reduction + Swaps)
        current_codewords = local_search(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick)
        perturbation_type = random.random()
        if perturbation_type < 0.5:
            k = random.randint(1, 2)
            if len(current_codewords) > k:
                least_critical = find_least_critical_codewords(current_codewords, k)
                for c in least_critical:
                    current_codewords.remove(c)
        elif perturbation_type < 0.8:
            if len(current_codewords) > 1:
                least_critical = find_least_critical_codewords(current_codewords, 1)
                to_remove = least_critical[0]
                nearby = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if nearby:
                    to_add = random.choice(nearby)
                    current_codewords.remove(to_remove)
                    current_codewords.add(to_add)
                else:
                    current_codewords.remove(to_remove)
        else:
            k = random.randint(2, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        current_codewords = greedy_repair(current_codewords)

    # Final polish
    final_codewords = set(best_codewords)
    final_codewords = local_search(final_codewords)
    if len(final_codewords) < best_size:
        best_size = len(final_codewords)
        best_codewords = set(final_codewords)
    
    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
