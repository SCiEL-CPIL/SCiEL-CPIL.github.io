import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Optimized ILS with coverage-count based redundancy checks.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but keep buffer)
    
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

    # Precompute neighbors as lists of indices for the specific case
    # This allows faster iteration for coverage updates
    neighbors = [[] for _ in range(total_words)]
    neighbor_masks = [0] * total_words
    
    for i in range(total_words):
        w = int_to_word(i, n, q)
        mask = 1 << i
        neighbors[i].append(i)
        
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                        mask |= (1 << neighbor_idx)
                w[pos] = original_val
        neighbor_masks[i] = mask

    # For the specific case q=3, n=6, r=1
    if q == 3 and n == 6 and radius == 1:
        # Initial construction: Direct sum of tetracode (81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = set()
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    initial_codewords.add(idx)
        
        best_codewords = set(initial_codewords)
        best_size = len(best_codewords)
        
        # Helper to check if a set of codewords covers the whole space
        def is_complete(codewords):
            covered_mask = 0
            for c in codewords:
                covered_mask |= neighbor_masks[c]
            return (covered_mask & ((1 << total_words) - 1)) == ((1 << total_words) - 1)
        
        # Optimized Reduce using coverage counts
        def reduce_codewords(codewords):
            # Build coverage count array
            # cover_count[w] = number of codewords in 'codewords' that cover w
            cover_count = [0] * total_words
            for c in codewords:
                for w in neighbors[c]:
                    cover_count[w] += 1
            
            changed = True
            while changed:
                changed = False
                # We iterate over a copy of the list to allow removal
                for c in list(codewords):
                    # Check if c is redundant:
                    # For all w in neighbors[c], is cover_count[w] > 1?
                    redundant = True
                    for w in neighbors[c]:
                        if cover_count[w] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        # Remove c and update coverage counts
                        codewords.remove(c)
                        for w in neighbors[c]:
                            cover_count[w] -= 1
                        changed = True
                        break # Restart loop as state changed
            
            return codewords

        # Main ILS Loop
        current_codewords = set(best_codewords)
        
        # Initial reduction
        current_codewords = reduce_codewords(current_codewords)
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            if best_size == 73:
                # We found the best known. We can try to improve further or stop.
                # Given the lower bound is 71, 73 is the target.
                # Let's continue for a bit to see if we can find 72 (unlikely but possible if bound is wrong)
                # or just return quickly.
                pass

        while time.time() < deadline:
            # 1. Perturbation
            # Strategy: Remove k random codewords. k=1 or 2 usually works well.
            # We can also try "swap" perturbations but remove+repair is standard.
            k = random.randint(1, 2)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
            
            # 2. Repair (Greedy Add)
            # We need to cover all words.
            covered_mask = 0
            for c in current_codewords:
                covered_mask |= neighbor_masks[c]
            
            total_mask = (1 << total_words) - 1
            
            # Greedily add best centers until covered
            while (covered_mask & total_mask) != total_mask:
                if time.time() > deadline:
                    break
                
                best_center = -1
                best_count = -1
                
                # Iterate through all possible centers
                # Optimization: We can skip centers that are already in the set (they add 0 new coverage)
                # But we need to find the one that adds the MOST new coverage.
                
                # To speed this up, we can pre-compute the list of uncovered words?
                # uncovered = ~covered_mask & total_mask
                # For each candidate, count bits in (neighbor_masks[candidate] & uncovered)
                
                uncovered = ~covered_mask & total_mask
                
                for candidate in range(total_words):
                    if candidate in current_codewords:
                        continue
                    
                    # Count how many uncovered words this candidate covers
                    # new_cover = neighbor_masks[candidate] & uncovered
                    # count = new_cover.bit_count()
                    
                    # bit_count is fast.
                    count = (neighbor_masks[candidate] & uncovered).bit_count()
                    
                    if count > best_count:
                        best_count = count
                        best_center = candidate
                
                if best_center == -1 or best_count == 0:
                    # Should not happen if radius allows, but safety check
                    break
                
                current_codewords.add(best_center)
                covered_mask |= neighbor_masks[best_center]
            
            # 3. Reduce
            current_codewords = reduce_codewords(current_codewords)
            
            # 4. Update Best
            if len(current_codewords) < best_size:
                best_size = len(current_codewords)
                best_codewords = set(current_codewords)
                
                if best_size <= 71:
                    break # Found optimal or better than known bound
            
            # If we hit 73, we might want to keep searching for 72 for a while
            # But if we are stuck at 73 for many iterations, we might just return.
            # The deadline guard handles the termination.
        
        result = [int_to_word(c, n, q) for c in best_codewords]
        return result
    else:
        # Fallback for other parameters: Simple Greedy
        codewords = set()
        covered_mask = 0
        total_mask = (1 << total_words) - 1
        
        while covered_mask != total_mask:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                new_cover = neighbor_masks[candidate] & (~covered_mask & total_mask)
                count = new_cover.bit_count()
                if count > best_count:
                    best_count = count
                    best_center = candidate
            
            if best_center == -1:
                break
            
            codewords.add(best_center)
            covered_mask |= neighbor_masks[best_center]
            
        # Local search reduction (simple version for generic case)
        def reduce_codewords_generic(codewords):
            changed = True
            while changed:
                changed = False
                for c in list(codewords):
                    mask_others = 0
                    for x in codewords:
                        if x != c:
                            mask_others |= neighbor_masks[x]
                    if (mask_others & neighbor_masks[c]) == neighbor_masks[c]:
                        codewords.remove(c)
                        changed = True
                        break
            return codewords
        
        codewords = reduce_codewords_generic(codewords)
        
        result = [int_to_word(c, n, q) for c in codewords]
        return result
