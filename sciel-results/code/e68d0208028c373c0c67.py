import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # For the specific task q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback to direct sum for other params
        # General direct sum is hard to generalize simply for arbitrary q,n
        # But we can use a simple greedy for general cases
        pass

    # Optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        start_time = time.time()
        deadline = start_time + 115  # safety margin before 150s kill

        # Precompute neighbors as bitmasks
        powers = [1, 3, 9, 27, 81, 243]
        
        neighbors = []
        for x in range(729):
            nb_set = {x}
            for i in range(n):
                digit = (x // powers[i]) % q
                if digit != 0:
                    nb_set.add(x - powers[i])
                if digit != q - 1:
                    nb_set.add(x + powers[i])
            mask = 0
            for nb in nb_set:
                mask |= (1 << nb)
            neighbors.append(mask)
        
        all_mask = (1 << 729) - 1
        
        best_code = None
        best_size = 81  # Initialize with known upper bound

        # Local search polish: try to remove a codeword or swap to reduce size
        def polish(code_indices, neighbors_list, all_mask):
            """Try to reduce code size via removal and swap operations."""
            code_set = set(code_indices)
            # Build coverage count array
            # For each position, count how many codewords cover it
            cover_count = [0] * 729
            for c in code_set:
                nb = neighbors_list[c]
                # Count bits in nb
                # We need to increment cover_count for each bit set in nb
                # This is expensive; instead, we'll use a different approach
                # Actually, let's just track which positions are covered by which codewords
                pass
            
            # Simpler approach: repeatedly try to remove a codeword if all its
            # covered positions are still covered by others
            changed = True
            while changed:
                changed = False
                # Build coverage: for each position, which codewords cover it
                # Use a list of sets or bitmasks
                # For efficiency, use numpy or just iterate
                # Let's use a dictionary: pos -> count of covering codewords
                cover_counts = {}
                for c in code_set:
                    nb = neighbors_list[c]
                    # Iterate through set bits
                    tmp = nb
                    while tmp:
                        # Get lowest set bit
                        lsb = tmp & (-tmp)
                        pos = lsb.bit_length() - 1
                        cover_counts[pos] = cover_counts.get(pos, 0) + 1
                        tmp ^= lsb
                
                # Try removing each codeword
                for c in list(code_set):
                    nb = neighbors_list[c]
                    # Check if all positions in nb are covered by at least 2 codewords
                    removable = True
                    tmp = nb
                    while tmp:
                        lsb = tmp & (-tmp)
                        pos = lsb.bit_length() - 1
                        if cover_counts.get(pos, 0) < 2:
                            removable = False
                            break
                        tmp ^= lsb
                    
                    if removable:
                        code_set.discard(c)
                        # Update cover_counts
                        tmp = nb
                        while tmp:
                            lsb = tmp & (-tmp)
                            pos = lsb.bit_length() - 1
                            cover_counts[pos] -= 1
                            tmp ^= lsb
                        changed = True
                        break  # Restart since coverage changed
            
            return list(code_set)

        # Run multiple Greedy instances with vectorized overlap
        for trial in range(2000):
            if time.time() > deadline - 20:  # Leave time for polish
                break
            
            uncovered = all_mask
            code = []
            
            while uncovered:
                # Find best candidate using vectorized approach
                # We need to compute overlap of each neighbor mask with uncovered
                # This is the bottleneck - do it efficiently
                # For 729 words, we can iterate but it's slow
                # Alternative: precompute neighbor positions as lists
                
                max_cov = -1
                best_cands = []
                for w in range(729):
                    overlap = (neighbors[w] & uncovered).bit_count()
                    if overlap > max_cov:
                        max_cov = overlap
                        best_cands = [w]
                    elif overlap == max_cov:
                        best_cands.append(w)
                        
                if not best_cands:
                    break
                chosen = random.choice(best_cands)
                code.append(chosen)
                uncovered &= ~neighbors[chosen]
                
            # Polish the result
            polished = polish(code, neighbors, all_mask)
            if len(polished) < best_size:
                best_size = len(polished)
                best_code = polished.copy()
            
            if best_size <= 73:
                if trial > 300:
                    break

        # Final polish on best code if time permits
        if best_code and time.time() < deadline - 5:
            final_polished = polish(best_code, neighbors, all_mask)
            if len(final_polished) < best_size:
                best_size = len(final_polished)
                best_code = final_polished

        if best_code:
            res = []
            for idx in best_code:
                word = []
                temp = idx
                for i in range(n):
                    word.append(temp % q)
                    temp //= q
                res.append(word)
            return res
        else:
            # Fallback to tetracode construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Generic fallback for other parameters
    # Simple greedy
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    neighbors = []
    for x in range(total_words):
        nb_set = {x}
        for i in range(n):
            digit = (x // powers[i]) % q
            for delta in range(1, radius + 1):
                if digit - delta >= 0:
                    nb_set.add(x - delta * powers[i])
                if digit + delta < q:
                    nb_set.add(x + delta * powers[i])
        mask = 0
        for nb in nb_set:
            mask |= (1 << nb)
        neighbors.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words
    
    for trial in range(100):
        if time.time() > deadline:
            break
        uncovered = all_mask
        code = []
        while uncovered:
            max_cov = -1
            best_cands = []
            for w in range(total_words):
                overlap = (neighbors[w] & uncovered).bit_count()
                if overlap > max_cov:
                    max_cov = overlap
                    best_cands = [w]
                elif overlap == max_cov:
                    best_cands.append(w)
            if not best_cands:
                break
            chosen = random.choice(best_cands)
            code.append(chosen)
            uncovered &= ~neighbors[chosen]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code:
        res = []
        for idx in best_code:
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            res.append(word)
        return res
    else:
        return [[0]*n]
