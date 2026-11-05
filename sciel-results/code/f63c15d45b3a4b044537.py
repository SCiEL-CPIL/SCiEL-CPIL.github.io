import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (120s limit, 8s buffer)

    # Specific case optimization
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors: each word has 13 neighbors (itself + 12 at dist 1)
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            neighbors[i, 0] = i
            k = 1
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        # Greedy Set Cover - pick word covering most uncovered
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        coverage_counts = np.zeros(total, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() > deadline - 20:
                break
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = np.argmax(counts)
            if counts[best_c] == 0:
                break
            codewords.append(int(best_c))
            uncovered[neighbors[best_c]] = 0
            coverage_counts[neighbors[best_c]] += 1
            
        # Ensure valid coverage
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                if c not in codewords:
                    codewords.append(c)
                    uncovered[neighbors[c]] = 0
                    coverage_counts[neighbors[c]] += 1

        best_codewords = list(codewords)
        best_size = len(codewords)
        
        # Local Search: try to reduce code size
        # 1. Remove redundant codewords (those whose neighbors are all covered >= 2 times)
        def try_remove():
            for idx, c in enumerate(codewords):
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    return True
            return False

        # 2. Swap: remove one codeword, add another that covers the same critical words
        def try_swap():
            if len(codewords) == 0:
                return False
            idx_out = random.randint(0, len(codewords) - 1)
            c_out = codewords[idx_out]
            
            # Words that would become uncovered if we remove c_out
            crit = neighbors[c_out][coverage_counts[neighbors[c_out]] == 1]
            if len(crit) == 0:
                return False  # Should have been caught by try_remove
            
            # Find candidates c_in that cover all critical words
            # c_in must be in intersection of N(crit)
            valid_in = np.ones(total, dtype=bool)
            for w in crit:
                valid_in &= np.isin(w, neighbors)
            
            # Exclude existing codewords and c_out
            for c in codewords:
                valid_in[c] = False
            
            candidates = np.where(valid_in)[0]
            if len(candidates) > 0:
                c_in = int(random.choice(candidates))
                codewords[idx_out] = c_in
                coverage_counts[neighbors[c_out]] -= 1
                coverage_counts[neighbors[c_in]] += 1
                return True
            return False

        # 3. 2-swap: remove 2 codewords, add 2 different ones
        def try_2_swap():
            if len(codewords) < 2:
                return False
            idxs = random.sample(range(len(codewords)), 2)
            c1, c2 = codewords[idxs[0]], codewords[idxs[1]]
            
            # Temporarily remove
            coverage_counts[neighbors[c1]] -= 1
            coverage_counts[neighbors[c2]] -= 1
            
            # Find words that are now uncovered
            critical_words = np.where(coverage_counts == 0)[0]
            
            if len(critical_words) == 0:
                # No new uncovered words, just need to add 2 random non-existing words
                existing_set = set(codewords)
                existing_set.discard(c1)
                existing_set.discard(c2)
                non_existing = [c for c in range(total) if c not in existing_set]
                if len(non_existing) >= 2:
                    c3, c4 = random.sample(non_existing, 2)
                    codewords[idxs[0]] = c3
                    codewords[idxs[1]] = c4
                    coverage_counts[neighbors[c3]] += 1
                    coverage_counts[neighbors[c4]] += 1
                    return True
                coverage_counts[neighbors[c1]] += 1
                coverage_counts[neighbors[c2]] += 1
                return False
            
            # Need to cover critical_words with 2 new codewords
            # Try all pairs of candidates from neighbors of first critical word
            w0 = critical_words[0]
            c3_candidates = neighbors[w0]
            
            for c3 in c3_candidates:
                if c3 in codewords or c3 == c1 or c3 == c2:
                    continue
                
                # Check which critical words are covered by c3
                covered_by_c3 = set(neighbors[c3])
                remaining = [w for w in critical_words if w not in covered_by_c3]
                
                if len(remaining) == 0:
                    # c3 covers all critical words, c4 can be any non-existing word
                    existing_set = set(codewords)
                    existing_set.discard(c1)
                    existing_set.discard(c2)
                    non_existing = [c for c in range(total) if c not in existing_set and c != c3]
                    if len(non_existing) > 0:
                        c4 = random.choice(non_existing)
                        codewords[idxs[0]] = c3
                        codewords[idxs[1]] = c4
                        coverage_counts[neighbors[c3]] += 1
                        coverage_counts[neighbors[c4]] += 1
                        return True
                else:
                    # Find c4 that covers all remaining critical words
                    valid_c4 = np.ones(total, dtype=bool)
                    for w in remaining:
                        valid_c4 &= np.isin(w, neighbors)
                    
                    for c in codewords:
                        valid_c4[c] = False
                    valid_c4[c1] = False
                    valid_c4[c2] = False
                    valid_c4[c3] = False
                    
                    c4_candidates = np.where(valid_c4)[0]
                    if len(c4_candidates) > 0:
                        c4 = int(random.choice(c4_candidates))
                        codewords[idxs[0]] = c3
                        codewords[idxs[1]] = c4
                        coverage_counts[neighbors[c3]] += 1
                        coverage_counts[neighbors[c4]] += 1
                        return True
            
            # Revert
            coverage_counts[neighbors[c1]] += 1
            coverage_counts[neighbors[c2]] += 1
            return False

        # Main local search loop
        stuck_count = 0
        max_stuck = 500
        
        while time.time() < deadline - 5:
            # 1. Try removal
            if try_remove():
                stuck_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
                
            # 2. Try swap
            if try_swap():
                stuck_count = 0
                continue
                
            # 3. Stuck - try 2-swap
            stuck_count += 1
            if stuck_count > max_stuck:
                if try_2_swap():
                    stuck_count = 0
                    max_stuck = random.randint(300, 800)
                else:
                    stuck_count = 0
                    max_stuck = random.randint(300, 800)

        # Final cleanup: remove any remaining redundant words
        while True:
            if not try_remove():
                break
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)

        # Return the best found
        if len(best_codewords) < len(codewords):
            final_indices = best_codewords
        else:
            final_indices = codewords
            
        return [idx_to_list(idx) for idx in final_indices]

    else:
        # Fallback for other parameters: simple greedy
        total = q ** n
        powers = [q ** p for p in range(n)]
        
        def idx_to_list(idx):
            return [(idx // powers[p]) % q for p in range(n)]
        
        # Precompute neighbors for radius
        neighbors = []
        for i in range(total):
            d = [(i // powers[p]) % q for p in range(n)]
            nb = [i]
            for p in range(n):
                for alt in range(q):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        nb.append(new_i)
            neighbors.append(nb)
        
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            counts = np.zeros(total, dtype=np.int32)
            for nb_list in neighbors:
                counts[nb_list[0]] += uncovered[nb_list].sum()
            best_c = int(np.argmax(counts))
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
        
        return [idx_to_list(idx) for idx in codewords]

# EVOLVE-BLOCK-END
