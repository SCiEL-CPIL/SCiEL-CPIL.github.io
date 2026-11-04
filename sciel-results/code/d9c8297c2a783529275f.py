import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute digits for all words as a numpy array (total_words, n)
    # This allows fast vectorized distance calculations
    digits_of = np.zeros((total_words, n), dtype=np.int8)
    for i in range(total_words):
        x = i
        for j in range(n):
            digits_of[i, j] = x % q
            x //= q
    
    # Precompute the "ball" of each word as a boolean mask or list of indices?
    # For n=8, q=3, total_words=6561. 
    # A ball of radius 1 has 1 + n*(q-1) = 1 + 8*2 = 17 words.
    # Storing balls as lists of indices is efficient for small radius.
    
    # Let's precompute neighbors for radius 1 specifically, as it's the target case.
    # If radius > 1, we fall back to a simpler method or extend, but the task is K_3(8,1).
    
    neighbors = [[] for _ in range(total_words)]
    if radius == 1:
        for i in range(total_words):
            neighbors[i].append(i)
            for pos in range(n):
                orig = digits_of[i, pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    # Calculate index of neighbor
                    # index = sum(digit[j] * q^(n-1-j))
                    # delta = (new_val - orig) * q^(n-1-pos)
                    idx = i + (new_val - orig) * (q ** (n - 1 - pos))
                    neighbors[i].append(idx)
    else:
        # General radius neighbors (slower to precompute, but n is small)
        for i in range(total_words):
            neighbors[i].append(i)
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        idx = i
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            if variation[k] != digits_of[i, pos]:
                                is_diff = True
                                idx += (variation[k] - digits_of[i, pos]) * (q ** (n - 1 - pos))
                        if is_diff:
                            neighbors[i].append(idx)
            
    # Helper to check if a set of codewords covers everything
    # Using a boolean array 'covered' of size total_words
    def check_cover(code_indices):
        covered = np.zeros(total_words, dtype=bool)
        for c in code_indices:
            for nb in neighbors[c]:
                covered[nb] = True
        return covered
    
    # Initial Construction: Tetracode Direct Sum
    # Tetracode is a (9,4,2) code over Z3. 
    # For n=8, we take Tetracode x Z3^4.
    # Size = 9 * 3^4 = 9 * 81 = 729.
    # This is a valid covering code for radius 1?
    # Tetracode has covering radius 1. Z3^4 has covering radius 1 (trivially, each word covers itself).
    # Direct sum of covering codes has covering radius equal to the sum of radii? 
    # Actually, if C1 covers A1 with radius r1 and C2 covers A2 with radius r2, 
    # then C1 x C2 covers A1 x A2 with radius r1 + r2.
    # Here r1=1, r2=0 (if we treat Z3^4 as a code of radius 0, it doesn't cover).
    # Wait, the direct sum construction for covering codes:
    # If C is a covering code of length n and radius R, and D is a covering code of length m and radius S,
    # then C x D is a covering code of length n+m and radius R+S.
    # Tetracode: (9, 4, 2) code, covering radius 1.
    # We need length 8. So we combine Tetracode (len 4) with something of len 4.
    # If we use the trivial code Z3^4 (all 81 words), its covering radius is 0.
    # So Tetracode x Z3^4 has covering radius 1 + 0 = 1.
    # Size = 9 * 81 = 729.
    # This is a valid starting point.
    
    tetracode_words = []
    for a in range(3):
        for b in range(3):
            tetracode_words.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    # Convert tetracode words to indices in Z3^4 space (0..80)
    tetracode_indices = []
    for w in tetracode_words:
        idx = 0
        for j in range(4):
            idx += w[j] * (3 ** (3 - j))
        tetracode_indices.append(idx)
        
    # Construct full code indices for n=8
    # Index in Z3^8 = idx_first_4 * 3^4 + idx_last_4
    initial_code = []
    for t_idx in tetracode_indices:
        for tail_idx in range(3**4):
            full_idx = t_idx * (3**4) + tail_idx
            initial_code.append(full_idx)
            
    # Convert to set for fast lookup
    current_code_set = set(initial_code)
    current_code_list = list(initial_code)
    
    # Verify initial cover
    covered = check_cover(current_code_list)
    if not np.all(covered):
        # Fallback if logic is wrong (shouldn't happen)
        # Just use greedy
        pass

    best_code = current_code_list[:]
    best_size = len(best_code)
    
    # Local Search: Greedy Removal with Repair
    # Strategy:
    # 1. Iterate through codewords.
    # 2. Try removing one.
    # 3. Find uncovered words.
    # 4. If none, accept removal.
    # 5. If some, try to find ONE codeword (not in code) that covers the most uncovered words.
    #    If it covers ALL uncovered words, accept swap (remove 1, add 1 -> size same, but maybe better structure? 
    #    Actually, size is same. We only care about size minimization. 
    #    However, swapping might allow further removals later. 
    #    But strictly, if size doesn't decrease, we should be careful. 
    #    Let's prioritize REMOVAL. If we can't remove, we might swap to enable removals later.
    #    For now, let's just try to reduce size.
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Shuffle to break symmetry
        random.shuffle(current_code_list)
        
        for i in range(len(current_code_list)):
            if time.time() > deadline - 10.0:
                break
                
            c_to_remove = current_code_list[i]
            
            # Simulate removal
            # We need to know which words become uncovered.
            # Instead of recalculating full cover, we can maintain a "coverage count" array.
            # Let's switch to a coverage count approach for efficiency.
            
            # This requires restructuring the loop to maintain counts.
            # Let's do that.
            pass
            
    # Let's implement the Coverage Count approach properly.
    # coverage_count[w] = number of codewords in current_code that cover w.
    
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for c in current_code_list:
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    uncovered_mask = (coverage_count == 0)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        random.shuffle(current_code_list)
        
        for i in range(len(current_code_list)):
            if time.time() > deadline - 5.0:
                break
                
            c = current_code_list[i]
            
            # Check if c is "essential" (i.e., removing it leaves some word with count 0)
            # We can quickly check: are there any neighbors of c with coverage_count == 1?
            # If yes, removing c will make them 0.
            # If no, removing c is safe.
            
            is_essential = False
            newly_uncovered = []
            
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    is_essential = True
                    newly_uncovered.append(nb)
                # If coverage_count[nb] > 1, it remains covered.
            
            if not is_essential:
                # Safe to remove
                current_code_list.pop(i)
                current_code_set.remove(c)
                
                for nb in neighbors[c]:
                    coverage_count[nb] -= 1
                    
                improved = True
                best_code = current_code_list[:]
                best_size = len(best_code)
                # We don't need to re-check everything, just continue
                continue
            
            # If essential, can we repair?
            # We need to find a codeword c_add such that it covers all words in newly_uncovered.
            # And ideally, c_add is not already in the code.
            # If we add c_add, size stays same (remove 1, add 1). 
            # This doesn't reduce size, but might change the structure to allow future removals.
            # However, this is a "neutral" move in terms of objective (size).
            # Let's try to find if there is a c_add that covers the union of newly_uncovered.
            
            if newly_uncovered:
                # Find candidate c_add
                # We can sample random candidates or check all? 
                # Checking all 6561 is expensive. 
                # Let's try a few random ones.
                # Or, better: check if any codeword in the current code (other than c) covers these? 
                # No, they don't, that's why they are uncovered.
                
                # Let's try to find a codeword from the entire space that covers all newly_uncovered.
                # This is a set cover problem on a small set (size <= 17).
                # We can iterate through all possible codewords? 6561 * 17 ops = ~100k ops. 
                # This might be too slow inside the inner loop.
                
                # Optimization: Only try to repair if the number of newly_uncovered is small.
                if len(newly_uncovered) <= 3:
                    # Try to find a repair
                    # We can precompute for each set of uncovered words? No, too many sets.
                    
                    # Let's just try a random search for repair.
                    found_repair = False
                    for _ in range(10): # Try 10 random candidates
                        c_add = random.randint(0, total_words - 1)
                        if c_add in current_code_set:
                            continue
                            
                        covers_all = True
                        for w in newly_uncovered:
                            # Check if c_add covers w
                            # dist(c_add, w) <= radius
                            # Using digits_of
                            # dist = sum(digits_of[c_add, j] != digits_of[w, j])
                            # This is vectorized if we do it for all w at once?
                            # Let's just do it simply for small count.
                            diff = np.sum(digits_of[c_add] != digits_of[w])
                            if diff > radius:
                                covers_all = False
                                break
                            
                        if covers_all:
                            # Accept swap
                            current_code_list[i] = c_add
                            current_code_set.remove(c)
                            current_code_set.add(c_add)
                            
                            # Update coverage counts
                            for nb in neighbors[c]:
                                coverage_count[nb] -= 1
                            for nb in neighbors[c_add]:
                                coverage_count[nb] += 1
                                
                            improved = True
                            found_repair = True
                            break
                    
                    if found_repair:
                        continue
                        
    # Final verification
    final_cover = check_cover(best_code)
    if np.all(final_cover):
        return [list(digits_of[i]) for i in best_code]
    else:
        # Fallback
        return [list(digits_of[i]) for i in initial_code]
