"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a local search algorithm starting from a seeded greedy solution to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words in the space
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

    # Precompute neighbors (ball of radius 1 for q=3, n=6 is small)
    # Note: The task specifies radius=1 in the description, but the function takes radius.
    # For general radius, precomputing is harder. But for this specific task (6,1), it's fast.
    # We will assume radius is small (1 or 2) for efficiency.
    
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        # If radius > 1, we'd need more, but for this task radius=1.
        # If radius is larger, the precomputation might be too slow for large n, 
        # but n=6 is small.

    # Initial Code: Greedy or Seeded
    # We use the seeded 81-word construction as a strong start.
    codewords = set()
    covered_count = [0] * total_words # Number of codewords covering each word
    
    def add_codeword(c_idx):
        codewords.add(c_idx)
        for nb in neighbors[c_idx]:
            covered_count[nb] += 1

    def remove_codeword(c_idx):
        codewords.remove(c_idx)
        for nb in neighbors[c_idx]:
            covered_count[nb] -= 1

    def is_valid():
        # Check if all words have covered_count > 0
        # This is O(total_words). 
        # Optimization: Keep track of uncovered count.
        pass

    # To make is_valid fast, we maintain `uncovered_count`.
    # Let's restart the state management to be cleaner.
    
    covered = [0] * total_words
    uncovered_count = total_words
    
    def add(c_idx):
        nonlocal uncovered_count
        codewords.add(c_idx)
        for nb in neighbors[c_idx]:
            if covered[nb] == 0:
                uncovered_count -= 1
            covered[nb] += 1

    def remove(c_idx):
        nonlocal uncovered_count
        codewords.remove(c_idx)
        for nb in neighbors[c_idx]:
            covered[nb] -= 1
            if covered[nb] == 0:
                uncovered_count += 1

    # 1. Initialize with Seed
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    add(idx)
    else:
        # Fallback: Greedy
        uncovered_set = set(range(total_words))
        while uncovered_set:
            if time.time() > deadline:
                break
            best_center = -1
            best_count = -1
            # Randomize start to break symmetry? 
            # For now, standard greedy
            for candidate in range(total_words):
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered_set:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1: break
            add(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered_set:
                    uncovered_set.remove(nb)

    # 2. Local Search
    # Goal: Minimize len(codewords)
    # Moves:
    # 1. Delete: If removing a codeword keeps uncovered_count == 0, do it.
    # 2. Swap: Replace c with c'. If valid, do it.
    # 3. Perturb: Add random, then greedy delete.
    
    best_size = len(codewords)
    best_codewords = set(codewords)
    
    no_improve_count = 0
    max_no_improve = 500 # Adjust based on speed
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to delete redundant codewords
        # We can iterate through a random permutation of codewords
        for c in list(codewords):
            if time.time() > deadline:
                break
            # Check if c is redundant
            # c is redundant if for all nb in neighbors[c], covered[nb] > 1
            redundant = True
            for nb in neighbors[c]:
                if covered[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                remove(c)
                improved = True
                # Restart deletion loop as state changed
                break 
        
        if improved:
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = set(codewords)
            no_improve_count = 0
            continue
        
        # Phase 2: Try Swaps
        # Pick a random codeword to replace
        if codewords:
            c_old = random.choice(list(codewords))
            # Generate a candidate c_new
            # Heuristic: Pick a word that is currently "barely covered" (covered by 1 or 2)
            # Or just random.
            # Let's try random first.
            c_new = random.randint(0, total_words - 1)
            
            # Check if swap is valid
            # We need to check if removing c_old and adding c_new keeps uncovered_count == 0
            # This is equivalent to: 
            # 1. Remove c_old -> check if any nb becomes 0. If yes, invalid (unless c_new covers them).
            # 2. Add c_new -> check if it covers the "dangerous" words.
            
            # Efficient check:
            # Words that become uncovered if we remove c_old:
            # danger_words = [nb for nb in neighbors[c_old] if covered[nb] == 1]
            # If danger_words is empty, the swap is always valid (we just change coverage).
            # If danger_words is not empty, c_new MUST cover all of them.
            
            danger_words = []
            for nb in neighbors[c_old]:
                if covered[nb] == 1:
                    danger_words.append(nb)
            
            if not danger_words:
                # Swap is safe.
                remove(c_old)
                add(c_new)
                improved = True
                no_improve_count = 0
                continue
            else:
                # Check if c_new covers all danger_words
                valid_swap = True
                # Convert c_new to set of neighbors for fast lookup? 
                # neighbors[c_new] is a list. 
                # For small n, checking membership is O(1) if we use a set, but list is O(k).
                # k is ~13. 
                # Let's check.
                new_neighbors_set = set(neighbors[c_new])
                for dw in danger_words:
                    if dw not in new_neighbors_set:
                        valid_swap = False
                        break
                
                if valid_swap:
                    remove(c_old)
                    add(c_new)
                    improved = True
                    no_improve_count = 0
                    continue
        
        # Phase 3: Perturbation if stuck
        no_improve_count += 1
        if no_improve_count > 100:
            # Add a random codeword
            c_perturb = random.randint(0, total_words - 1)
            if c_perturb not in codewords:
                add(c_perturb)
            # Then run greedy deletion for a few steps
            # We just let the next loop's deletion phase handle it
            no_improve_count = 0
            
    # Return the best found
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
# EVOLVE-BLOCK-END
