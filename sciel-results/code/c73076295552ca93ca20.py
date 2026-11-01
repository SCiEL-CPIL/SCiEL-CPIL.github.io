import itertools
import time
import random
import numpy as np

def _get_neighbors_indices(c, q, n):
    """Return list of indices of neighbors at Hamming distance <= 1 from c (tuple)."""
    # This is slow to compute on the fly for every call in a tight loop.
    # We should precompute.
    pass

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if n == 0:
        return [[]]
    
    # Handle small cases or specific known perfect codes
    if q == 3 and radius == 1 and n == 4:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words as lists of indices
    # neighbors[i] is a list of indices j such that dist(all_words[i], all_words[j]) <= radius
    # For radius=1, this is just the star neighborhood.
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[pos] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
    
    # Function to check if a set of codeword indices covers all words
    def is_valid(code_indices):
        covered = [0] * total_words
        for c_idx in code_indices:
            for nb_idx in neighbors[c_idx]:
                covered[nb_idx] += 1
        return all(c > 0 for c in covered)

    # Function to get coverage counts
    def get_coverage(code_indices):
        covered = [0] * total_words
        for c_idx in code_indices:
            for nb_idx in neighbors[c_idx]:
                covered[nb_idx] += 1
        return covered

    # Greedy Initialization
    def greedy_init(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(total_words))
        code_indices = []
        # To speed up, we can shuffle the order of candidates or just iterate
        # A simple greedy: pick word covering most uncovered
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            # Optimization: Instead of checking all words, we can be smarter, but for 729 words it's fast
            for i in range(total_words):
                if i in code_indices:
                    continue
                cnt = sum(1 for nb in neighbors[i] if nb in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            for nb in neighbors[best_idx]:
                uncovered.discard(nb)
                
        return code_indices

    # Direct Sum Initialization (for q=3, n=6, radius=1)
    def direct_sum_init():
        if q == 3 and n == 6 and radius == 1:
            tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
            code = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    c = h + tail
                    code.append(word_to_idx[c])
            return code
        return greedy_init(0)

    # Local Search
    def local_search(init_code, seed=0):
        rng = random.Random(seed)
        active = list(init_code)
        active_set = set(active)
        coverage = get_coverage(active)
        
        # Ensure valid
        if not is_valid(active):
            # Fix by adding words
            uncovered = [i for i in range(total_words) if coverage[i] == 0]
            while uncovered and time.time() < deadline:
                best_idx = -1
                best_count = -1
                for i in range(total_words):
                    if i in active_set:
                        continue
                    cnt = sum(1 for nb in neighbors[i] if coverage[nb] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_idx == -1:
                    break
                active.append(best_idx)
                active_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    coverage[nb] += 1
                uncovered = [i for i in uncovered if coverage[i] == 0]

        best_size = len(active)
        best_code = list(active)
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Removal
            # Try to remove a codeword if it doesn't uncover any words
            removal_found = False
            # Shuffle to avoid bias
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                can_remove = True
                for nb in neighbors[c]:
                    if coverage[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for nb in neighbors[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set.discard(c)
                    improved = True
                    removal_found = True
                    break # Restart removal phase with new state
            
            if removal_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                continue # Loop back to try more removals

            # Phase 2: Swaps
            # If no removal possible, try swaps
            # We look for a codeword c to remove, and a new codeword c' to add
            # such that the code remains valid.
            # To be effective, c' must cover all words that were uniquely covered by c.
            
            swap_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                # Find words uniquely covered by c
                unique_words = []
                for nb in neighbors[c]:
                    if coverage[nb] == 1:
                        unique_words.append(nb)
                
                if not unique_words:
                    # This is a removal case, but phase 1 should have caught it?
                    # Maybe coverage changed? No, we are in a stable state.
                    # So unique_words is never empty if removal failed?
                    # Wait, if removal failed, it means there IS a word with coverage 1.
                    # So unique_words is not empty.
                    pass

                # Find a candidate c' that covers all unique_words
                # And is not in active
                best_candidate = -1
                best_score = -1
                
                # Optimization: We don't need to check all 729 words if we are lucky,
                # but 729 is small.
                for c_prime in range(total_words):
                    if c_prime in active_set:
                        continue
                    
                    # Check if c_prime covers all unique_words
                    covers_all = True
                    # We can precompute a bitmask or just check
                    # Since unique_words is small (usually < 10), this is fast
                    for w in unique_words:
                        if w not in neighbors[c_prime]:
                            covers_all = False
                            break
                            
                    if not covers_all:
                        continue
                    
                    # Score the candidate
                    # We want c_prime to be "redundant" i.e., it covers words that are already well-covered.
                    # This might allow other codewords to be removed later.
                    # Metric: Number of neighbors of c_prime that have coverage > 1.
                    score = 0
                    for nb in neighbors[c_prime]:
                        if coverage[nb] > 1:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_candidate = c_prime

                if best_candidate != -1:
                    # Perform swap
                    # Remove c
                    for nb in neighbors[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set.discard(c)
                    
                    # Add c_prime
                    active.append(best_candidate)
                    active_set.add(best_candidate)
                    for nb in neighbors[best_candidate]:
                        coverage[nb] += 1
                    
                    improved = True
                    swap_found = True
                    break # Restart local search loop

            if not improved:
                # Stuck in local optimum. Perturb.
                # Remove 1-2 random codewords, then repair.
                if time.time() < deadline - 5:
                    perturb_count = rng.randint(1, 2)
                    removed = []
                    for _ in range(perturb_count):
                        if not active:
                            break
                        idx = rng.randint(0, len(active) - 1)
                        c = active.pop(idx)
                        active_set.discard(c)
                        for nb in neighbors[c]:
                            coverage[nb] -= 1
                        removed.append(c)
                    
                    # Repair
                    uncovered = [i for i in range(total_words) if coverage[i] == 0]
                    while uncovered and time.time() < deadline:
                        best_idx = -1
                        best_count = -1
                        for i in range(total_words):
                            if i in active_set:
                                continue
                            cnt = sum(1 for nb in neighbors[i] if coverage[nb] == 0)
                            if cnt > best_count:
                                best_count = cnt
                                best_idx = i
                        if best_idx == -1:
                            # Revert
                            for c in removed:
                                if c not in active_set:
                                    active.append(c)
                                    active_set.add(c)
                                    for nb in neighbors[c]:
                                        coverage[nb] += 1
                            removed = []
                            break
                        active.append(best_idx)
                        active_set.add(best_idx)
                        for nb in neighbors[best_idx]:
                            coverage[nb] += 1
                        uncovered = [i for i in uncovered if coverage[i] == 0]
                    
                    if len(active) < best_size:
                        best_size = len(active)
                        best_code = list(active)
                    
            if len(active) < best_size:
                best_size = len(active)
                best_code = list(active)

        return best_code

    # Main Logic: Try multiple strategies
    best_code = None
    best_size = float('inf')
    
    strategies = []
    
    # 1. Direct Sum
    if q == 3 and n == 6 and radius == 1:
        strategies.append(('direct_sum', direct_sum_init()))
        
    # 2. Greedy
    strategies.append(('greedy', greedy_init(123)))
    
    # 3. Greedy with different seed
    strategies.append(('greedy2', greedy_init(456)))
    
    # Run local search on each
    for name, init in strategies:
        if time.time() > deadline - 10:
            break
            
        # Allocate remaining time to this strategy
        # Simple: just run until deadline, but we have multiple strategies.
        # Let's split time.
        
        result = local_search(init, seed=hash(name) % 1000)
        
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
            
        # If we found a very good code, we might want to spend more time polishing it
        # But for now, let's just run the next strategy if time permits.
        
    if best_code is None:
        # Fallback
        best_code = greedy_init(0)

    # Convert indices back to lists of ints
    return [list(all_words[idx]) for idx in best_code]
