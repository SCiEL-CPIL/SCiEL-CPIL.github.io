import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin for overhead/return
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        return _general_greedy(q, n, radius)

    # Specific optimization for q=3, n=6, r=1
    return _solve_q3_n6_r1(deadline)

def _solve_q3_n6_r1(deadline):
    # Initial construction: Direct sum of tetracode and all words on remaining 2 coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    code = initial_code
    # Precompute spheres for all possible words? 
    # Space is 3^6 = 729. 
    # Spheres of radius 1 in length 6 have size 1 + 6*2 = 13.
    # We can precompute the index of each word and its neighbors.
    
    words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for each word index
    neighbors = [[] for _ in range(729)]
    for i, w in enumerate(words):
        neighbors[i].append(i) # self
        for pos in range(6):
            for val in range(3):
                if w[pos] == val: continue
                new_w = list(w)
                new_w[pos] = val
                neighbors[i].append(word_to_idx[tuple(new_w)])
                
    # Coverage counts
    coverage = [0] * 729
    
    def update_coverage(codeword, delta):
        idx = word_to_idx[tuple(codeword)]
        for nb in neighbors[idx]:
            coverage[nb] += delta
            
    # Initialize coverage
    for c in code:
        update_coverage(c, 1)
        
    # Helper to check if code is valid (should always be true if we start valid and only do safe moves)
    def is_valid():
        return all(c > 0 for c in coverage)
        
    # Main loop
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove redundant codewords
        # A codeword is redundant if removing it doesn't uncover any word.
        # i.e., for all w in Sphere(c), coverage[w] > 1.
        # We iterate in reverse to allow removal without index shifting issues if we pop, 
        # but we are using a list, so we can just mark for removal or pop.
        
        to_remove = []
        for i, c in enumerate(code):
            idx = word_to_idx[tuple(c)]
            redundant = True
            for nb in neighbors[idx]:
                if coverage[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
                
        if to_remove:
            # Remove them
            # Sort indices descending to remove from end first
            for i in sorted(to_remove, reverse=True):
                c = code[i]
                update_coverage(c, -1)
                code.pop(i)
            improved = True
            continue # Re-check redundancy immediately as removals might expose more
        
        # 2. Try to swap codewords to enable further removals or just improve structure
        # Strategy: Find a codeword c_old and a candidate c_new such that 
        # replacing c_old with c_new maintains coverage (all words in Sphere(c_old) are still covered)
        # AND c_new is not in code.
        # To be effective, we look for c_old that covers some "critical" words (count 1) 
        # but we want to replace it with something that covers those critical words too, 
        # potentially covering other critical words as well, or just maintaining validity 
        # to allow a different structure.
        
        # A simpler heuristic: 
        # For each c_old in code:
        #   Let U be the set of words in Sphere(c_old) with coverage == 1.
        #   If U is empty, c_old is redundant (handled above).
        #   We want to find c_new not in code such that U is subset of Sphere(c_new).
        #   If found, swap.
        
        # To speed up, we can precompute for each word w, which codewords cover it?
        # That's the inverse of neighbors.
        # But 729 words * 13 neighbors is small.
        
        # Let's try a limited number of swaps per iteration to avoid getting stuck.
        swaps_made = 0
        max_swaps = 50
        
        # We need to efficiently find c_new.
        # For a given c_old, the set of required words is U.
        # c_new must be a neighbor of all w in U.
        # Intersection of neighborhoods of all w in U.
        
        # Optimization: 
        # If |U| is large, the intersection is likely small or empty.
        # If |U| is small, it's easier.
        
        # Let's iterate through code
        found_swap = False
        # Randomize order to escape local minima?
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            c_old = code[i]
            idx_old = word_to_idx[tuple(c_old)]
            
            # Find critical words covered by c_old
            critical_words = []
            for nb in neighbors[idx_old]:
                if coverage[nb] == 1:
                    critical_words.append(nb)
                    
            if not critical_words:
                # This codeword is redundant, but we didn't catch it? 
                # Should have been caught in step 1. 
                # Maybe coverage changed? No, we continue after removal.
                continue
                
            # We need c_new such that all critical_words are in Sphere(c_new).
            # This means c_new must be a common neighbor of all w in critical_words.
            # Note: c_new can be any word in the space.
            
            # Find intersection of neighbors of all critical words.
            # Start with neighbors of first critical word
            candidates = set(neighbors[critical_words[0]])
            
            for w in critical_words[1:]:
                candidates.intersection_update(neighbors[w])
                if not candidates:
                    break
            
            if not candidates:
                continue
                
            # Filter candidates:
            # 1. Must not be in current code.
            # 2. Ideally, it should cover more words or be "better" somehow?
            # For now, any valid swap is good to break symmetry/local minima.
            
            code_set = set(tuple(c) for c in code)
            
            valid_new = []
            for c_new_idx in candidates:
                c_new = list(words[c_new_idx])
                if tuple(c_new) not in code_set:
                    valid_new.append(c_new)
                    
            if valid_new:
                # Pick one. Maybe the one that covers the most total words? 
                # Or just random?
                # Let's pick the one that maximizes the minimum coverage of the words it covers?
                # Simpler: Just pick the first one.
                c_new = valid_new[0]
                
                # Perform swap
                update_coverage(c_old, -1)
                update_coverage(c_new, 1)
                code[i] = c_new
                found_swap = True
                swaps_made += 1
                if swaps_made >= max_swaps:
                    break
                    
        if found_swap:
            improved = True
            continue # Re-evaluate redundancy
            
        # If no improvement, we might be stuck in a local minimum.
        # We can try a random restart or a "kick".
        # Given the time limit, and that we likely found the optimal or near-optimal 
        # (73 is target, 81 is start), if we can't remove more, we stop.
        # However, we might be stuck at 74 or 75.
        
        # Try a more aggressive move: 
        # Remove a codeword, see how many words become uncovered.
        # Try to cover those uncovered words with as few new codewords as possible?
        # This is complex.
        
        # Alternative: Randomized restart of the local search from the current state?
        # Or just break if no improvement in a full pass.
        
        break

    return code

def _general_greedy(q, n, radius):
    # Simple greedy for other parameters
    # This is a placeholder to ensure the function doesn't crash if parameters differ,
    # though the task specifies q=3, n=6, r=1.
    return []
