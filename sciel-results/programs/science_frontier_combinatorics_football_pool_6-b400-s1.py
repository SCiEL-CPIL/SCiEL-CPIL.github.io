import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Parameters for this specific task
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task specifies 3,6,1
        # Simple greedy for general case
        return _generic_greedy(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Total words in the space
    total_words = q ** n
    # Map each word (tuple) to an index 0..728
    # And precompute the coverage of each potential codeword
    
    # Generate all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: covers[c] is a set of word indices covered by codeword c
    covers = [None] * total_words
    for i, w in enumerate(all_words):
        cov = set()
        # Distance 0
        cov.add(i)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                cov.add(word_to_idx[new_w])
        covers[i] = cov
    
    # Precompute distance-2 neighbors using bitset approach for speed
    # Each word is 2 bits per position (since q=3), so 12 bits total
    # But we'll use a simpler approach: precompute which words differ in exactly 2 positions
    dist2_neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        w1 = all_words[i]
        # Only check j > i to avoid duplicates, but we need both directions
        for j in range(i + 1, total_words):
            w2 = all_words[j]
            # Fast distance check
            d = 0
            for k in range(n):
                if w1[k] != w2[k]:
                    d += 1
                    if d > 2:
                        break
            if d == 2:
                dist2_neighbors[i].add(j)
                dist2_neighbors[j].add(i)

    # Greedy independent set on distance-2 graph
    # This gives a structurally better initial cover
    # Use a randomized greedy approach with multiple restarts
    best_independent_set = []
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        random.seed(restart * 12345 + 678)
        remaining = set(range(total_words))
        independent_set = []
        while remaining:
            # Pick a random vertex from remaining, weighted toward low degree
            # Sample a few candidates and pick the one with lowest degree
            candidates = random.sample(list(remaining), min(50, len(remaining)))
            best_v = None
            best_deg = float('inf')
            for v in candidates:
                deg = len(dist2_neighbors[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is not None:
                independent_set.append(best_v)
                remaining.remove(best_v)
                remaining -= dist2_neighbors[best_v]
        if len(independent_set) > len(best_independent_set):
            best_independent_set = independent_set
    
    # If independent set is too large or we ran out of time, fall back to direct sum code
    if len(best_independent_set) > 81 or time.time() > deadline - 20:
        # Use direct sum code as fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                cw = tuple(h + list(tail))
                initial_indices.append(word_to_idx[cw])
    else:
        initial_indices = best_independent_set
    
    active_indices = set(initial_indices)
    
    # cover_count[w] = number of active codewords that cover w
    cover_count = np.zeros(total_words, dtype=int)
    for idx in active_indices:
        for w in covers[idx]:
            cover_count[w] += 1
    
    # Check if initial cover is valid (covers all words)
    if np.any(cover_count == 0):
        # Need to add more codewords to cover uncovered words
        uncovered = set(np.where(cover_count == 0)[0])
        while uncovered:
            # Greedy: pick codeword that covers most uncovered words
            best_c = -1
            best_count = -1
            for c in range(total_words):
                if c in active_indices:
                    continue
                count = len(covers[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1 or best_count == 0:
                break
            active_indices.add(best_c)
            for w in covers[best_c]:
                cover_count[w] += 1
                if cover_count[w] == 1:
                    uncovered.discard(w)
    
    def is_redundant(idx):
        # Check if all words covered by idx are covered by at least one other active codeword
        for w in covers[idx]:
            if cover_count[w] == 1:
                return False
        return True

    def remove_codeword(idx):
        nonlocal cover_count
        for w in covers[idx]:
            cover_count[w] -= 1
        active_indices.remove(idx)

    def add_codeword(idx):
        nonlocal cover_count
        for w in covers[idx]:
            cover_count[w] += 1
        active_indices.add(idx)
    
    # Phase 1: Greedy Redundancy Removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(active_indices):
            if is_redundant(idx):
                remove_codeword(idx)
                changed = True
    
    # Phase 2: Local Search / Simulated Annealing
    # We try to find moves that reduce size (2-for-1) or maintain size (swap) to escape local minima
    
    current_size = len(active_indices)
    best_size = current_size
    best_state = set(active_indices)
    
    # If we already reached 73, return
    if best_size <= 73:
        return [list(all_words[i]) for i in sorted(best_state)]
    
    # Precompute inverse coverage: for each word w, which codewords cover it
    # This helps in finding good swap candidates
    covered_by = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in covers[c]:
            covered_by[w].add(c)
    
    steps = 0
    max_steps = 300000 # Limit steps to avoid infinite loop if time check fails
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Periodically try to find 2-for-1 moves
        if steps % 50 == 0:
            # Try to find c1, c2 such that their union is covered by some c3
            # Sample a few random pairs to avoid full O(N^3)
            if len(active_indices) > 2:
                active_list = list(active_indices)
                c1, c2 = random.sample(active_list, 2)
                union_cov = covers[c1] | covers[c2]
                # Check if any c3 covers union_cov
                # Optimization: c3 must cover all words in union_cov
                # Pick a word from union_cov and only check codewords that cover it
                sample_word = random.choice(list(union_cov))
                candidate_c3s = covered_by[sample_word]
                found_c3 = -1
                for c3 in candidate_c3s:
                    if union_cov.issubset(covers[c3]):
                        found_c3 = c3
                        break
                
                if found_c3 != -1 and found_c3 not in active_indices:
                    # This is a valid 2-for-1 move: size decreases by 1
                    remove_codeword(c1)
                    remove_codeword(c2)
                    add_codeword(found_c3)
                    best_size = len(active_indices)
                    best_state = set(active_indices)
                    if best_size <= 73:
                        break
                    continue
        
        # Try a Swap
        if len(active_indices) > 0:
            active_list = list(active_indices)
            c1 = random.choice(active_list)
            # Find a c2 such that covers[c1] is covered by (Active - {c1}) U {c2}
            # i.e., words uniquely covered by c1 must be covered by c2.
            unique_c1 = set()
            for w in covers[c1]:
                if cover_count[w] == 1:
                    unique_c1.add(w)
            
            if not unique_c1:
                # c1 is redundant, remove it
                remove_codeword(c1)
                best_size = len(active_indices)
                best_state = set(active_indices)
                if best_size <= 73:
                    break
                continue
            
            # Find c2 that covers unique_c1
            # Pick a word from unique_c1 and find codewords that cover it
            target_word = random.choice(list(unique_c1))
            candidate_c2s = covered_by[target_word]
            
            # Try a few random candidates
            found_swap = False
            for _ in range(10):
                c2 = random.choice(list(candidate_c2s))
                if c2 == c1:
                    continue
                if unique_c1.issubset(covers[c2]):
                    # This swap is valid: c2 covers all uniquely covered words of c1
                    remove_codeword(c1)
                    if c2 not in active_indices:
                        add_codeword(c2)
                    # Size stays same or decreases if c2 was already active
                    if len(active_indices) < best_size:
                        best_size = len(active_indices)
                        best_state = set(active_indices)
                    found_swap = True
                    break
            
            if found_swap and best_size <= 73:
                break
    
    # Final Redundancy Check
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in list(active_indices):
            if is_redundant(idx):
                remove_codeword(idx)
                changed = True
                best_size = len(active_indices)
                best_state = set(active_indices)
    
    # Use the best state found
    if len(active_indices) < best_size:
        best_state = set(active_indices)
        best_size = len(active_indices)
    
    # Return the list of codewords
    return [list(all_words[i]) for i in sorted(best_state)]

def _generic_greedy(q, n, radius):
    # Fallback for other parameters
    # Simple greedy set cover
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = []
    for w in all_words:
        cov = set()
        # Simple BFS/DFS for radius? 
        # For small radius, just iterate
        # This is slow for large n, but fallback only
        for cand in all_words:
            dist = sum(1 for a, b in zip(w, cand) if a != b)
            if dist <= radius:
                cov.add(word_to_idx[cand])
        covers.append(cov)
        
    uncovered = set(range(len(all_words)))
    selected = []
    
    while uncovered:
        best_c = -1
        best_count = -1
        for i, cov in enumerate(covers):
            if i in [word_to_idx[c] for c in selected]:
                continue
            count = len(cov & uncovered)
            if count > best_count:
                best_count = count
                best_c = i
        if best_c == -1:
            break
        selected.append(all_words[best_c])
        uncovered -= covers[best_c]
        
    return selected
