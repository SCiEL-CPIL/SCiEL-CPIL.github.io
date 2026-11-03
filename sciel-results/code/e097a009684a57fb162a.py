import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for this specific task
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task specifies 3,6,1
        # Simple greedy for general case
        return _generic_greedy(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Total words in the space
    total_words = q ** n
    # Map each word (tuple) to an index 0..728
    # And precompute the coverage of each potential codeword
    
    # Generate all words
    all_words = [tuple(w) for w in __import__('itertools').product(range(q), repeat=n)]
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

    # Initialize with the direct sum code (81 words)
    # Tetracode part: [a, b, (a+b)%3, (a+2b)%3]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in __import__('itertools').product(range(3), repeat=n - 4):
            cw = h + list(tail)
            initial_codewords.append(tuple(cw))
    
    active_codewords = set(initial_codewords)
    active_indices = set(word_to_idx[cw] for cw in active_codewords)
    
    # cover_count[w] = number of active codewords that cover w
    cover_count = np.zeros(total_words, dtype=int)
    for idx in active_indices:
        for w in covers[idx]:
            cover_count[w] += 1
    
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

    # Simulated Annealing parameters
    temperature = 50.0
    cooling = 0.99
    
    # We will iterate through potential moves
    # Move types:
    # 1. Remove redundant (handled above, but might become redundant after swaps)
    # 2. Swap: Remove c1, Add c2. Accept if valid (no coverage loss)
    # 3. 2-for-1: Remove c1, c2, Add c3. Accept if valid (no coverage loss)
    
    # To speed up 2-for-1, we can precompute which pairs are "close" or just iterate.
    # Given 729 words, iterating all pairs is ~260k. For each pair, checking if a c3 exists is expensive.
    # Instead, let's focus on Swaps first to break symmetry, then check for 2-for-1 periodically.
    
    steps = 0
    max_steps = 500000 # Limit steps to avoid infinite loop if time check fails
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Periodically try to find 2-for-1 moves
        if steps % 100 == 0:
            # Try to find c1, c2 such that their union is covered by some c3
            # Sample a few random pairs to avoid full O(N^3)
            if len(active_indices) > 2:
                c1, c2 = random.sample(list(active_indices), 2)
                union_cov = covers[c1] | covers[c2]
                # Check if any c3 covers union_cov
                # Optimization: If c3 covers union_cov, it must cover all words in union_cov.
                # We can check candidates.
                # A candidate c3 must be within distance 1 of any word in union_cov? No.
                # c3 just needs to be a codeword such that union_cov is subset of covers[c3].
                # This implies c3 is a "center" for the union.
                # Let's just check all 729 candidates for this pair.
                found_c3 = -1
                for c3 in range(total_words):
                    if union_cov.issubset(covers[c3]):
                        found_c3 = c3
                        break
                
                if found_c3 != -1 and found_c3 not in active_indices:
                    # Check if removing c1, c2 and adding c3 is valid
                    # It is valid if c3 covers all words that were uniquely covered by c1 or c2
                    # Actually, if union_cov is subset of covers[c3], then any word covered by c1 or c2 is covered by c3.
                    # So we just need to ensure that words NOT covered by c1 or c2 but covered by others are not affected.
                    # The coverage of the system is the union of covers of active codewords.
                    # New coverage = (Old Coverage - covers[c1] - covers[c2]) + covers[c3]
                    # Since covers[c1] U covers[c2] is subset of covers[c3], the new coverage is a superset of the old coverage?
                    # No, old coverage included covers[c1] and covers[c2]. New includes covers[c3].
                    # If covers[c1] U covers[c2] subset covers[c3], then the new coverage is exactly the same as the old coverage 
                    # PLUS any words covered by c3 but not by c1 or c2.
                    # So it is a valid move (coverage is maintained or increased).
                    # Size decreases by 1.
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
            c1 = random.choice(list(active_indices))
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
            # Randomly pick a candidate from the space
            # To be more directed, pick a word from unique_c1, and pick a neighbor as c2
            target_word = random.choice(list(unique_c1))
            # c2 must be within distance 1 of target_word
            # Actually c2 must cover target_word, so c2 is in covers[target_word] (inverse)
            # Let's just pick a random c2 from all words
            c2 = random.randint(0, total_words - 1)
            
            if c2 == c1:
                continue
                
            if unique_c1.issubset(covers[c2]):
                # Check if adding c2 and removing c1 is valid
                # We need to ensure that words covered by c2 but not c1 don't cause issues? No, adding coverage is fine.
                # We just need to ensure we don't lose coverage.
                # We lose coverage of c1. c2 covers unique_c1.
                # What about words covered by c1 but also by others? They are still covered by others.
                # So the move is valid.
                remove_codeword(c1)
                if c2 not in active_indices:
                    add_codeword(c2)
                    # Size is same
                    # Accept with probability based on SA? 
                    # Since size is same, we accept to explore state space.
                    # But we should track best size.
                    # We don't update best_state unless size decreases.
                    # However, this move might enable future 2-for-1 moves.
                    pass
                else:
                    # c2 already active. Removing c1 makes size smaller? No, c2 was already there.
                    # So we just remove c1. Size decreases by 1.
                    # This is a redundancy removal that was missed? 
                    # If c2 is active and covers unique_c1, then c1 was redundant.
                    # Why didn't Phase 1 catch it? Maybe it became redundant after previous swaps.
                    pass

        # Occasionally, try to add a random word and remove two? (1-for-2)
        # This increases size, which we don't want, unless it leads to a better configuration.
        # Given time constraints, stick to size-reducing or size-maintaining moves.

        # Cooling
        temperature *= cooling

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
