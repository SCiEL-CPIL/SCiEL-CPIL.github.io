import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # The task is specifically q=3, n=6, radius=1.
    # We handle this specific case with an optimized algorithm.
    # For other cases, we provide a generic fallback (though unlikely to be tested).
    
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius, start_time, deadline)

    # --- Specific Optimization for K_3(6,1) ---
    
    # 1. Precomputation
    # Map words to indices 0..728
    def word_to_idx(w):
        # w is a tuple of 6 digits in {0,1,2}
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)

    total_words = 3 ** n  # 729
    
    # Precompute the neighbors (ball of radius 1) for every word.
    # neighbors[i] is a list of indices of words within distance 1 of word i.
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                neighbors[idx].append(word_to_idx(tuple(new_w_list)))
    
    # 2. Initial Construction: Greedy Independent Set on Distance-2 Graph
    # Strategy: We want a set of codewords that covers all words.
    # A known good starting point is to avoid having two codewords at distance 2,
    # as they might be redundant or swappable.
    # We can construct an independent set in the graph where edges connect words at distance 2.
    # Actually, a simpler and effective strategy for covering codes is a greedy set cover,
    # but the "independent set" insight from the prompt suggests avoiding distance-2 pairs.
    # Let's try a greedy approach that prioritizes words that are "far" from already chosen ones,
    # or simply a greedy set cover which is robust.
    
    # Let's try the "Greedy Independent Set" idea more carefully.
    # If we pick a set S such that no two elements in S are at distance 2,
    # does it cover everything? Not necessarily.
    # However, the prompt says: "Greedy independent set on distance-2 graph provides a structurally better initial cover... Measured: 0.9759 -> 1.012."
    # This implies that starting from such a set and then pruning/repairing works well.
    
    # Let's implement a greedy set cover first, as it's a solid baseline.
    # Then we will try to improve it via local search (swaps).
    
    # Greedy Set Cover:
    # uncovered: set of indices not yet covered
    # For each step, pick the word that covers the most uncovered words.
    
    # To speed this up, we can use the precomputed neighbors.
    # coverage_count[i] = number of times word i is covered by current code.
    # But for greedy set cover, we need to know which words are NOT covered.
    
    uncovered = set(range(total_words))
    code = []
    
    # Optimization: Precompute the set of neighbors as a frozenset or list for fast intersection?
    # With 729 words, simple list iteration is fine.
    
    while uncovered:
        if time.time() > deadline - 10:
            break # Safety break for initial construction
        
        best_word = -1
        max_covered = 0
        
        # We can restrict the search to words that are in `uncovered`?
        # No, a word outside `uncovered` (already covered) might cover many other uncovered words.
        # So we must check all words.
        
        # To make this faster, we can maintain a list of candidate words.
        # But 729 is small.
        
        for w_idx in range(total_words):
            # How many of w_idx's neighbors are in `uncovered`?
            # neighbors[w_idx] is a list of ~19 indices.
            # Intersection with `uncovered` (a set) is fast.
            covered_count = sum(1 for nb in neighbors[w_idx] if nb in uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w_idx
                if max_covered == len(neighbors[w_idx]):
                    break # Can't do better than covering all neighbors
        
        if best_word == -1:
            break # Should not happen if uncovered is not empty and neighbors are correct
            
        code.append(best_word)
        # Remove covered words from `uncovered`
        for nb in neighbors[best_word]:
            uncovered.discard(nb)
            
    # Now we have a valid covering code `code`.
    # Size is likely around 80-100.
    
    # 3. Local Search: Redundancy Removal and Swaps
    # Goal: Minimize len(code).
    
    # First, try to remove redundant codewords.
    # A codeword c is redundant if every word in its ball is covered by at least one other codeword.
    # We maintain a coverage count array.
    
    def update_coverage(code_list, coverage):
        coverage[:] = [0] * total_words
        for c in code_list:
            for nb in neighbors[c]:
                coverage[nb] += 1

    coverage = [0] * total_words
    update_coverage(code, coverage)
    
    # Remove redundant codewords
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        new_code = []
        for c in code:
            # Check if c is redundant
            redundant = True
            for nb in neighbors[c]:
                if coverage[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for nb in neighbors[c]:
                    coverage[nb] -= 1
                improved = True
            else:
                new_code.append(c)
        code = new_code
        
    # Now try swaps: Replace one codeword with another to reduce size or maintain size while enabling further removals.
    # Since we just removed redundants, we are at a local minimum for removal.
    # To get smaller, we need to swap.
    # Swap: Remove c, add c'. Then try to remove redundants.
    # If the final size is smaller, accept.
    
    best_code = code[:]
    best_size = len(best_code)
    
    # Random restarts / swaps
    # We have ~140s left.
    # Each swap attempt involves:
    # 1. Removing c (O(19))
    # 2. Adding c' (O(19))
    # 3. Redundancy removal loop (O(|code| * 19))
    
    # Let's do a simulated annealing or simple hill climbing with random swaps.
    
    current_code = best_code[:]
    current_size = best_size
    current_coverage = [0] * total_words
    update_coverage(current_code, current_coverage)
    
    # Precompute a list of all words for random selection
    all_words = list(range(total_words))
    
    swap_count = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2:
        if no_improve_count > max_no_improve:
            # Reset or break? Let's try a random restart of the removal phase from a perturbed state?
            # Or just break if stuck for too long.
            break
            
        # Pick a random codeword to remove
        if not current_code:
            break
        c_idx = random.randint(0, len(current_code) - 1)
        c = current_code[c_idx]
        
        # Pick a random word to add (could be the same? No, must be distinct)
        # Ideally, pick a word that is NOT in current_code.
        # For simplicity, pick random word from all_words.
        c_prime = random.choice(all_words)
        
        if c_prime in current_code:
            continue
            
        # Tentative swap
        # Remove c
        for nb in neighbors[c]:
            current_coverage[nb] -= 1
            
        # Check if coverage is still valid?
        # If not, this swap is invalid immediately unless we add c_prime first.
        # We must add c_prime.
        
        # Add c_prime
        for nb in neighbors[c_prime]:
            current_coverage[nb] += 1
            
        # Now check if the code is still a valid cover (it should be, since we added one).
        # Wait, removing c might have broken coverage for some words that c_prime doesn't cover.
        # So we must check validity.
        
        valid = True
        for nb in neighbors[c]:
            if current_coverage[nb] < 1:
                valid = False
                break
                
        if not valid:
            # Revert
            for nb in neighbors[c_prime]:
                current_coverage[nb] -= 1
            for nb in neighbors[c]:
                current_coverage[nb] += 1
            continue
            
        # If valid, try to remove redundants from the new set
        # Build a list of current codewords
        new_code_list = [x for i, x in enumerate(current_code) if i != c_idx]
        new_code_list.append(c_prime)
        
        # Perform redundancy removal on new_code_list
        # We need a temporary coverage array? Or we can just use the current one and adjust?
        # It's safer to compute redundancy on the fly or use a helper.
        
        # Let's just run the removal loop on a copy of the state.
        # To save time, we only do this if the swap is promising?
        # Actually, the swap itself doesn't change the size.
        # The size only decreases if redundancy removal succeeds.
        
        # So:
        # 1. Swap c -> c_prime (size unchanged)
        # 2. Run redundancy removal.
        # 3. If size decreases, accept. Else, revert.
        
        # To run redundancy removal, we need the coverage counts.
        # We have current_coverage which reflects the state after swap.
        
        # Let's make a copy of the code and coverage to test
        test_code = new_code_list[:]
        test_coverage = current_coverage[:]
        
        test_improved = True
        while test_improved:
            if time.time() > deadline - 2:
                test_improved = False
                break
            test_improved = False
            new_test_code = []
            for tc in test_code:
                redundant = True
                for nb in neighbors[tc]:
                    if test_coverage[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    for nb in neighbors[tc]:
                        test_coverage[nb] -= 1
                    test_improved = True
                else:
                    new_test_code.append(tc)
            test_code = new_test_code
            
        new_size = len(test_code)
        
        if new_size < current_size:
            # Accept the swap and the removals
            current_code = test_code
            current_size = new_size
            current_coverage = test_coverage
            no_improve_count = 0
            swap_count += 1
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
                
        else:
            # Revert swap
            for nb in neighbors[c_prime]:
                current_coverage[nb] -= 1
            for nb in neighbors[c]:
                current_coverage[nb] += 1
            no_improve_count += 1
            swap_count += 1

    return [list(idx_to_word(c)) for c in best_code]


def _generic_construct(q, n, radius, start_time, deadline):
    # Fallback for other parameters
    # Simple greedy set cover
    total_words = q ** n
    if total_words > 10000:
        # Too large for simple python
        return []
        
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                neighbors[idx].append(word_to_idx(tuple(new_w_list)))
                
    uncovered = set(range(total_words))
    code = []
    
    while uncovered and time.time() < deadline - 5:
        best_word = -1
        max_covered = 0
        for w_idx in range(total_words):
            covered_count = sum(1 for nb in neighbors[w_idx] if nb in uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w_idx
        if best_word == -1:
            break
        code.append(best_word)
        for nb in neighbors[best_word]:
            uncovered.discard(nb)
            
    return [list(idx_to_word(c)) for c in code]
