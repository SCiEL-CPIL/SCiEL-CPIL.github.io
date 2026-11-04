import numpy as np
import random
import itertools
import time

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple greedy for other parameters, though task is specific
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].append(word_to_idx[new_word])
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = sum(1 for nb in neighbors[c] if nb in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.remove(nb)
        
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words) # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1)
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                new_word = list(w)
                new_word[pos] = val
                new_word = tuple(new_word)
                neighbors[i].append(word_to_idx[new_word])
        
    # Helper: Greedy Set Cover for a specific set of uncovered indices
    def greedy_cover(uncovered_set):
        if not uncovered_set:
            return []
        solution = []
        remaining = set(uncovered_set)
        while remaining:
            best_c = -1
            best_gain = -1
            # Optimization: Only consider candidates that cover at least one remaining word
            # We can iterate through all N, but checking gain is O(neighbors[c])
            # To speed up, we could maintain a list of candidates, but for small N, full scan is okay.
            # However, we can optimize by only checking candidates that are neighbors of any word in remaining.
            # But simpler: just scan all N.
            for c in range(N):
                gain = 0
                for nb in neighbors[c]:
                    if nb in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            for nb in neighbors[best_c]:
                if nb in remaining:
                    remaining.remove(nb)
        return solution

    # Helper: Prune redundant codewords
    def prune(code):
        if not code:
            return []
        # Count how many codewords cover each word
        cover_count = np.zeros(N, dtype=int)
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        final_code = list(code)
        # Iterate and remove if redundant
        # We must be careful: removing one might make another redundant.
        # So we iterate until no changes or pass through list.
        changed = True
        while changed:
            changed = False
            for i in range(len(final_code)):
                c = final_code[i]
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    final_code.pop(i)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    changed = True
                    break # Restart loop as indices changed
        return final_code

    # Initial Solution: Greedy
    current_code = greedy_cover(set(range(N)))
    current_code = prune(current_code)
    best_code = current_code
    best_size = len(best_code)
    
    start_time = time.time()
    timeout = 115.0 # Leave 5s buffer
    
    random.seed(42)
    
    while time.time() - start_time < timeout:
        # Perturb: Remove k random codewords
        # k should be small, e.g., 1 to 3
        k = random.randint(1, 3)
        if len(current_code) <= k:
            continue
            
        removed_indices = random.sample(range(len(current_code)), k)
        removed_codewords = [current_code[i] for i in removed_indices]
        
        # Calculate words that become uncovered
        # A word is uncovered if it was ONLY covered by the removed codewords.
        # We need to know the cover count of the current_code.
        cover_count = np.zeros(N, dtype=int)
        for c in current_code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        uncovered_set = set()
        for c in removed_codewords:
            for nb in neighbors[c]:
                # If we remove c, cover_count[nb] decreases.
                # If it becomes 0, it's uncovered.
                # But we are removing multiple. 
                # Let's simulate:
                pass
        
        # Better way:
        # 1. Build cover count for current_code
        # 2. Decrement for removed_codewords
        # 3. Collect indices where count becomes 0
        
        # Rebuild cover count for efficiency in loop? No, do it once.
        # We already have cover_count from the check above? No, that was inside the if.
        # Let's just compute it here.
        
        # Actually, we can just take the union of neighbors of removed codewords,
        # and check if they are covered by (current_code - removed_codewords).
        
        remaining_code = [c for i, c in enumerate(current_code) if i not in removed_indices]
        
        # Set of words covered by remaining_code
        covered_by_remaining = set()
        for c in remaining_code:
            covered_by_remaining.update(neighbors[c])
        
        # Words that were covered by removed but not by remaining
        for c in removed_codewords:
            for nb in neighbors[c]:
                if nb not in covered_by_remaining:
                    uncovered_set.add(nb)
        
        if not uncovered_set:
            # We removed codewords and nothing broke. This is a pruning opportunity we missed?
            # Or just a valid move.
            new_code = remaining_code
            new_code = prune(new_code)
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
            current_code = new_code
            continue
        
        # Repair: Greedy cover for uncovered_set
        repair_code = greedy_cover(uncovered_set)
        
        # Combine
        candidate_code = remaining_code + repair_code
        candidate_code = prune(candidate_code)
        
        if len(candidate_code) < best_size:
            best_size = len(candidate_code)
            best_code = candidate_code
            current_code = candidate_code
        else:
            # Accept move with some probability? Or just reject?
            # For simplicity and speed, accept if it doesn't make things worse?
            # Or accept if size is same?
            # Let's accept if size <= current_size to allow exploration of same-size solutions
            # which might lead to better ones later.
            if len(candidate_code) <= len(current_code):
                current_code = candidate_code

    return [list(all_words[i]) for i in best_code]
