import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = w[:pos] + (new_val,) + w[pos+1:]
                j = word_to_idx[new_word]
                neighbors[i].append(j)
                
    def get_idx(word):
        return word_to_idx[word]

    # 1. Greedy Initialization
    # We will use a slightly better greedy: pick word covering most uncovered.
    # Symmetry breaking: fix (0,...,0)
    
    def greedy_init():
        uncovered = [1] * total_words
        uncovered_count = total_words
        code = []
        code_set = set()
        
        first_word = (0,) * n
        first_idx = get_idx(first_word)
        code.append(first_word)
        code_set.add(first_word)
        
        for v in neighbors[first_idx]:
            if uncovered[v]:
                uncovered[v] = 0
                uncovered_count -= 1
                
        while uncovered_count > 0:
            best_idx = -1
            best_gain = 0
            
            # Optimization: Only check words that cover at least one uncovered word?
            # Or just scan all. 729 is small.
            for i in range(total_words):
                if all_words[i] in code_set:
                    continue
                
                gain = 0
                for v in neighbors[i]:
                    if uncovered[v]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            best_word = all_words[best_idx]
            code.append(best_word)
            code_set.add(best_word)
            
            for v in neighbors[best_idx]:
                if uncovered[v]:
                    uncovered[v] = 0
                    uncovered_count -= 1
                    
        return code

    # Helper to compute coverage counts
    def compute_counts(code):
        counts = [0] * total_words
        for w in code:
            idx = get_idx(w)
            for v in neighbors[idx]:
                counts[v] += 1
        return counts

    # Helper to check if code is valid (all counts > 0)
    def is_valid(counts):
        return all(c > 0 for c in counts)

    # Helper to reduce code by removing redundant words
    def reduce_code(code, counts):
        code_set = set(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                w = code[i]
                idx = get_idx(w)
                # Check if w is redundant
                # w is redundant if for all v in neighbors[idx], counts[v] > 1
                # Wait, if counts[v] > 1, it means at least 2 words cover v.
                # If we remove w, counts[v] becomes counts[v]-1.
                # If min(counts[v] for v in neighbors[idx]) >= 2, then after removal min >= 1.
                # So w is redundant.
                
                is_redundant = True
                for v in neighbors[idx]:
                    if counts[v] <= 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove w
                    for v in neighbors[idx]:
                        counts[v] -= 1
                    code.pop(i)
                    code_set.discard(w)
                    changed = True
                    break # Restart loop
        return code, counts

    # Main Search: Iterated Local Search with Remove-Repair Perturbation
    best_code = greedy_init()
    best_counts = compute_counts(best_code)
    best_size = len(best_code)
    
    current_code = list(best_code)
    current_counts = compute_counts(current_code)
    
    # Run ILS
    while time.time() < deadline:
        # 1. Reduce current code
        current_code, current_counts = reduce_code(current_code, current_counts)
        
        if len(current_code) < best_size:
            best_code = list(current_code)
            best_counts = compute_counts(best_code)
            best_size = len(best_code)
            
        # 2. Perturb: Remove a random word, then repair
        if not current_code:
            break
            
        # Pick a random word to remove
        idx_to_remove = random.randint(0, len(current_code) - 1)
        w_to_remove = current_code[idx_to_remove]
        idx_r = get_idx(w_to_remove)
        
        # Remove it
        for v in neighbors[idx_r]:
            current_counts[v] -= 1
        current_code.pop(idx_to_remove)
        
        # Now some words might be uncovered (count 0)
        # Repair: Greedily add words to cover uncovered
        uncovered_indices = [i for i, c in enumerate(current_counts) if c == 0]
        
        if uncovered_indices:
            # Greedy repair
            # We need to cover uncovered_indices
            # We can reuse the greedy logic but only on uncovered set
            # To save time, we just pick the best word among all words not in code
            code_set = set(current_code)
            
            while uncovered_indices and time.time() < deadline - 5.0:
                best_idx = -1
                best_gain = 0
                
                # Optimization: Only consider words that cover at least one uncovered word
                # We can iterate all words
                for i in range(total_words):
                    if all_words[i] in code_set:
                        continue
                    
                    gain = 0
                    for v in neighbors[i]:
                        if current_counts[v] == 0: # Uncovered
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        
                if best_idx == -1:
                    break
                    
                best_word = all_words[best_idx]
                current_code.append(best_word)
                code_set.add(best_word)
                
                for v in neighbors[best_idx]:
                    if current_counts[v] == 0:
                        current_counts[v] = 1 # Just set to 1? No, increment
                    else:
                        current_counts[v] += 1
                        
                # Update uncovered_indices? 
                # It's easier to recompute or just check in next loop condition.
                # For efficiency in this small loop, let's just recompute uncovered list if it's not too big.
                # Or simply: the while condition checks `uncovered_indices`.
                # We need to update `uncovered_indices` or the condition.
                # Let's just break the inner loop logic and rely on the outer `while uncovered_indices` 
                # but we need to update the list.
                
                # Efficient update:
                # We added best_word. Which uncovered words did it cover?
                newly_covered = []
                for v in neighbors[best_idx]:
                    if current_counts[v] == 1: # Was 0, now 1
                        newly_covered.append(v)
                
                if newly_covered:
                    newly_covered_set = set(newly_covered)
                    uncovered_indices = [i for i in uncovered_indices if i not in newly_covered_set]

        # 3. After repair, we have a valid code (hopefully).
        # Check validity
        if any(c == 0 for c in current_counts):
            # If repair failed (timeout or no words), revert to best?
            # Or just restart from best?
            current_code = list(best_code)
            current_counts = compute_counts(current_code)
            continue

        # 4. Reduce the repaired code
        current_code, current_counts = reduce_code(current_code, current_counts)
        
        if len(current_code) < best_size:
            best_code = list(current_code)
            best_counts = compute_counts(best_code)
            best_size = len(best_code)
            
        # Optional: If we found a new best, restart ILS from there?
        # Or just continue.
        
    # Final reduction on best
    best_code, best_counts = reduce_code(best_code, best_counts)
    
    return [list(w) for w in best_code]
