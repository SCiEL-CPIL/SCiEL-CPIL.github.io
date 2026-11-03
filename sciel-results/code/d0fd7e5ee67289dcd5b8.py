import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball masks for each word
    # Ball of radius 1 around w includes w and all words at distance 1.
    ball_masks = [0] * total_words
    ball_indices = [None] * total_words  # List of indices in the ball for faster iteration
    
    for i, w in enumerate(all_words):
        mask = 1 << i  # Include self
        indices = [i]
        
        # Generate distance 1 neighbors
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new word
                new_word = w[:pos] + (new_val,) + w[pos+1:]
                j = word_to_idx[new_word]
                mask |= (1 << j)
                indices.append(j)
                
        ball_masks[i] = mask
        ball_indices[i] = indices

    # Helper to get index from word
    def get_idx(word):
        return word_to_idx[word]

    # 1. Greedy Initialization
    uncovered_mask = (1 << total_words) - 1
    code = []
    code_set = set()
    
    # Symmetry breaking: Fix first codeword to all zeros
    first_word = (0,) * n
    first_idx = get_idx(first_word)
    code.append(first_word)
    code_set.add(first_word)
    uncovered_mask &= ~ball_masks[first_idx]
    
    while uncovered_mask != 0 and time.time() < deadline - 10.0:
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
        for i in range(total_words):
            if all_words[i] in code_set:
                continue
            covered = uncovered_mask & ball_masks[i]
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        best_word = all_words[best_idx]
        code.append(best_word)
        code_set.add(best_word)
        uncovered_mask &= ~ball_masks[best_idx]

    # 2. Intensive Local Search with Pruning and Annealing
    
    # State variables
    current_code = list(code)
    current_set = set(current_code)
    
    # cover_count[w] = number of codewords in current_code that cover word w
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    def update_cover_count(word, delta):
        idx = get_idx(word)
        for j in ball_indices[idx]:
            cover_count[j] += delta
            
    # Initialize cover_count
    for w in current_code:
        update_cover_count(w, 1)
        
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Check if current is valid and minimal
    def is_valid():
        return np.all(cover_count > 0)
        
    # Pruning step: Remove any redundant codeword
    def prune():
        nonlocal current_code, current_set, cover_count, best_code, best_size
        
        improved = True
        while improved:
            improved = False
            # Iterate in reverse to allow easy removal
            for i in range(len(current_code) - 1, -1, -1):
                w = current_code[i]
                idx = get_idx(w)
                
                # Check if any word in the ball has count 1
                # If so, removing w would uncover it.
                redundant = True
                for j in ball_indices[idx]:
                    if cover_count[j] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Safe to remove
                    update_cover_count(w, -1)
                    current_code.pop(i)
                    current_set.discard(w)
                    improved = True
                    if is_valid() and len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    break # Restart loop after removal
        
    # Initial prune
    prune()
    
    # Simulated Annealing Loop
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # We will alternate between SA steps and Pruning
    last_prune_time = time.time()
    prune_interval = 2.0 # seconds
    
    while time.time() < deadline - 1.0:
        
        # Check if we should prune
        if time.time() - last_prune_time > prune_interval:
            prune()
            last_prune_time = time.time()
            T = max(T, 1.0) # Reheat slightly after prune? Or keep cooling.
            
        if not current_code:
            break
            
        # Choose a move: Remove (0.4), Swap (0.6)
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove Move
            idx_to_remove = random.randint(0, len(current_code) - 1)
            w = current_code[idx_to_remove]
            idx = get_idx(w)
            
            # Check for uncovered words if removed
            # A word j is uncovered if cover_count[j] == 1 and j is in ball(w)
            num_uncovered = 0
            for j in ball_indices[idx]:
                if cover_count[j] == 1:
                    num_uncovered += 1
            
            old_cost = len(current_code)
            new_size = len(current_code) - 1
            new_cost = new_size + 10.0 * num_uncovered
            
            delta = new_cost - old_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                update_cover_count(w, -1)
                current_code.pop(idx_to_remove)
                current_set.discard(w)
                
                if num_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        
        else:
            # Swap Move
            idx_to_swap = random.randint(0, len(current_code) - 1)
            old_w = current_code[idx_to_swap]
            old_idx = get_idx(old_w)
            
            # Pick a random new word not in code
            # To ensure we pick a valid candidate, we can just pick random and check
            # But picking from all_words is O(1)
            new_idx = random.randint(0, total_words - 1)
            new_w = all_words[new_idx]
            
            if new_w in current_set:
                continue
                
            new_idx_word = get_idx(new_w)
            
            # Calculate change in coverage
            # Remove old_w, Add new_w
            # We need to check if the union of (Code - {old_w}) U {new_w} covers everything.
            
            # Words that become uncovered:
            # Those in ball(old_w) that are NOT in ball(new_w) and have count 1 (only covered by old_w)
            # Words that become covered:
            # Those in ball(new_w) that are NOT in ball(old_w) and have count 0 (currently uncovered? No, we assume valid state mostly)
            
            # Efficient check:
            # 1. Tentatively remove old_w
            # 2. Check if any word in ball(old_w) \ ball(new_w) has count 1
            # 3. If yes, they become uncovered.
            # 4. Also check if new_w covers any currently uncovered words? 
            #    Actually, if we start from a valid state, cover_count >= 1 everywhere.
            #    If we remove old_w, some words in ball(old_w) might drop to 0.
            #    If we add new_w, some words in ball(new_w) might go from 0 to 1.
            
            # Let's just calculate the new cover counts for the affected words.
            # Affected words are in ball(old_w) U ball(new_w).
            
            # To be safe and fast, let's just check the condition:
            # For all j in ball(old_w):
            #   if j not in ball(new_w) and cover_count[j] == 1:
            #       num_uncovered += 1
            
            # This is O(|Ball|) ~ 19 checks.
            
            new_ball_set = set(ball_indices[new_idx_word])
            
            num_uncovered = 0
            for j in ball_indices[old_idx]:
                if j not in new_ball_set:
                    if cover_count[j] == 1:
                        num_uncovered += 1
            
            # Note: This logic assumes that words NOT in ball(old_w) are still covered.
            # This is true because we only removed old_w.
            # Words in ball(new_w) that were uncovered (count 0) are now covered.
            # But if we started valid, count was >= 1.
            # So num_uncovered is the only penalty.
            
            old_cost = len(current_code)
            new_cost = len(current_code) + 10.0 * num_uncovered
            
            delta = new_cost - old_cost
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                # Update cover counts
                # For j in ball(old_w) U ball(new_w):
                #   if j in ball(old_w) and j not in ball(new_w): cover_count[j] -= 1
                #   if j in ball(new_w) and j not in ball(old_w): cover_count[j] += 1
                #   if j in both: no change
                #   if j in neither: no change
                
                old_ball_set = set(ball_indices[old_idx])
                
                for j in ball_indices[old_idx]:
                    if j not in new_ball_set:
                        cover_count[j] -= 1
                        
                for j in ball_indices[new_idx_word]:
                    if j not in old_ball_set:
                        cover_count[j] += 1
                        
                current_code[idx_to_swap] = new_w
                current_set.discard(old_w)
                current_set.add(new_w)
                
                if num_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final prune on best_code to ensure it's minimal
    # We need to reconstruct cover_count for best_code
    final_code = list(best_code)
    final_set = set(final_code)
    
    # Re-initialize cover_count for final_code
    cover_count_final = np.zeros(total_words, dtype=np.int32)
    for w in final_code:
        idx = get_idx(w)
        for j in ball_indices[idx]:
            cover_count_final[j] += 1
            
    # Prune final_code
    improved = True
    while improved:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            w = final_code[i]
            idx = get_idx(w)
            
            redundant = True
            for j in ball_indices[idx]:
                if cover_count_final[j] == 1:
                    redundant = False
                    break
                    
            if redundant:
                for j in ball_indices[idx]:
                    cover_count_final[j] -= 1
                final_code.pop(i)
                final_set.discard(w)
                improved = True
                break

    return [list(w) for w in final_code]
