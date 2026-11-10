import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety/overhead
    
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    idx_map = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (distance <= radius)
    # For radius=1, neighbors are the word itself and all words at distance 1.
    neighbors = [[] for _ in range(total_words)]
    
    for i, w in enumerate(all_words):
        # The word itself
        neighbors[i].append(i)
        
        if radius >= 1:
            # Words at distance 1
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = idx_map[tuple(new_w)]
                    neighbors[i].append(new_idx)
                    
        # Note: For radius > 1, we would need more logic, but task is radius=1.
        # The prompt says "checked over the whole space" and "radius" is passed.
        # The task is specifically K_3(6,1). But construct takes radius.
        # If radius > 1, the neighbor generation above is incomplete.
        # However, the task contract says "ternary covering code K_3(6,1)".
        # I will assume radius=1 is the primary case, but I should handle radius=0 trivially.
        # If radius > 1, the greedy approach might be too slow or need different neighbor gen.
        # Given the specific task description, I will optimize for radius=1.
        
    covered = [False] * total_words
    picked = [False] * total_words
    code_indices = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        # Optimization: keep track of uncovered count
        # Initially total_words. Decrease by gain each step.
        # Let's just check at the end of the loop or use a counter.
        
        best_gain = -1
        best_idx = -1
        
        for i in range(total_words):
            if picked[i]:
                continue
            
            # Calculate gain
            gain = 0
            for j in neighbors[i]:
                if not covered[j]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break
            
        picked[best_idx] = True
        code_indices.append(best_idx)
        
        # Update covered
        for j in neighbors[best_idx]:
            covered[j] = True
            
        # Early termination if all covered
        # We can check if sum(covered) == total_words, but that's O(N).
        # Better to maintain a count.
        
    # Actually, let's add a counter for uncovered to avoid O(N) check or rely on gain==0.
    # If gain==0, it means no new points are covered by any unpicked codeword.
    # This implies all points are covered.
    
    result = [all_words[i] for i in code_indices]
    return [list(w) for w in result]
