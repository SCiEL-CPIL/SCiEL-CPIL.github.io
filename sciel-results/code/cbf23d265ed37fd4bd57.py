import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120  # Leave some margin for 150s limit
    
    # Fallback: Tetracode sum
    def tetracode_sum():
        if n < 4:
            # Handle small n
            # For n=1, q=3, r=1: need all 3 words? No, dist 1 covers all.
            # Sphere size 1 + 1*(3-1) = 3. Space size 3. 1 word covers all.
            # So for n=1, return [[0]].
            # For n=2, q=3, r=1. Sphere 1+2*2=5. Space 9. Need ceil(9/5)=2?
            # Just use a greedy for small n or return full space if too small.
            pass
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific solver for q=3, n=6, r=1
    if q == 3 and n == 6 and radius == 1:
        # Precompute neighbors
        # Map word to index 0..728
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * 3 + w[i]
            return idx
        
        def idx_to_word(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % 3
                idx //= 3
            return w

        neighbors = [[] for _ in range(729)]
        for i in range(729):
            w = idx_to_word(i)
            neighbors[i].append(i)
            for pos in range(n):
                orig = w[pos]
                for val in range(3):
                    if val != orig:
                        w[pos] = val
                        neighbors[i].append(word_to_idx(w))
            w[pos] = orig # restore, though w is local

        # Initial code: Tetracode sum (81 words)
        initial_code = tetracode_sum()
        initial_indices = [word_to_idx(w) for w in initial_code]
        
        best_code = initial_code
        best_size = len(initial_code)
        
        # Local search: Greedy deletion with random restarts
        # We will try many random orderings
        
        max_restarts = 5000 # Adjust based on time
        
        for _ in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Shuffle the indices
            shuffled_indices = initial_indices[:]
            random.shuffle(shuffled_indices)
            
            # counts array
            counts = [0] * 729
            current_code_indices = []
            
            # Add all initial words to counts
            for idx in shuffled_indices:
                for nb in neighbors[idx]:
                    counts[nb] += 1
                current_code_indices.append(idx)
            
            # Greedy deletion pass
            # We iterate through the shuffled list.
            # If a word is redundant, we remove it.
            # Note: We must check redundancy against the *current* set of kept words.
            # Since we are iterating and removing, we need to be careful.
            # Standard approach:
            # 1. Start with all words.
            # 2. For each word in random order:
            #    Check if removing it leaves all its neighbors covered by OTHERS.
            #    This is equivalent to: for all nb in neighbors[word], counts[nb] > 1.
            #    If yes, remove it (decrement counts).
            #    If no, keep it.
            
            # However, the "counts" array reflects the state BEFORE the decision for this word?
            # If we process word W, and decide to remove it, we decrement counts.
            # If we keep it, counts stay.
            # The check `counts[nb] > 1` ensures that even if we remove W, the count drops to >= 1.
            # This is correct.
            
            for idx in shuffled_indices:
                # Check if idx is currently in the active set?
                # We are iterating through a list of all initial words.
                # If we already removed it in a previous step? 
                # No, each word appears once in shuffled_indices.
                # So we only make the decision once per restart.
                
                redundant = True
                for nb in neighbors[idx]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove idx
                    for nb in neighbors[idx]:
                        counts[nb] -= 1
                    # Do not add to final list
                else:
                    # Keep idx
                    current_code_indices.append(idx) # Wait, I added them all initially?
            
            # Correction: The logic above is slightly flawed for a single pass if we don't manage the set properly.
            # Let's restart the logic for the inner loop.
            
            # Reset counts for this restart
            counts = [0] * 729
            for idx in shuffled_indices:
                for nb in neighbors[idx]:
                    counts[nb] += 1
            
            kept_indices = []
            
            for idx in shuffled_indices:
                # Check if idx is redundant given the current counts (which include idx)
                redundant = True
                for nb in neighbors[idx]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove idx
                    for nb in neighbors[idx]:
                        counts[nb] -= 1
                else:
                    kept_indices.append(idx)
            
            if len(kept_indices) < best_size:
                best_size = len(kept_indices)
                best_code = [idx_to_word(idx) for idx in kept_indices]
                
        return best_code

    else:
        # General case: use tetracode sum if possible, else greedy
        if n >= 4:
            return tetracode_sum()
        else:
            # Simple greedy for small n
            # Not strictly required if task is specific, but good for robustness
            pass

    # Fallback if not handled
    return tetracode_sum()
