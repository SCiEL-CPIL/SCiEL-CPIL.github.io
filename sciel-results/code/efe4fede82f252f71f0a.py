import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # But the function signature is general, so we handle general case?
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we should assume q=3, n=6, radius=1 but write code that works for these.
    # For other parameters, we might not have a good strategy, but we can try the same heuristic.
    
    total_words = q ** n
    
    # Precompute neighbors for each word in {0..q-1}^n
    # A word is an index 0..total_words-1.
    # We need to convert index to digits and back, or just generate neighbors by index manipulation.
    
    def index_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_index(digits):
        idx = 0
        for i in range(n):
            idx = idx * q + digits[i]
        return idx

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        digits = index_to_digits(i)
        neighbors[i].append(i)
        for pos in range(n):
            original = digits[pos]
            for val in range(q):
                if val == original:
                    continue
                digits[pos] = val
                neighbors[i].append(digits_to_index(digits))
            digits[pos] = original

    # Greedy Initialization
    counts = np.zeros(total_words, dtype=np.int32)
    code_set = set()
    code_list = []
    
    # To speed up greedy, we can track uncovered words
    # But with 729 words, simple loop is fine.
    while True:
        # Check if all covered
        if np.all(counts > 0):
            break
        
        best_w = -1
        best_gain = -1
        # Random tie breaker
        random.shuffle(range(total_words)) # This is O(N log N), maybe just iterate and randomize tie
        # Better:
        for w in range(total_words):
            gain = 0
            for nb in neighbors[w]:
                if counts[nb] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_w = w
            elif gain == best_gain and best_w != -1:
                if random.random() < 0.5:
                    best_w = w

        if best_w == -1:
            break # Should not happen if radius >= 1 and q >= 2
            
        code_set.add(best_w)
        code_list.append(best_w)
        for nb in neighbors[best_w]:
            counts[nb] += 1

    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    best_code = code_list[:]
    best_size = len(code_list)
    best_counts = counts.copy()
    
    # Local Search
    while time.time() - start_time < deadline:
        # 1. Try to remove a word
        removable = []
        for c in code_list:
            # Check if c is redundant
            redundant = True
            for nb in neighbors[c]:
                if counts[nb] == 1:
                    redundant = False
                    break
            if redundant:
                removable.append(c)
        
        if removable:
            c_to_remove = random.choice(removable)
            code_list.remove(c_to_remove)
            code_set.remove(c_to_remove)
            for nb in neighbors[c_to_remove]:
                counts[nb] -= 1
            
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = code_list[:]
                best_counts = counts.copy()
            
            if best_size <= 73: # Known optimum for 3,6,1
                break
            continue
        
        # 2. Try to swap
        # Calculate unique counts for each c in code
        unique_counts = {}
        for c in code_list:
            u = 0
            for nb in neighbors[c]:
                if counts[nb] == 1:
                    u += 1
            unique_counts[c] = u
        
        # Sort by unique count ascending
        sorted_c = sorted(code_list, key=lambda x: unique_counts[x])
        
        swap_made = False
        # Try swapping the most "redundant" words first
        for c_out in sorted_c:
            if swap_made:
                break
            unique_nbs = [nb for nb in neighbors[c_out] if counts[nb] == 1]
            
            # If no unique neighbors, it should have been removed.
            # So unique_nbs is not empty.
            
            # Find a c_in that covers all unique_nbs
            # To make it faster, we can filter candidates
            # A candidate c_in must cover all w in unique_nbs.
            # This means c_in must be within distance 1 of all w in unique_nbs.
            # Since unique_nbs are all within distance 1 of c_out,
            # and c_out is within distance 1 of them...
            # Actually, if c_out covers w, d(c_out, w) <= 1.
            # If c_in covers w, d(c_in, w) <= 1.
            
            # Let's just check all candidates not in code_set
            # Optimization: If unique_nbs is large, it's harder to find such c_in.
            
            candidates = [w for w in range(total_words) if w not in code_set]
            random.shuffle(candidates)
            
            for c_in in candidates:
                # Check if c_in covers all unique_nbs
                valid_swap = True
                for nb in unique_nbs:
                    if nb not in neighbors[c_in]:
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Perform swap
                    code_list.remove(c_out)
                    code_set.remove(c_out)
                    for nb in neighbors[c_out]:
                        counts[nb] -= 1
                    
                    code_list.append(c_in)
                    code_set.add(c_in)
                    for nb in neighbors[c_in]:
                        counts[nb] += 1
                    
                    swap_made = True
                    break
            
            if swap_made:
                break
        
        if not swap_made:
            # Stuck?
            # Try a random valid move?
            # Or wait?
            # To avoid infinite loop with no progress, we could try to add a word and then remove two.
            # But for now, just idle or randomize.
            # Let's try a few random swaps ignoring the "unique" constraint, 
            # but checking validity explicitly.
            for _ in range(5):
                c_out = random.choice(code_list)
                c_in = random.choice([w for w in range(total_words) if w not in code_set])
                
                # Check validity of swap
                # Remove c_out, Add c_in
                # Check if any count becomes 0
                temp_counts = counts.copy()
                for nb in neighbors[c_out]:
                    temp_counts[nb] -= 1
                for nb in neighbors[c_in]:
                    temp_counts[nb] += 1
                
                if np.all(temp_counts > 0):
                    # Accept
                    code_list.remove(c_out)
                    code_set.remove(c_out)
                    for nb in neighbors[c_out]:
                        counts[nb] -= 1
                    code_list.append(c_in)
                    code_set.add(c_in)
                    for nb in neighbors[c_in]:
                        counts[nb] += 1
                    break

    # Convert best_code indices to list of digits
    result = []
    for idx in best_code:
        result.append(index_to_digits(idx))
    
    return result
