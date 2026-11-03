import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute neighbors (ball of radius 1) for each word index
    # Word index is base-q representation.
    # We need to map index to digits to find neighbors.
    
    def to_digits(val):
        digits = [0] * n
        for i in range(n):
            digits[i] = val % q
            val //= q
        return digits

    def to_int(digits):
        val = 0
        for d in digits:
            val = val * q + d
        return val

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        digits = to_digits(i)
        # Center
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    neighbors[i].append(to_int(new_digits))
                    
    # Greedy Initialization
    def greedy_init():
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        code = []
        
        while uncovered_count > 0:
            best_c = -1
            best_gain = -1
            
            # We can optimize this by only checking candidates that might improve.
            # But with N=2187, checking all is fine.
            for c in range(total_words):
                gain = 0
                for nb in neighbors[c]:
                    if not covered[nb]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            if best_gain == 0:
                # Should not happen if uncovered_count > 0 and we check all
                break
                
            code.append(best_c)
            for nb in neighbors[best_c]:
                if not covered[nb]:
                    covered[nb] = True
                    uncovered_count -= 1
                    
        return code

    # Local Search: Removal and Swap
    def local_search(code):
        if not code:
            return code
            
        current_code = code[:]
        
        # cover_count[w] = number of codewords in current_code covering w
        cover_count = np.zeros(total_words, dtype=np.int32)
        
        def update_cover_count(code_list):
            cc = np.zeros(total_words, dtype=np.int32)
            for c in code_list:
                for nb in neighbors[c]:
                    cc[nb] += 1
            return cc

        cover_count = update_cover_count(current_code)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try Removals
            # Find a codeword that can be removed
            removable_idx = -1
            for i, c in enumerate(current_code):
                # Check if all neighbors have count > 1
                # If min count in neighbors[c] > 1, it is removable
                min_count = min(cover_count[nb] for nb in neighbors[c])
                if min_count > 1:
                    removable_idx = i
                    break
            
            if removable_idx != -1:
                # Remove it
                c_removed = current_code.pop(removable_idx)
                for nb in neighbors[c_removed]:
                    cover_count[nb] -= 1
                improved = True
                continue
            
            # 2. Try Swaps (1-for-1)
            # For each c_old, find c_new such that replacing c_old with c_new is valid
            # Condition: For all w in B(c_old) \ B(c_new), cover_count[w] >= 2
            # (Because c_old contributes 1, so if we remove it, count drops by 1.
            #  If new count >= 1, then old count >= 2)
            
            swap_found = False
            # Shuffle to randomize search order
            indices = list(range(len(current_code)))
            random.shuffle(indices)
            
            for i in indices:
                c_old = current_code[i]
                bad_words = []
                for nb in neighbors[c_old]:
                    if cover_count[nb] == 1:
                        bad_words.append(nb)
                        
                if not bad_words:
                    # This would have been caught by removal, but just in case
                    # Actually, if bad_words is empty, min count > 1, so it's removable.
                    # But we already checked removal. So this shouldn't happen.
                    continue
                
                # Find a c_new that covers all bad_words
                # And c_new is not already in the code (or if it is, it's a swap with itself? No, distinct codewords required usually, but even if not, swapping with same is no-op)
                # We want to replace c_old with c_new.
                # c_new can be any word in the space.
                
                # Optimization: Precompute for each word w, which codewords cover it?
                # Or just iterate c_new.
                for c_new in range(total_words):
                    if c_new == c_old:
                        continue
                    # Check if c_new covers all bad_words
                    # bad_words is small (<= 15)
                    # We can check if set(bad_words) is subset of set(neighbors[c_new])
                    # But creating sets is slow.
                    # Just check each w in bad_words if w in neighbors[c_new]
                    # neighbors[c_new] is a list of 15 ints.
                    # We can convert neighbors[c_new] to a set? 
                    # Or just use 'in' on the list. 15 elements is fast.
                    
                    valid = True
                    for w in bad_words:
                        if w not in neighbors[c_new]:
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        # Remove c_old
                        for nb in neighbors[c_old]:
                            cover_count[nb] -= 1
                        # Add c_new
                        for nb in neighbors[c_new]:
                            cover_count[nb] += 1
                        
                        current_code[i] = c_new
                        improved = True
                        swap_found = True
                        break
                        
                if swap_found:
                    break # Restart the while loop
            
        return current_code

    best_solution = None
    best_len = float('inf')
    
    # Iterated Local Search
    seed = 0
    while time.time() < deadline:
        # 1. Greedy Init
        code = greedy_init()
        
        # 2. Local Search
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_solution = code[:]
            
        # 3. Perturb
        if best_solution:
            perturbed = best_solution[:]
            if len(perturbed) > 0:
                # Pick a random codeword to replace
                idx = random.randint(0, len(perturbed) - 1)
                # Replace with a random word from the space
                new_c = random.randint(0, total_words - 1)
                perturbed[idx] = new_c
                
            # 4. Local Search on Perturbed
            code = local_search(perturbed)
            
            if len(code) < best_len:
                best_len = len(code)
                best_solution = code[:]
                
        seed += 1
        if time.time() > deadline:
            break

    if best_solution is None:
        # Fallback: Greedy only
        code = greedy_init()
        return [to_digits(c) for c in code]
        
    return [to_digits(c) for c in best_solution]
