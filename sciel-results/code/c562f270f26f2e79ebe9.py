import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit for safety
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters if any, though task is specific
        # Simple random or greedy for generic case
        pass

    # Specific optimization for q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        # Precompute neighbors for all possible codewords
        # neighbors[i] is a list of 9 integers (the word and its 8 neighbors)
        # Word representation: integer 0..6560
        neighbors = []
        for i in range(total_words):
            # Convert i to base-3 digits
            digits = []
            temp = i
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            
            nb = [i]
            for pos in range(n):
                original = digits[pos]
                for val in range(3):
                    if val == original:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    # Convert back to int
                    new_val = 0
                    for d in new_digits:
                        new_val = new_val * 3 + d
                    nb.append(new_val)
            neighbors.append(nb)

        # Greedy Set Cover
        # covered[w] = 1 if covered, 0 otherwise
        covered = [0] * total_words
        C = []
        
        # Initial gain for all candidates is 9
        # But we can just compute gain on the fly to save memory and init time
        # gain[c] = sum(1 - covered[w] for w in neighbors[c])
        
        while True:
            # Check time
            if time.time() > deadline:
                break
            
            best_gain = 0
            best_c = -1
            
            # Find candidate with max gain
            # Optimization: skip candidates already in C? 
            # Actually, if a candidate is in C, its gain is 0 because its neighbors are covered.
            # But other candidates might have gain.
            # We must check all candidates not in C? 
            # If we add a codeword, its neighbors are covered.
            # Any codeword sharing a neighbor with an existing codeword has reduced gain.
            
            # To speed up, we can maintain a list of "active" candidates?
            # Or just iterate all 6561. 6561 is small.
            
            for c in range(total_words):
                if c in C: # This check is slow if C is a list. Use a set or boolean array.
                    continue
                gain = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    if gain == 9: # Max possible
                        # Can't do better
                        # But we should check if there are others with 9?
                        # Greedy usually picks the first one found.
                        # Breaking early might miss a better distribution?
                        # For covering codes, any max gain is fine.
                        pass
            
            if best_gain == 0:
                # All covered?
                break
            
            # Add best_c to C
            C.append(best_c)
            
            # Update covered
            for w in neighbors[best_c]:
                covered[w] = 1
            
            # Check if all covered
            # We can check if best_gain was 0 in the next iteration
            # Or check sum(covered) == total_words
            
        # Removal Phase
        # cover_count[w] = number of codewords in C covering w
        cover_count = [0] * total_words
        for c in C:
            for w in neighbors[c]:
                cover_count[w] += 1
        
        # Iterative removal
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate over a copy of C to allow removal
            # Randomize order to escape local minima?
            # Or fixed order.
            for i in range(len(C) - 1, -1, -1):
                c = C[i]
                # Check if removal is safe
                # Safe if min(cover_count[w] for w in neighbors[c]) >= 2
                safe = True
                for w in neighbors[c]:
                    if cover_count[w] < 2:
                        safe = False
                        break
                
                if safe:
                    # Remove c
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    C.pop(i)
                    improved = True
                    # Check time
                    if time.time() > deadline:
                        break
        
        # Convert C back to list of lists
        result = []
        for c in C:
            digits = []
            temp = c
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
            
        return result
    else:
        # Generic fallback: Random or simple greedy
        # For this task, only 3,8,1 is expected
        # Return a trivial cover (all words) if not specific
        return [list(map(int, str(i).zfill(n))) for i in range(q**n)] # Very bad, but valid
