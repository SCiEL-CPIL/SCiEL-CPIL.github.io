import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Parameters specific to the task instance
    # q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        pass 
        
    N = q ** n
    # Precompute neighbors (ball of radius 1)
    # word is int 0..N-1
    # neighbors[word] is list of ints
    neighbors = [[] for _ in range(N)]
    
    # Helper to convert int to list of digits
    def int_to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
        
    # Helper to convert list of digits to int
    def digits_to_int(digits, q):
        x = 0
        for d in reversed(digits):
            x = x * q + d
        return x

    for w in range(N):
        digits = int_to_digits(w, n, q)
        # Center
        neighbors[w].append(w)
        # Radius 1: change one digit to other q-1 values
        for i in range(n):
            orig = digits[i]
            for v in range(q):
                if v == orig: continue
                new_digits = digits[:]
                new_digits[i] = v
                neighbors[w].append(digits_to_int(new_digits, q))
                
    covered = [False] * N
    cover_count = [0] * N
    code = []
    
    # Greedy Phase
    while True:
        if time.time() > deadline - 10.0: break
        if all(covered): break
        
        best_word = -1
        best_gain = -1
        
        # Optimization: Only consider words that cover at least one uncovered word?
        # Or just scan all.
        for w in range(N):
            # Quick check: if w is already in code? No, we add distinct words.
            # But a word not in code might still be a good candidate.
            # Actually, if a word is already in code, we shouldn't add it again.
            # We can maintain a set of selected words.
            pass
        
        # To avoid re-adding, we can just check if w is in code_set
        code_set = set(code)
        
        for w in range(N):
            if w in code_set:
                continue
            gain = 0
            for x in neighbors[w]:
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
                
        if best_word == -1:
            break
            
        # Add best_word
        code.append(best_word)
        code_set.add(best_word)
        for x in neighbors[best_word]:
            if not covered[x]:
                covered[x] = True
            cover_count[x] += 1
            
        if time.time() > deadline - 10.0: break

    # Local Optimization: Redundancy Removal
    # Repeat until no change
    changed = True
    while changed:
        if time.time() > deadline: break
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            for x in neighbors[c]:
                # If cover_count[x] is 1, it means only c covers x (since we haven't removed c yet, 
                # and cover_count includes c).
                # So if any x has count 1, c is NOT redundant.
                if cover_count[x] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for x in neighbors[c]:
                    cover_count[x] -= 1
                code.pop(i)
                code_set.remove(c)
                changed = True
                if time.time() > deadline: break

    # Convert back to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result
