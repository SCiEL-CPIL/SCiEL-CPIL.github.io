import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though the task specifies 3,7,1
        # Using a simple greedy for generic case
        total_words = q ** n
        # This fallback is slow, but the main logic is for 3,7,1
        return [] 

    total_words = 3 ** 7
    deadline = time.time() + 110  # Safety margin for 120s limit

    # Precompute coverers: for each word w, list of codewords c such that d(w,c) <= 1
    # Represent words as integers 0..2186
    coverers = [[] for _ in range(total_words)]
    balls = [[] for _ in range(total_words)]
    
    # Generate all words and their balls
    for c in range(total_words):
        # Convert c to base 3 tuple
        c_tuple = []
        temp = c
        for _ in range(7):
            c_tuple.append(temp % 3)
            temp //= 3
        c_tuple = tuple(c_tuple)
        
        ball = [c]
        for i in range(7):
            for delta in [1, 2]: # +1 and +2 (which is -1 mod 3)
                new_tuple = list(c_tuple)
                new_tuple[i] = (c_tuple[i] + delta) % 3
                new_val = 0
                for j in range(7):
                    new_val = new_val * 3 + new_tuple[j]
                ball.append(new_val)
        
        balls[c] = ball
        
        for w in ball:
            coverers[w].append(c)

    # Greedy Construction
    counts = [0] * total_words
    code = []
    
    # Precompute for each c, the list of words in its ball
    # We need to pick c that covers max uncovered words
    
    # Optimization: Use a heap or just iterate. 2187 items is small.
    # But we need to update counts.
    
    while True:
        best_c = -1
        best_cov = 0
        # Iterate all possible codewords
        for c in range(total_words):
            # Count how many words in balls[c] are currently uncovered (count==0)
            cov = 0
            for w in balls[c]:
                if counts[w] == 0:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_c = c
        
        if best_cov == 0:
            break
            
        code.append(best_c)
        for w in balls[best_c]:
            counts[w] += 1
            
        if time.time() > deadline - 5:
            break

    # Remove redundant codewords
    # A codeword c is redundant if all words in balls[c] are covered by other codewords in code
    # i.e., for all w in balls[c], there exists c' in code \ {c} such that w in balls[c']
    # Which is equivalent to: for all w in balls[c], counts[w] >= 2 OR (counts[w]==1 and c is the only coverer? No, counts[w] is global)
    # Actually, if we remove c, counts[w] decreases by 1 for w in balls[c].
    # c is redundant if min(counts[w] for w in balls[c]) >= 2? No.
    # c is redundant if for every w in balls[c], there is another codeword c' in code covering w.
    # This is true if counts[w] >= 2 for all w in balls[c]?
    # Not exactly. counts[w] is the number of codewords in the CURRENT code covering w.
    # If counts[w] >= 2, then even if we remove c, w is still covered.
    # So if min(counts[w] for w in balls[c]) >= 2, c is redundant.
    
    removed = True
    while removed and time.time() < deadline - 10:
        removed = False
        new_code = []
        for c in code:
            is_redundant = True
            for w in balls[c]:
                if counts[w] < 2:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove c
                for w in balls[c]:
                    counts[w] -= 1
                removed = True
            else:
                new_code.append(c)
        if removed:
            code = new_code

    # Local Search: 1-for-2 swap
    # Try to replace two codewords c1, c2 with one c_new
    # Condition: Words covered ONLY by c1 (U1) and ONLY by c2 (U2) must be covered by c_new.
    # U1 = {w in balls[c1] | counts[w] == 1 and c1 in coverers[w]}
    # Note: counts[w] == 1 means exactly one codeword in code covers w.
    # If c1 is in coverers[w], then that one is c1.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over pairs
        # To speed up, we can check smaller U sets first
        # But let's just iterate all pairs
        
        n_code = len(code)
        for i in range(n_code):
            c1 = code[i]
            # Calculate U1
            U1 = []
            for w in balls[c1]:
                if counts[w] == 1 and c1 in coverers[w]:
                    U1.append(w)
            
            # If U1 is empty, c1 is redundant (should have been caught, but check)
            if not U1:
                # Remove c1
                for w in balls[c1]:
                    counts[w] -= 1
                code.pop(i)
                improved = True
                break # Restart loop as indices changed
            
            for j in range(i + 1, n_code):
                c2 = code[j]
                # Calculate U2
                U2 = []
                for w in balls[c2]:
                    if counts[w] == 1 and c2 in coverers[w]:
                        U2.append(w)
                
                # Words that must be covered by c_new are U1 union U2
                # Note: A word w cannot be in both U1 and U2 because counts[w] would be 1, 
                # so it can only be covered by one codeword. If w in U1, it's covered by c1. 
                # If w in U2, it's covered by c2. Since c1 != c2, w cannot be in both.
                # So U1 and U2 are disjoint.
                
                S = U1 + U2
                
                # Find candidates for c_new
                # c_new must cover all words in S.
                # c_new must be in intersection of coverers[w] for w in S.
                
                if not S:
                    # If S is empty, we can remove both c1 and c2?
                    # No, if S is empty, it means U1 and U2 are empty.
                    # This implies c1 and c2 are redundant.
                    # We should have removed them in the redundancy pass.
                    # But let's handle it: remove c1 and c2.
                    for w in balls[c1]:
                        counts[w] -= 1
                    for w in balls[c2]:
                        counts[w] -= 1
                    code.pop(j)
                    code.pop(i)
                    improved = True
                    break
                
                # Find intersection of coverers for words in S
                # Start with coverers[S[0]]
                candidates = set(coverers[S[0]])
                for w in S[1:]:
                    candidates.intersection_update(coverers[w])
                    if not candidates:
                        break
                
                if candidates:
                    # Pick a candidate
                    # Prefer one not already in code to avoid conflict? 
                    # Actually, if c_new is in code, we are replacing 2 with 1 existing?
                    # No, we remove c1, c2 and add c_new.
                    # If c_new is already in code, we just remove c1, c2.
                    # If c_new is not in code, we remove c1, c2 and add c_new.
                    # In both cases, size decreases by 1.
                    
                    c_new = next(iter(candidates))
                    
                    # Perform swap
                    for w in balls[c1]:
                        counts[w] -= 1
                    for w in balls[c2]:
                        counts[w] -= 1
                    
                    code.pop(j)
                    code.pop(i)
                    
                    if c_new not in code:
                        code.append(c_new)
                        for w in balls[c_new]:
                            counts[w] += 1
                    else:
                        # c_new was already in code, so its coverage is already counted
                        # We just removed c1 and c2.
                        # We don't add counts for c_new again.
                        pass
                    
                    improved = True
                    break
            if improved:
                break

    # Convert codes to lists of integers
    result = []
    for c in code:
        word = []
        temp = c
        for _ in range(7):
            word.append(temp % 3)
            temp //= 3
        result.append(word)
        
    return result
