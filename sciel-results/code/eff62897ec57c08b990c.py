import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Precompute all words and their neighbors
    # Map word tuple to integer index
    # Since q=3, n=7, total words = 3**7 = 2187
    total_words = q ** n
    
    # Generate all words as tuples
    words = []
    for i in range(total_words):
        w = []
        val = i
        for _ in range(n):
            w.append(val % q)
            val //= q
        words.append(tuple(w))
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= radius
    # For radius 1, neighbors are the word itself and words with one symbol changed
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = words[i]
        # The word itself
        neighbors[i].append(i)
        # Neighbors with 1 change
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    # Create new word
                    new_w = list(w)
                    new_w[pos] = new_val
                    # Find index of new word
                    # Instead of searching, we can compute index directly
                    idx = 0
                    mult = 1
                    for k in range(n):
                        idx += new_w[k] * mult
                        mult *= q
                    neighbors[i].append(idx)
    
    # 2. Greedy Construction
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    # To speed up greedy, we can maintain counts of uncovered neighbors
    # uncovered_count[i] = number of uncovered words in neighbors[i]
    uncovered_count = np.array([len(neighbors[i]) for i in range(total_words)], dtype=np.int32)
    # Wait, uncovered_count should be the number of words in neighbors[i] that are NOT covered.
    # Initially, all are uncovered. So uncovered_count[i] = len(neighbors[i]).
    
    # When we add a word c, we cover neighbors[c].
    # For every word i, if any neighbor of i is in neighbors[c] and was uncovered, uncovered_count[i] decreases.
    # This is complex to update efficiently without iterating over all i.
    # Given N=2187, iterating over all i to find max is $O(N)$.
    # Updating uncovered_count after adding c:
    # We need to decrement uncovered_count[i] for all i such that neighbors[i] intersects neighbors[c] in uncovered words.
    # Actually, simpler:
    # Just compute the gain for each candidate i from scratch? $O(N \times \text{avg\_neighbors})$.
    # $2187 \times 15 \approx 32000$ ops per step.
    # Number of steps ~ 200. Total ops ~ 6.4M. This is very fast in Python if optimized, or NumPy.
    
    # Let's use a simpler greedy loop:
    while not np.all(covered):
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_gain = -1
        
        # Vectorized gain calculation
        # For each candidate i, gain = sum(1 for j in neighbors[i] if not covered[j])
        # We can precompute a matrix M where M[i, j] = 1 if j in neighbors[i].
        # Then gain = M @ (~covered).astype(int)
        
        # Let's build the adjacency matrix once.
        # It's sparse, but N is small. Dense matrix 2187x2187 is ~4.7M elements. 
        # 4.7M * 4 bytes = ~19 MB. Acceptable.
        
        if not hasattr(construct, 'adj_matrix'):
            construct.adj_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
            for i in range(total_words):
                for j in neighbors[i]:
                    construct.adj_matrix[i, j] = 1
        
        M = construct.adj_matrix
        # gain[i] = sum(M[i] * ~covered)
        gains = M.dot(~covered)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break # Should not happen if not all covered
            
        code.append(best_idx)
        covered[neighbors[best_idx]] = True
        
    # 3. Local Search
    # We have a valid cover. Now try to minimize size.
    # Strategy: 
    # 1. Remove redundant codewords.
    # 2. Try swaps: replace c with u if it allows removal of another codeword or reduces size.
    
    # Maintain counts: count[j] = number of codewords covering j
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code:
        counts[neighbors[c]] += 1
        
    # Helper to check if removing c is safe
    # Safe if min(counts[neighbors[c]]) > 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if any neighbor has count 1
            if np.min(counts[neighbors[c]]) > 1:
                # Remove c
                counts[neighbors[c]] -= 1
                code.pop(i)
                improved = True
                # Restart removal pass? Or continue? 
                # Continuing is fine, but indices shift. 
                # Since we iterate backwards, popping is safe.
                # But counts changed, so earlier checks might be invalid? 
                # No, removing a word only decreases counts. 
                # If a word was safe before, it might become unsafe? 
                # No, if min(counts) > 1 before, and we decrease counts, 
                # it might drop to 1. So it's not safe to just continue.
                # We should restart the removal loop or be careful.
                # Given small size, restarting is safer.
                break
        
        if time.time() > deadline:
            break
            
        # Pass 2: Try to improve via swaps or additions
        # This is the expensive part.
        # We can try: For each c in code, try to replace it with a random u not in code.
        # If the new code is smaller after cleanup, accept.
        
        # To keep it fast, we limit the number of swap attempts.
        num_swaps = 100
        for _ in range(num_swaps):
            if time.time() > deadline:
                break
            
            if len(code) < 2:
                break
                
            # Pick a random codeword to remove
            c_idx = random.randint(0, len(code) - 1)
            c = code[c_idx]
            
            # Tentatively remove c
            counts[neighbors[c]] -= 1
            code.pop(c_idx)
            
            # Now we have a "hole" or uncovered words?
            # Actually, we just removed c. Some words might now have count 0.
            # We need to add words to cover them, or find a swap.
            
            # Let's try to find a word u that covers the most "deficient" words (count == 0 or low count).
            # Actually, a simpler heuristic: 
            # If removing c made the cover invalid, we must add something.
            # Let's check if cover is still valid.
            if np.any(counts == 0):
                # Find words with count 0
                uncovered_idx = np.where(counts == 0)[0]
                
                # Find best u to add
                # Gain of u = number of uncovered words it covers
                # We can just pick the u that covers the most uncovered words.
                # This is similar to greedy step.
                
                # To save time, just pick one randomly from candidates?
                # Or do a quick greedy step.
                
                # Let's try all u not in code? Too slow.
                # Let's try a few random u's.
                best_u = -1
                best_gain = -1
                
                # Sample some candidates
                # Candidates are likely near the uncovered words.
                # Let's just pick a random word from the whole space.
                # Or better: pick a word that is a neighbor of an uncovered word.
                
                candidate_pool = []
                for u_idx in uncovered_idx:
                    candidate_pool.extend(neighbors[u_idx])
                
                # Remove duplicates
                candidate_pool = list(set(candidate_pool))
                
                # Filter out those already in code
                code_set = set(code)
                candidate_pool = [u for u in candidate_pool if u not in code_set]
                
                if not candidate_pool:
                    # Revert
                    code.insert(c_idx, c)
                    counts[neighbors[c]] += 1
                    continue
                
                # Evaluate candidates
                # To be fast, evaluate only a subset
                random.shuffle(candidate_pool)
                for u in candidate_pool[:20]:
                    # Gain = number of words in neighbors[u] with count 0
                    gain = np.sum(counts[neighbors[u]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                        
                if best_u != -1 and best_gain > 0:
                    # Add best_u
                    code.append(best_u)
                    counts[neighbors[best_u]] += 1
                    # Now try to remove redundant words again
                    # This is handled by the next iteration of the outer loop
                    improved = True
                else:
                    # Revert
                    code.insert(c_idx, c)
                    counts[neighbors[c]] += 1
            else:
                # Cover is still valid! We successfully removed a word.
                # This is a pure improvement.
                improved = True
                # No need to add back.

    # Convert indices back to word lists
    result = []
    for idx in code:
        result.append(list(words[idx]))
        
    return result

# Note: The above logic has a flaw in the swap section.
# If removing c keeps the cover valid, we just removed a word. That's great.
# If removing c breaks the cover, we try to add a word.
# This effectively performs a "remove and repair" step.
# It might not find the global optimum but should improve on greedy.

# Let's refine the "Remove and Repair" to be more robust.
# Also, the `construct.adj_matrix` caching might fail if `n` or `q` changes, but the task says `n=7, q=3` is fixed for this run?
# "The evaluator passes this task's alphabet size q, length n and covering radius."
# It implies `construct` might be called with different parameters?
# "Keep the function names and signatures...".
# "Score: the number of codewords...".
# The prompt says "Football pool problem for 7 matches".
# It's likely a single fixed instance.
# However, to be safe, we should not cache based on global state if parameters can vary.
# But 150s limit suggests a single heavy run.
# I will assume n=7, q=3, radius=1.

# Let's rewrite cleanly with a focus on correctness and speed for n=7.

import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    total_words = q ** n
    words = []
    for i in range(total_words):
        w = []
        val = i
        for _ in range(n):
            w.append(val % q)
            val //= q
        words.append(tuple(w))
        
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = words[i]
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    idx = 0
                    mult = 1
                    for k in range(n):
                        idx += new_w[k] * mult
                        mult *= q
                    neighbors[i].append(idx)
                    
    # Adjacency matrix for fast gain calculation
    # M[i, j] = 1 if j in neighbors[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in neighbors[i]:
            M[i, j] = 1
            
    # Greedy
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    while not np.all(covered):
        if time.time() > deadline:
            break
        gains = M.dot(~covered)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        code.append(best_idx)
        covered[neighbors[best_idx]] = True
        
    # Local Search
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code:
        counts[neighbors[c]] += 1
        
    code_set = set(code)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Remove redundant
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if np.min(counts[neighbors[c]]) > 1:
                counts[neighbors[c]] -= 1
                code.pop(i)
                code_set.remove(c)
                improved = True
                break # Restart loop to be safe with indices/counts
        
        if time.time() > deadline:
            break
            
        # 2. Try to improve by swapping
        # We try to remove a word and replace it with a better one, or just remove if possible.
        # The "Remove redundant" pass above handles pure removals.
        # Now we try "Swap": Remove c, if invalid, add u. If valid, keep removed.
        # This is effectively what the previous logic did, but let's make it explicit and robust.
        
        # We perform a limited number of swap attempts per pass
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline or len(code) < 2:
                break
                
            c_idx = random.randint(0, len(code) - 1)
            c = code[c_idx]
            
            # Try removing c
            counts[neighbors[c]] -= 1
            code.pop(c_idx)
            code_set.remove(c)
            
            if np.any(counts == 0):
                # Need to repair
                uncovered_idx = np.where(counts == 0)[0]
                
                # Find best replacement
                # Candidates are neighbors of uncovered words
                candidates = set()
                for u in uncovered_idx:
                    candidates.update(neighbors[u])
                candidates -= code_set
                
                if not candidates:
                    # Revert
                    code.insert(c_idx, c)
                    code_set.add(c)
                    counts[neighbors[c]] += 1
                    continue
                    
                # Evaluate top few candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                best_u = -1
                best_gain = -1
                
                for u in cand_list[:30]:
                    gain = np.sum(counts[neighbors[u]] == 0)
                    # Tie-breaker: maybe prefer u that covers more low-count words?
                    # For now, just max uncovered.
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                        
                if best_u != -1 and best_gain > 0:
                    code.append(best_u)
                    code_set.add(best_u)
                    counts[neighbors[best_u]] += 1
                    improved = True
                else:
                    # Revert
                    code.insert(c_idx, c)
                    code_set.add(c)
                    counts[neighbors[c]] += 1
            else:
                # Successfully removed c, cover still valid
                improved = True

    result = []
    for idx in code:
        result.append(list(words[idx]))
    return result
