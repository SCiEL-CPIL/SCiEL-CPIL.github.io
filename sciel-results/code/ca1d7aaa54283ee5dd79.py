import time
import random
import math

def _precompute_neighbors(q, n, radius):
    """
    Precompute neighbors for all words in {0..q-1}^n.
    Returns a list of lists, where neighbors[i] is a list of indices
    for words within Hamming distance <= radius of word i.
    """
    total = q ** n
    neigh = [[] for _ in range(total)]
    
    for idx in range(total):
        # The word itself
        neigh[idx].append(idx)
        
        if radius >= 1:
            # Words at distance 1
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        
        if radius >= 2:
             # Words at distance 2
             # This is O(n^2 * q^2) per word, which is fine for n=6, q=3
             for i in range(n):
                 for j in range(i + 1, n):
                     pow_q_i = q ** i
                     pow_q_j = q ** j
                     digit_i = (idx // pow_q_i) % q
                     digit_j = (idx // pow_q_j) % q
                     
                     for new_i in range(q):
                         if new_i == digit_i: continue
                         for new_j in range(q):
                             if new_j == digit_j: continue
                             new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                             neigh[idx].append(new_idx)

    # Remove duplicates just in case (though logic above avoids them for distinct radii)
    for i in range(total):
        neigh[i] = list(set(neigh[i]))
        
    return neigh, total

def _greedy_init(q, n, radius, neigh, total, rng):
    """
    Greedy set cover initialization.
    """
    covered = [False] * total
    code = []
    
    # Start with 0 to break symmetry
    start_idx = 0
    code.append(start_idx)
    for u in neigh[start_idx]:
        covered[u] = True
        
    rem_count = total - len(neigh[start_idx])
    
    while rem_count > 0:
        best_c = -1
        best_gain = 0
        
        for c in range(total):
            if c in code: # Optimization: use a set for O(1) lookup
                continue
            
            gain = 0
            for u in neigh[c]:
                if not covered[u]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1 or best_gain == 0:
            # Should not happen if radius >= 0 and code is valid, 
            # but if we ran out of options, break
            break
            
        code.append(best_c)
        for u in neigh[best_c]:
            if not covered[u]:
                covered[u] = True
                rem_count -= 1
                
    return code

def _local_search(code, neigh, total, deadline, rng):
    """
    Local search to remove redundant codewords.
    """
    code_set = set(code)
    
    # Count how many times each word is covered
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    changed = True
    while changed:
        if time.monotonic() > deadline:
            break
        changed = False
        # Iterate in reverse to allow popping
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if c is redundant
            redundant = True
            for u in neigh[c]:
                if counts[u] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                code.pop(i)
                code_set.discard(c)
                for u in neigh[c]:
                    counts[u] -= 1
                changed = True
                
    return code, counts

def _kick_and_repair(code, neigh, total, rng, kick_size=2):
    """
    Remove random codewords and repair the cover.
    """
    code = code[:]
    code_set = set(code)
    
    # 1. Compute counts
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    # 2. Kick: Remove random codewords
    # Ensure we don't remove too many at once if code is small
    actual_kick = min(kick_size, len(code) - 1) 
    if actual_kick <= 0:
        return code
        
    removed = []
    while actual_kick > 0 and len(code) > 0:
        idx = rng.randint(0, len(code) - 1)
        c = code.pop(idx)
        code_set.discard(c)
        removed.append(c)
        for u in neigh[c]:
            counts[u] -= 1
        actual_kick -= 1
        
    # 3. Identify holes
    holes = []
    for u in range(total):
        if counts[u] == 0:
            holes.append(u)
            
    # 4. Repair: Greedily add words to cover holes
    while holes:
        if not holes:
            break
            
        # Find the word that covers the most holes
        best_c = -1
        best_gain = 0
        
        # Optimization: Only consider neighbors of a random hole?
        # Or consider all words?
        # Considering all words is O(729 * deg). deg is 1 + 6*2 = 13.
        # 729 * 13 is small.
        
        # To speed up, we can just pick a random hole and look at its neighbors
        # but global greedy is better for minimizing size.
        
        # Let's use global greedy for repair
        for c in range(total):
            if c in code_set:
                continue
                
            gain = 0
            for u in neigh[c]:
                if counts[u] == 0:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1 or best_gain == 0:
            # Cannot repair? Should not happen if radius >= 1
            break
            
        code.append(best_c)
        code_set.add(best_c)
        for u in neigh[best_c]:
            counts[u] += 1
            
        # Update holes list efficiently?
        # Just rebuild it or filter it
        holes = [u for u in holes if counts[u] == 0]
        
    return code

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    rng = random.Random(42)
    
    # 1. Precompute
    neigh, total = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Init
    code = _greedy_init(q, n, radius, neigh, total, rng)
    
    # 3. Initial Local Search
    code, counts = _local_search(code, neigh, total, deadline, rng)
    
    best_code = code[:]
    best_size = len(code)
    
    # 4. Iterated Local Search (Kick and Repair)
    # We vary the kick size to explore different neighborhoods
    kick_sizes = [1, 2, 3, 4, 5]
    
    while time.monotonic() < deadline - 5.0:
        # Choose a kick size
        ks = rng.choice(kick_sizes)
        
        # Kick and Repair
        current_code = _kick_and_repair(best_code, neigh, total, rng, kick_size=ks)
        
        # Local Search to minimize
        current_code, _ = _local_search(current_code, neigh, total, deadline, rng)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
            # If we hit the theoretical lower bound or a very good number, 
            # we can increase kick size to try to break further, 
            # or just keep going.
            # Lower bound is 71. Best known 73.
            if best_size <= 71:
                break

    # 5. Final Conversion to Words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
