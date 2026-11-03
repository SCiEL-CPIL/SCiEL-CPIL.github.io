import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Precompute all words as integers for fast distance computation
    # Each word is an integer in [0, q^n)
    total = q ** n
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
    
    def to_word(val):
        word = []
        for i in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        return word
    
    # Precompute neighbors: for each word, the set of words at distance <= radius
    # For radius 1: the word itself + all words differing in exactly 1 position
    # We'll use integer representation for speed
    
    # Build neighbor list for all words
    # neighbors[i] = list of words at distance <= 1 from word i
    # This is expensive for large spaces, but 729 is small
    neighbor_cache = {}
    def get_neighbors(val):
        if val in neighbor_cache:
            return neighbor_cache[val]
        word = to_word(val)
        nbrs = [val]
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                word[pos] = new_val
                nbrs.append(to_int(word))
            word[pos] = orig
        neighbor_cache[val] = nbrs
        return nbrs
    
    # Greedy construction
    def greedy_construct():
        covered = set()
        codewords = []
        # Start with all words uncovered
        uncovered = set(range(total))
        
        while uncovered:
            # Find the word that covers the most uncovered words
            best_word = None
            best_count = -1
            # Sample if too many to check all
            candidates = list(uncovered) if len(uncovered) <= 200 else random.sample(uncovered, 200)
            for cand in candidates:
                nbrs = get_neighbors(cand)
                count = sum(1 for nb in nbrs if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_word = cand
            
            if best_word is None:
                break
            
            codewords.append(best_word)
            nbrs = get_neighbors(best_word)
            for nb in nbrs:
                uncovered.discard(nb)
        
        return codewords
    
    # Check if a set of codewords covers everything
    def is_covering(codewords):
        covered = set()
        for cw in codewords:
            covered.update(get_neighbors(cw))
        return len(covered) == total
    
    # Try to remove a codeword if its coverage is redundant
    def try_remove(codewords):
        for i in range(len(codewords)):
            remaining = codewords[:i] + codewords[i+1:]
            if is_covering(remaining):
                return remaining
        return codewords
    
    # Local search: try swapping or replacing codewords
    def local_search(codewords, time_limit):
        codewords = list(codewords)
        best = codewords
        best_len = len(codewords)
        
        while time.time() < time_limit:
            improved = False
            
            # Try to remove redundant codewords
            new_codewords = try_remove(codewords)
            if len(new_codewords) < len(codewords):
                codewords = new_codewords
                improved = True
                if len(codewords) < best_len:
                    best = codewords[:]
                    best_len = len(codewords)
            
            if len(codewords) == best_len and time.time() > time_limit - 5:
                break
            
            # Try to find a codeword to replace with a better one
            if time.time() > time_limit - 10:
                break
                
            for i in range(len(codewords)):
                if time.time() > time_limit - 10:
                    break
                # Try replacing codeword i with a random word
                # Check if the new set still covers everything
                # Sample some candidates
                for _ in range(5):
                    new_cw = random.randint(0, total - 1)
                    if new_cw == codewords[i]:
                        continue
                    candidate = codewords[:i] + [new_cw] + codewords[i+1:]
                    if is_covering(candidate):
                        codewords = candidate
                        improved = True
                        break
            
            # Try adding a codeword that allows removing two others (complex, skip for now)
            
            if not improved and time.time() > time_limit - 20:
                # Try a different strategy: greedy rebuild from scratch
                new_cw = greedy_construct()
                if len(new_cw) < len(codewords):
                    codewords = new_cw
                    if len(codewords) < best_len:
                        best = codewords[:]
                        best_len = len(codewords)
                    improved = True
            
            if not improved:
                # Perturb: randomly remove one and rebuild
                if len(codewords) > 1:
                    idx = random.randint(0, len(codewords) - 1)
                    perturbed = codewords[:idx] + codewords[idx+1:]
                    # Greedily add back to cover
                    uncovered = set(range(total))
                    for cw in perturbed:
                        uncovered.difference_update(get_neighbors(cw))
                    while uncovered and time.time() < time_limit - 5:
                        candidates = list(uncovered) if len(uncovered) <= 100 else random.sample(uncovered, 100)
                        best_word = None
                        best_count = -1
                        for cand in candidates:
                            nbrs = get_neighbors(cand)
                            count = sum(1 for nb in nbrs if nb in uncovered)
                            if count > best_count:
                                best_count = count
                                best_word = cand
                        if best_word is None:
                            break
                        perturbed.append(best_word)
                        uncovered.difference_update(get_neighbors(best_word))
                    if len(perturbed) < len(codewords):
                        codewords = perturbed
                        if len(codewords) < best_len:
                            best = codewords[:]
                            best_len = len(codewords)
            
            if time.time() > time_limit - 5:
                break
        
        return best if len(best) <= len(codewords) else codewords
    
    # Initial construction
    random.seed(42)
    initial = greedy_construct()
    
    # Also try the product code as another starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    product_code = []
    for h in tetracode:
        for tail in range(3 ** (n - 4)):
            word = h[:]
            t = tail
            for i in range(n - 4, -1, -1):
                word.append(t % 3)
                t //= 3
            product_code.append(to_int(word))
    
    # Pick the better initial solution
    if len(initial) <= len(product_code):
        current = initial
    else:
        current = product_code
    
    # Run local search with time budget
    time_left = deadline - time.time()
    if time_left > 5:
        current = local_search(current, deadline - 5)
    
    # Final cleanup: remove any redundant codewords
    current = try_remove(current)
    
    # Verify coverage
    if not is_covering(current):
        # Fallback to a known valid solution
        current = product_code
        current = try_remove(current)
    
    # Convert back to word format
    return [to_word(cw) for cw in current]
