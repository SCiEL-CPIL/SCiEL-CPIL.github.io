import numpy as np

def construct(q, n, radius):
    # This implementation targets the specific case q=3, n=8, radius=1 for efficiency,
    # but falls back to a generic greedy for other parameters if needed (though the task is fixed).
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy for general case
        # This is likely not needed for the specific task but ensures safety.
        return _generic_greedy(q, n, radius)
    
    N = q ** n
    # Precompute neighbors for all words in {0,1,2}^8
    # neighbors[i] is a list of indices j such that d(i, j) <= 1
    
    # Convert integer to base-3 tuple
    def to_base3(x, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = x % 3
            x //= 3
        return digits

    # Convert base-3 tuple to integer
    def from_base3(digits):
        val = 0
        for d in digits:
            val = val * 3 + d
        return val

    neighbors = [[] for _ in range(N)]
    for i in range(N):
        digits = to_base3(i, n)
        # The word itself
        neighbors[i].append(i)
        # Neighbors
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                digits[pos] = new_val
                neighbors[i].append(from_base3(digits))
            digits[pos] = orig # restore
            
    # Greedy algorithm
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        # Pick the first uncovered word
        u = next(iter(uncovered))
        
        # Find the best center in the sphere of u
        best_center = u
        best_coverage = -1
        
        for c in neighbors[u]:
            # Count how many neighbors of c are in uncovered
            # We can do this by iterating neighbors[c] and checking membership in uncovered
            # Since uncovered is a set, lookup is O(1)
            count = 0
            for w in neighbors[c]:
                if w in uncovered:
                    count += 1
            if count > best_coverage:
                best_coverage = count
                best_center = c
                
        # Add best_center to code
        code.append(best_center)
        
        # Remove all words covered by best_center from uncovered
        for w in neighbors[best_center]:
            if w in uncovered:
                uncovered.remove(w)
                
    # Convert indices back to lists of digits
    result = []
    for idx in code:
        result.append(to_base3(idx, n))
        
    return result

def _generic_greedy(q, n, radius):
    # Simple fallback
    N = q ** n
    all_words = [list(map(int, np.base_repr(i, base=q).zfill(n))) for i in range(N)]
    uncovered = set(range(N))
    code = []
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
        
    while uncovered:
        u_idx = next(iter(uncovered))
        u = all_words[u_idx]
        # Find best center in sphere of u
        # For generic radius, generating sphere is harder. 
        # For radius 1, it's simple. 
        # Since the task is fixed to q=3, n=8, r=1, this fallback is just a safety net.
        # We'll just pick u as center for simplicity in fallback.
        c = u
        code.append(c)
        # Remove covered
        to_remove = []
        for w_idx in uncovered:
            if hamming_dist(all_words[w_idx], c) <= radius:
                to_remove.append(w_idx)
        for w_idx in to_remove:
            uncovered.remove(w_idx)
            
    return code
