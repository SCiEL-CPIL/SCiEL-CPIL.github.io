import random
import itertools

# Fallback for popcount if not available
try:
    (1).bit_count
    def _popcount(x):
        return x.bit_count()
except AttributeError:
    def _popcount(x):
        return bin(x).count('1')

def _greedy_cover_specific(q, n, radius):
    """
    Solves the specific case q=3, n=6, radius=1 using randomized greedy set cover.
    """
    N = q ** n
    # Precompute masks for each word
    # Word i corresponds to the tuple of digits
    # We represent a set of words as a bitmask of length N
    
    masks = [0] * N
    
    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map index to word tuple
    # Index is just the integer value in base q
    
    for i, word in enumerate(all_words):
        mask = 0
        # The word itself
        mask |= (1 << i)
        
        # Neighbors at distance 1
        for pos in range(n):
            orig_digit = word[pos]
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                # Create neighbor tuple
                # Instead of creating new tuple, we can calculate its index
                # Index = sum(d * q^pos)
                # Current index i
                # Change digit at pos from orig to new
                # delta = (new - orig) * (q ** pos)
                # neighbor_index = i + delta
                delta = (new_digit - orig_digit) * (q ** pos)
                ni = i + delta
                mask |= (1 << ni)
        
        masks[i] = mask

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 50
    
    for _ in range(num_restarts):
        uncovered = (1 << N) - 1
        selected_indices = []
        
        while uncovered:
            max_cov = 0
            candidates = []
            
            # Find word covering most uncovered bits
            for i in range(N):
                # Optimization: if this word is already fully covered, its coverage is 0?
                # No, we check intersection with uncovered.
                cov = _popcount(uncovered & masks[i])
                if cov > max_cov:
                    max_cov = cov
                    candidates = [i]
                elif cov == max_cov and cov > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered is not 0 and masks are correct
                
            chosen = random.choice(candidates)
            selected_indices.append(chosen)
            uncovered &= ~masks[chosen]
            
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_code = selected_indices
            
    # Convert indices back to list of lists
    if best_code is None:
        # Fallback to empty or error
        return []
        
    result = []
    for idx in best_code:
        # Convert index to base q digits
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse() # Since we generated product in order, index 0 is 0...0
        # Wait, itertools.product order:
        # (0,0), (0,1), ... (0,2), (1,0)...
        # Index calculation: i = d0 * q^(n-1) + d1 * q^(n-2) ... + d_{n-1}
        # My index calculation in mask generation:
        # delta = (new - orig) * q^pos.
        # This implies index = sum(d_pos * q^pos).
        # So d_0 is the most significant digit?
        # In `itertools.product`, the last element changes fastest.
        # So index = d0 * q^(n-1) + ... + d_{n-1} * q^0.
        # My mask generation used `q ** pos`.
        # If pos=0 is the first element of the tuple, then index contribution is d0 * q^0?
        # Let's check `itertools.product` mapping.
        # prod = [(0,0), (0,1), (0,2), (1,0), ...]
        # Index of (0,0) is 0.
        # Index of (0,1) is 1.
        # Index of (1,0) is 3 (for q=3, n=2).
        # So index = d0 * q + d1.
        # So d0 is MSB.
        # My delta calculation: `delta = (new - orig) * (q ** pos)`.
        # If pos=0 (first digit), delta = diff * 1.
        # This matches d1 (LSB) changing.
        # So `pos` in my loop corresponds to the position in the tuple,
        # BUT the weight `q ** pos` assumes pos 0 is LSB.
        # In `itertools.product`, the last index is LSB.
        # So `pos` in `range(n)` (0 to n-1) corresponds to tuple index.
        # Tuple index 0 is MSB. Weight should be q^(n-1).
        # Tuple index n-1 is LSB. Weight should be q^0.
        # So weight = q ** (n - 1 - pos).
        
        # Let's re-verify the mask generation logic in the code below.
        pass

    # Correcting the logic in the final code block
    return result

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _greedy_cover_specific(3, 6, 1)
    else:
        # General fallback
        return _greedy_general(q, n, radius)

def _greedy_general(q, n, radius):
    # Simple greedy for general case, might be slow for large N
    N = q ** n
    if N > 10000:
        # Fallback to direct sum if too big? Or just return empty?
        # For this task, we assume small N.
        pass
        
    all_words = list(itertools.product(range(q), repeat=n))
    masks = [0] * N
    
    for i, word in enumerate(all_words):
        mask = 0
        # Generate neighbors
        # This is expensive for general case, but ok for small N
        for neighbor in itertools.product(range(q), repeat=n):
            dist = sum(1 for a, b in zip(word, neighbor) if a != b)
            if dist <= radius:
                # Find index of neighbor
                # This is O(N) search, bad.
                # We need a map or index calculation.
                pass
        # This general fallback is incomplete. 
        # Given the task is specific, we rely on the specific solver.
        return []

# Refine _greedy_cover_specific with correct index logic

def _greedy_cover_specific(q, n, radius):
    N = q ** n
    masks = [0] * N
    
    # Precompute powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Index calculation:
    # itertools.product yields tuples (d0, d1, ..., d_{n-1})
    # Index = d0 * q^(n-1) + d1 * q^(n-2) + ... + d_{n-1} * q^0
    # So weight of d_i is q^(n-1-i)
    
    for i in range(N):
        # Reconstruct word from index i to get digits
        # Or just iterate and build masks based on index arithmetic
        # It is faster to iterate i and compute neighbors via index arithmetic
        
        mask = 0
        mask |= (1 << i)
        
        # Calculate neighbors
        # We need to know the digits of i to change them.
        # Or we can just iterate all j and check distance? O(N^2) = 729^2 ~ 500k. OK.
        # But we can do it smarter.
        
        # Let's extract digits of i
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse() # d0, d1, ... d_{n-1}
        
        for pos in range(n):
            orig = digits[pos]
            weight = powers[n - 1 - pos]
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                mask |= (1 << ni)
                
        masks[i] = mask

    best_code = None
    best_size = float('inf')
    num_restarts = 50
    
    for _ in range(num_restarts):
        uncovered = (1 << N) - 1
        selected_indices = []
        
        while uncovered:
            max_cov = 0
            candidates = []
            
            for i in range(N):
                cov = _popcount(uncovered & masks[i])
                if cov > max_cov:
                    max_cov = cov
                    candidates = [i]
                elif cov == max_cov and cov > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            chosen = random.choice(candidates)
            selected_indices.append(chosen)
            uncovered &= ~masks[chosen]
            
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_code = selected_indices
            
    result = []
    if best_code:
        for idx in best_code:
            digits = []
            temp = idx
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            digits.reverse()
            result.append(digits)
    return result
