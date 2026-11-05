import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: Simple Greedy (slow but correct)
        # Or return a trivial valid code (e.g. all 0s doesn't work for R=1)
        # For R=1, the set of all words with first element 0 is valid for any n>=1, q>=2.
        # Size q^(n-1).
        code_ints = []
        for i in range(q ** (n - 1)):
            # Convert i to base q, length n-1, prepend 0
            digits = [0] * n
            tmp = i
            for j in range(n - 1):
                digits[j + 1] = tmp % q
                tmp //= q
            code_ints.append(tuple(digits))
        return [list(c) for c in code_ints]

    total_words = 3 ** n
    neighbors = [[] for _ in range(total_words)]
    powers = [3**i for i in range(n)]
    
    for w in range(total_words):
        digits = [0]*n
        tmp = w
        for i in range(n):
            digits[i] = tmp % 3
            tmp //= 3
        neighbors[w].append(w)
        for i in range(n):
            d = digits[i]
            p = powers[i]
            new_d = (d + 1) % 3
            new_w = w + (new_d - d) * p
            neighbors[w].append(new_w)
            new_d = (d - 1) % 3
            new_w = w + (new_d - d) * p
            neighbors[w].append(new_w)
            
    # Initial code: All words with first digit 0
    # In integer representation, first digit is the most significant?
    # My integer mapping: w = d_0 * 3^0 + d_1 * 3^1 + ...
    # So d_0 is the least significant digit.
    # "First digit" in the problem usually implies index 0.
    # If I want index 0 to be 0, then w % 3 == 0.
    initial_code = [w for w in range(total_words) if w % 3 == 0]
    
    cover_count = [0] * total_words
    
    # Initialize cover counts
    for c in initial_code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    code_set = set(initial_code)
    code_list = initial_code
    
    # Pruning Loop
    # We try to remove words.
    # To make it faster, we can iterate multiple times until no change.
    # To avoid O(N^2) in the worst case, we can just iterate through the current list.
    
    # Optimization: If we remove a word, we update cover_count.
    # We should check words that are "easy" to remove?
    # A word is easy to remove if its neighbors have high cover counts.
    # But calculating this for all words every step is expensive.
    # Simple sequential scan is fine for 2000 words.
    
    improved = True
    while improved:
        improved = False
        # Iterate over a copy of the list to allow modification
        # Or iterate by index and delete
        # Deleting from list is O(N). N is small (<2500).
        
        # Randomize order to avoid bias?
        # random.shuffle(code_list)
        
        i = 0
        while i < len(code_list):
            c = code_list[i]
            # Check if c can be removed
            can_remove = True
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                code_list.pop(i)
                code_set.remove(c)
                improved = True
                # Do not increment i, check next element
            else:
                i += 1
                
    # Convert back to list of lists
    result = []
    for c in code_list:
        digits = [0]*n
        tmp = c
        for i in range(n):
            digits[i] = tmp % 3
            tmp //= 3
        result.append(digits)
        
    return result
