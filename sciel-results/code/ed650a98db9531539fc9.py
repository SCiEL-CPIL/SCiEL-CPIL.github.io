import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 1:
        return "".join(map(str, range(1, n+1)))
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    # For n=5,6,7 use the classic construction with optimization
    # The classic construction builds word level by level
    # At level m, we have a word containing all permutations of 1..m-1 as substrings of length m-1
    # We need to add symbol m to create all permutations of 1..m
    
    def classic_construct(n):
        """Classic superpermutation construction."""
        word = [1]
        for m in range(2, n + 1):
            # Find all distinct (m-1)-length substrings that are permutations of 1..m-1
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # For each permutation p of 1..m-1, we create a block: p + [m] + p
            # We need to find maximum overlap between consecutive blocks
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap with current new
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Get the classic construction
    word = classic_construct(n)
    best_word = word[:]
    best_len = len(word)
    
    # Verify the classic construction
    def verify(word, n):
        found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                found.add(window)
        return len(found) == len(list(permutations(range(1, n+1))))
    
    if not verify(best_word, n):
        # Fallback - should not happen
        return "".join(map(str, best_word))
    
    # Try to improve with local search on the ordering of permutations at each level
    # Key insight: the ordering of permutations at each level affects the total length
    # We can try different orderings to maximize overlaps
    
    def try_ordering(n, orderings, time_limit):
        """Try given orderings for each level and return best word."""
        start = time.time()
        best = None
        best_len = float('inf')
        
        for ordering in orderings:
            if time.time() - start > time_limit:
                break
            
            word = [1]
            for m in range(2, n + 1):
                # Get the permutations of 1..m-1 in the given order
                perms = ordering.get(m, [])
                
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    overlap = 0
                    max_ov = min(len(new), len(block))
                    for o in range(max_ov, -1, -1):
                        if o == 0 or new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                word = new
            
            if verify(word, n) and len(word) < best_len:
                best_len = len(word)
                best = word[:]
        
        return best, best_len
    
    # Generate several orderings to try
    # Ordering 1: natural order (from classic construction)
    # Ordering 2: reverse order
    # Ordering 3: lexicographic
    # Ordering 4: Gray code order (adjacent permutations differ by one swap)
    
    def get_gray_code_order(m):
        """Get permutations in Gray code order."""
        perms = list(permutations(range(1, m+1)))
        # Convert to Gray code ordering
        result = [perms[0]]
        for i in range(1, len(perms)):
            # Find the permutation that differs by one adjacent swap from result[-1]
            # This is a simplification; true Gray code is more complex
            # For now, just use lexicographic
            result.append(perms[i])
        return result
    
    # For now, use simple orderings
    def get_orderings(n):
        orderings = []
        
        # Natural order
        natural = {}
        for m in range(2, n+1):
            natural[m] = list(permutations(range(1, m)))
        orderings.append(natural)
        
        # Reverse order
        reverse = {}
        for m in range(2, n+1):
            reverse[m] = list(reversed(list(permutations(range(1, m)))))
        orderings.append(reverse)
        
        return orderings
    
    # Try different orderings
    orderings = get_orderings(n)
    best, best_len = try_ordering(n, orderings, 30)
    
    if best is not None and best_len < len(best_word):
        best_word = best
        best_len = best_len
    
    # Final verification
    if not verify(best_word, n):
        # Use the classic construction which we know is valid
        best_word = classic_construct(n)
    
    return "".join(map(str, best_word))
