import itertools
import time

def standard_construct(n):
    """
    Generates the standard superpermutation of length sum(k! for k in 1..n).
    Returns a list of integers.
    """
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations from word in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation of 1..m-1
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Construct block: P m P
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    # Handle small n where standard is optimal
    if n <= 6:
        return "".join(map(str, standard_construct(n)))
    
    # For n > 7, local search is too slow, return standard
    if n > 7:
        return "".join(map(str, standard_construct(n)))
    
    # For n == 7, use optimized local search
    word = standard_construct(n)
    L = len(word)
    
    # Pre-compute permutation encoding for fast window hashing
    # Use base 8 encoding
    base = 8
    
    def encode_window_fast(w, start, n):
        """Encode window w[start:start+n] as integer using base 8."""
        val = 0
        for k in range(n):
            val = val * base + w[start + k]
        return val
    
    # Build initial set of all valid permutation codes
    valid_codes = set()
    for p in itertools.permutations(range(1, n + 1)):
        val = 0
        for x in p:
            val = val * base + x
        valid_codes.add(val)
    
    # Count how many times each valid permutation appears in current word
    # This allows O(1) validity check after deletion
    perm_count = {}
    for i in range(L - n + 1):
        code = encode_window_fast(word, i, n)
        if code in valid_codes:
            perm_count[code] = perm_count.get(code, 0) + 1
    
    # Number of valid permutations that appear at least once
    num_present = sum(1 for c in perm_count.values() if c > 0)
    total_perms = len(valid_codes)
    
    def is_still_valid():
        return num_present == total_perms
    
    # Deadline guard
    start_time = time.time()
    deadline = 115  # 5s margin before 120s limit
    
    # Greedy deletion with efficient validity tracking
    # When we delete position i, windows affected are those that include position i
    # Window j includes position i iff j <= i <= j + n - 1
    # So j ranges from max(0, i - n + 1) to min(L - n, i)
    
    while time.time() - start_time < deadline:
        changed = False
        i = 0
        while i < len(word):
            L = len(word)
            if L < n:
                break
            
            # Try deleting word[i]
            # Windows that will be affected: j in [max(0, i-n+1), min(L-n, i)]
            j_start = max(0, i - n + 1)
            j_end = min(L - n, i)
            
            # Check if deletion would make any valid permutation disappear
            safe = True
            for j in range(j_start, j_end + 1):
                code = encode_window_fast(word, j, n)
                if code in valid_codes and perm_count.get(code, 0) == 1:
                    safe = False
                    break
            
            if safe:
                # Perform deletion
                # Update perm_count for affected windows
                for j in range(j_start, j_end + 1):
                    code = encode_window_fast(word, j, n)
                    if code in valid_codes:
                        perm_count[code] -= 1
                        if perm_count[code] == 0:
                            num_present -= 1
                
                word = word[:i] + word[i + 1:]
                changed = True
                # Do not increment i, check the new element at position i
            else:
                i += 1
        
        if not changed:
            break
    
    # Try a few swap moves to see if we can improve further
    if time.time() - start_time < deadline:
        # Rebuild perm_count for current word
        L = len(word)
        perm_count = {}
        for i in range(L - n + 1):
            code = encode_window_fast(word, i, n)
            if code in valid_codes:
                perm_count[code] = perm_count.get(code, 0) + 1
        num_present = sum(1 for c in perm_count.values() if c > 0)
        
        # Try adjacent swaps
        for _ in range(10):  # Limited number of swap passes
            if time.time() - start_time > deadline:
                break
            improved = False
            for i in range(len(word) - 1):
                if time.time() - start_time > deadline:
                    break
                # Try swapping word[i] and word[i+1]
                # Windows affected: j in [max(0, i-n+1), min(L-n, i+1)]
                L = len(word)
                j_start = max(0, i - n + 1)
                j_end = min(L - n, i + 1)
                
                # Check if swap is safe
                safe = True
                for j in range(j_start, j_end + 1):
                    # Get code before swap
                    code_before = encode_window_fast(word, j, n)
                    # Get code after swap (simulate)
                    # Window j after swap: positions j..j+n-1, but i and i+1 are swapped
                    code_after = 0
                    for k in range(n):
                        pos = j + k
                        if pos == i:
                            val = word[i + 1]
                        elif pos == i + 1:
                            val = word[i]
                        else:
                            val = word[pos]
                        code_after = code_after * base + val
                    
                    if code_before in valid_codes and perm_count.get(code_before, 0) == 1:
                        # This permutation would disappear
                        safe = False
                        break
                    if code_after in valid_codes and code_after != code_before:
                        # New permutation appears, that's fine
                        pass
                
                if safe:
                    # Perform swap
                    for j in range(j_start, j_end + 1):
                        code_before = encode_window_fast(word, j, n)
                        if code_before in valid_codes:
                            perm_count[code_before] -= 1
                            if perm_count[code_before] == 0:
                                num_present -= 1
                    
                    word[i], word[i + 1] = word[i + 1], word[i]
                    
                    for j in range(j_start, j_end + 1):
                        code_after = encode_window_fast(word, j, n)
                        if code_after in valid_codes:
                            if code_after not in perm_count:
                                num_present += 1
                            perm_count[code_after] = perm_count.get(code_after, 0) + 1
                    
                    improved = True
            
            if not improved:
                break
    
    return "".join(map(str, word))
