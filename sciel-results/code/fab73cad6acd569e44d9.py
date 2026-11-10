import time
import itertools
from collections import defaultdict

def _generate_standard_superperm(n):
    """Generates the standard superpermutation of length sum(k! for k in 1..n)."""
    if n == 1:
        return [1]
    
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of size m-1 in order of appearance
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation (all distinct)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is P, m, P
            block = list(p) + [m] + list(p)
            
            # Find max overlap with end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        word = new_word
    
    return word

def _optimize_superperm(word, n, time_limit=110.0):
    """
    Optimizes the superpermutation by removing redundant characters.
    word: list of ints
    n: number of symbols
    """
    start_time = time.time()
    target_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(target_perms)
    
    # Initialize counts
    perm_counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if w in perm_set:
            perm_counts[w] += 1
            
    found_count = len(perm_counts)
    target_count = len(perm_set)
    
    # If we don't have all perms, something is wrong with input, but assume valid
    if found_count < target_count:
        return word

    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        i = 0
        L = len(word)
        while i < L:
            if time.time() - start_time > time_limit:
                break
            
            # Try removing word[i]
            # Windows affected are those starting at i-n+1 to i
            affected_windows = []
            for j in range(max(0, i - n + 1), min(i, L - n) + 1):
                w = tuple(word[j:j + n])
                if w in perm_set:
                    affected_windows.append((j, w))
            
            # Tentatively remove
            for j, w in affected_windows:
                perm_counts[w] -= 1
                if perm_counts[w] == 0:
                    found_count -= 1
            
            if found_count < target_count:
                # Restore
                for j, w in affected_windows:
                    perm_counts[w] += 1
                    if perm_counts[w] == 1:
                        found_count += 1
                i += 1
            else:
                # Accept removal
                word.pop(i)
                improved = True
                # Do not increment i, check new element at i
                L = len(word)
        
    return word

def construct(n):
    start_time = time.time()
    
    if n == 1:
        return "1"
        
    if n <= 6:
        # Generate standard, then optimize
        word = _generate_standard_superperm(n)
        word = _optimize_superperm(word, n, time_limit=110.0)
        return "".join(map(str, word))
        
    elif n == 7:
        # Step 1: Get a good S_6
        # Generate standard S_6
        word_6 = _generate_standard_superperm(6)
        # Optimize S_6
        word_6 = _optimize_superperm(word_6, 6, time_limit=10.0)
        
        # Step 2: Extend S_6 to S_7
        # We reuse the logic from _generate_standard_superperm but starting from word_6
        m = 7
        perms, seen = [], set()
        for i in range(len(word_6) - (m - 1) + 1):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        word_7 = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(word_7), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or word_7[len(word_7) - o:] == block[:o]:
                    overlap = o
                    break
            word_7 += block[overlap:]
            
        # Step 3: Optimize S_7
        word_7 = _optimize_superperm(word_7, 7, time_limit=110.0)
        
        return "".join(map(str, word_7))
        
    else:
        # Fallback for n > 7
        word = _generate_standard_superperm(n)
        word = _optimize_superperm(word, n, time_limit=110.0)
        return "".join(map(str, word))
