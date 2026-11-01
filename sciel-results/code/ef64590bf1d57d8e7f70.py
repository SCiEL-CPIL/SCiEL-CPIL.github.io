import time
import random
import itertools
import sys

def _standard_superperm(n):
    """Standard recursive superpermutation construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    word = _standard_superperm(n - 1)
    # Extract all (n-1)-permutations from word
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        # Find max overlap between end of new_word and start of block
        max_ov = 0
        for ov in range(min(len(new_word), len(block)), -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if new_word[-ov:] == block[:ov]:
                max_ov = ov
                break
        new_word.extend(block[max_ov:])
    return new_word

def _find_superperm_5(time_limit=15):
    """Find a short superpermutation for n=5 using greedy random restarts."""
    n = 5
    symbols = list(range(1, n + 1))
    perms = [tuple(p) for p in itertools.permutations(symbols)]
    perm_set = set(perms)
    best_len = float('inf')
    best_word = None
    start_time = time.time()
    
    def greedy_start(start_perm):
        word = list(start_perm)
        seen = set()
        # Add perms in current word
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i+n]))
        remaining = perm_set - seen
        if not remaining:
            return word
            
        while remaining:
            # Find the permutation with maximum overlap
            best_overlap = -1
            best_next = None
            # For speed, iterate over a random subset if remaining is large
            if len(remaining) > 50:
                candidates = random.sample(list(remaining), 50)
            else:
                candidates = list(remaining)
                
            for p in candidates:
                p_list = list(p)
                max_ov = 0
                # Check overlaps from largest to smallest
                for ov in range(min(len(word), len(p_list)), 0, -1):
                    if word[-ov:] == p_list[:ov]:
                        max_ov = ov
                        break
                if max_ov > best_overlap:
                    best_overlap = max_ov
                    best_next = p
            if best_next is None:
                best_next = random.choice(list(remaining))
                best_overlap = 0
                
            # Append best_next with overlap
            word.extend(best_next[best_overlap:])
            remaining.remove(best_next)
            # Update seen
            # Only need to check the new windows
            start_idx = len(word) - n
            for i in range(start_idx, len(word) - n + 1):
                if i >= 0:
                    seen.add(tuple(word[i:i+n]))
            # Remove newly seen from remaining
            for s in list(remaining):
                if s in seen:
                    remaining.remove(s)
                    
        return word

    # Try a few deterministic starts first?
    # Or just random restarts.
    while time.time() - start_time < time_limit:
        start_perm = tuple(random.sample(symbols, n))
        word = greedy_start(start_perm)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
            if best_len == 152: # Known optimal
                break
                
    if best_word is None:
        # Fallback to standard
        return _standard_superperm(5)
        
    return best_word

def _extend_superperm(word, m):
    """Extend a superpermutation of 1..m-1 to 1..m."""
    n = m
    # Extract all (n-1)-permutations from word
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        # Find max overlap between end of new_word and start of block
        max_ov = 0
        for ov in range(min(len(new_word), len(block)), -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if new_word[-ov:] == block[:ov]:
                max_ov = ov
                break
        new_word.extend(block[max_ov:])
    return new_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if n <= 4:
        return "".join(map(str, _standard_superperm(n)))
        
    if n == 5:
        word = _find_superperm_5(10)
        return "".join(map(str, word))
        
    if n == 6 or n == 7:
        # Use best S5 as base
        s5 = _find_superperm_5(10)
        word = list(s5)
        
        # Recursive extension
        for m in range(6, n + 1):
            if time.time() > deadline - 5:
                break
            word = _extend_superperm(word, m)
        
        return "".join(map(str, word))
        
    # Fallback for n > 7
    word = _standard_superperm(n)
    return "".join(map(str, word))

if __name__ == "__main__":
    # Quick test
    print(construct(3))
    print(len(construct(3)))
