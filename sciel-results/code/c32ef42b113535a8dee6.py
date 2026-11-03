import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # 1. Generate initial superpermutation using the classic recursive method
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # Convert to list of ints for easier manipulation
    current_word = list(word)
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
            if len(seen) == len(target_perms):
                return True
        return False

    # 2. Local Search: Greedy Deletion
    # Try to remove characters one by one if the result is still valid.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(current_word) and time.time() < deadline:
            # Try removing current_word[i]
            new_word = current_word[:i] + current_word[i+1:]
            if is_valid(new_word):
                current_word = new_word
                improved = True
                # Don't increment i, check the same position again (which is now the next char)
                # Actually, since we removed i, the next char is now at i.
                # So we just continue with the same i.
            else:
                i += 1
                
    # If time remains, we could try 2-deletions or random restarts, but single deletion is a good start.
    # For n=7, the seed is 5913. The best is 5905.
    # Greedy deletion usually gets to a local minimum.
    
    return "".join(map(str, current_word))
