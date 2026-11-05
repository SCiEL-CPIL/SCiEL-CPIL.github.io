import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132412132413214321"
    
    # Standard recursive construction for n=7
    # This yields length 5913
    word = [1]
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations from current word in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find overlap with end of new_word
            max_overlap = 0
            # Max possible overlap is m-1
            for o in range(min(len(new_word), len(block)), -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word
        
    return "".join(map(str, word))
