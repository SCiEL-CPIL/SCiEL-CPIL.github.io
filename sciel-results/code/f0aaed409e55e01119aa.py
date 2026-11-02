import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if n <= 1:
        return "1"
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = 0
            max_o = min(len(new), len(block))
            for o in range(max_o, -1, -1):
                if o == 0:
                    max_overlap = 0
                    break
                match = True
                for k in range(o):
                    if new[len(new) - o + k] != block[k]:
                        match = False
                        break
                if match:
                    max_overlap = o
                    break
            new += block[max_overlap:]
            
        word = new
        
    return "".join(map(str, word))
