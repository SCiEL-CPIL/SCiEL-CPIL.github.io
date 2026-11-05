import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        perms, seen = [], set()
        # Extract unique (m-1)-length windows with distinct symbols
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_check = min(len(new), len(block))
            overlap = 0
            for o in range(max_check, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    return "".join(map(str, word))
