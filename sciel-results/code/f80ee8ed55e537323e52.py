import time
from itertools import permutations

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
    current = word[:]
    
    # Precompute initial windows
    seen = set()
    for i in range(len(current) - n + 1):
        seen.add(tuple(current[i:i + n]))
    
    # Verify classic construction is valid
    if not all_perms.issubset(seen):
        return "".join(map(str, current))

    # Fast local search: try deleting each character
    # We iterate multiple times until no more deletions are possible or time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try deleting from left to right
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            # Remove current[i]
            saved = current.pop(i)
            
            # Update seen set
            # Windows affected are those overlapping index i
            # Old windows: [i-n+1, i]
            # New windows: [i-n+1, i] (shifted)
            
            # To be safe and simple, we can just re-calculate validity for the affected region
            # But O(n) update is better.
            
            # Let's just check if the new word is valid by checking only the changed windows
            # However, maintaining the set incrementally is tricky with pops.
            # Given L=5900, n=7, a full check is ~5900 ops.
            # 5900 deletions * 5900 check = 35M ops. In Python, this might be ~10-20s.
            # So a full check per deletion is feasible within 140s.
            
            is_valid = True
            new_seen = set()
            for j in range(len(current) - n + 1):
                new_seen.add(tuple(current[j:j + n]))
            
            if all_perms.issubset(new_seen):
                improved = True
                seen = new_seen
                continue
            else:
                current.insert(i, saved)
                # seen remains unchanged as we reverted

    return "".join(map(str, current))
