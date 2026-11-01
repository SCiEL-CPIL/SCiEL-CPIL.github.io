"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115.0
    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=n - 4)]
    
    total = 3**n
    # Precompute neighbors for all words
    # neighbors[i] is a list of words at Hamming distance <= 1 from i
    neighbors = np.zeros((total, 3*n - 2*n//2), dtype=int)  # 3*8 - 4 = 20? No, 1 + 8 = 9 neighbors
    neighbors = np.zeros((total, 9), dtype=int)
    for i in range(total):
        digits = np.array([0]*n)
        x = i
        for j in range(n):
            digits[j] = x % 3
            x //= 3
        digits = digits[::-1]  # little endian
        neighbors[i, 0] = i
        for j in range(n):
            for d in range(3):
                if d == digits[j]:
                    continue
                new_x = i + (d - digits[j]) * (3**j)
                neighbors[i, j+1] = new_x
    
    # Greedy set cover
    covered = np.zeros(total, dtype=np.uint8)
    counts = np.zeros(total, dtype=np.int32)
    code = []
    
    # We want to pick words that cover the most uncovered points.
    # But checking all 6561 words every step is O(6561 * 9 * |code|) which might be slow if |code| is large.
    # Actually, we can just maintain a "potential" score for each word?
    # Or just recompute scores every few steps?
    # Let's try:
    # 1. Pick word with max uncovered neighbors.
    # 2. Mark as covered.
    # 3. Update uncovered counts.
    
    # To speed up, we can precompute the list of neighbors for each word.
    # Score of word w = number of neighbors of w that are currently uncovered.
    
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break
        
        # Find word with max uncovered neighbors
        # scores = np.sum(1 - covered[neighbors], axis=1)
        # This is vectorized and should be fast enough for 6561 x 9.
        scores = np.sum(1 - covered[neighbors], axis=1)
        max_score = scores.max()
        if max_score == 0:
            break
        # Pick one of the max score words (randomly if ties, or just argmax)
        # random tie-breaking can help escape local optima
        candidates = np.where(scores == max_score)[0]
        w = candidates[random.randint(0, len(candidates)-1)]
        code.append(w)
        
        # Update covered and counts
        for nb in neighbors[w]:
            if not covered[nb]:
                covered[nb] = 1
                counts[nb] += 1
            else:
                counts[nb] += 1
        
        # If we have too many words, we might want to stop and try to optimize
        # Best known is 486. If we exceed 500, we can start removing.
        
    # Redundancy removal
    # A word w is removable if for all nb in neighbors[w], counts[nb] >= 2.
    # i.e., removing w leaves counts[nb] >= 1 for all nb.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code) - 1, -1, -1):
            w = code[i]
            removable = True
            for nb in neighbors[w]:
                if counts[nb] < 2:
                    removable = False
                    break
            if removable:
                code.pop(i)
                for nb in neighbors[w]:
                    counts[nb] -= 1
                improved = True
                if time.time() > deadline:
                    break
    
    # Convert to list of lists of ints
    result = []
    for w in code:
        digits = [0]*n
        x = w
        for j in range(n):
            digits[j] = x % 3
            x //= 3
        result.append(digits)
    
    return result
# EVOLVE-BLOCK-END
