"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    nums = list(range(2, N + 1))
    
    # Precompute gcd for all pairs
    n = len(nums)
    gcd_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_table[i][j] = gcd(nums[i], nums[j])
    
    def can_add(a, S):
        """Check if a can be added to set S without creating violations."""
        ai = a - 2
        # Check: for b,c in S, b≠c, a | b*c
        for b in S:
            bi = b - 2
            req = nums[ai] // gcd_table[ai][bi]
            for c in S:
                if c == b:
                    continue
                if c % req == 0:
                    return False
        # Check: for b,c in S, b≠c, b | a*c
        for b in S:
            for c in S:
                if b == c:
                    continue
                if (a * c) % b == 0:
                    return False
        # Check: for b,c in S, b≠c, c | a*b
        for b in S:
            for c in S:
                if b == c:
                    continue
                if (a * b) % c == 0:
                    return False
        return True
    
    # Greedy: add numbers from largest to smallest
    current = set()
    for a in range(N, 1, -1):
        if time.time() > deadline:
            break
        if can_add(a, current):
            current.add(a)
    
    # Local search: try to add excluded numbers
    excluded = [a for a in range(2, N+1) if a not in current]
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for a in excluded:
            if a in current:
                continue
            if can_add(a, current):
                current.add(a)
                excluded.remove(a)
                improved = True
                break
        if not improved:
            break
    
    return sorted(list(current))
# EVOLVE-BLOCK-END
