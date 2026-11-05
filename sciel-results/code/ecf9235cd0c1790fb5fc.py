"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START = time.time()
_DEADLINE = 82.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    """
    Constructs 101 lists of denominators (1-10) such that the sum of their
    modular inverses (mod 101) equals the list index.
    Minimizes the maximum length of any list.
    Uses BFS to find minimum size subsets for each residue.
    """
    n = 10
    mod = 101
    
    # Precompute modular inverses
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[i] = pow(i, -1, mod)
        
    # min_size[r] stores the minimum subset size found for residue r
    # best_mask[r] stores the bitmask of that subset
    min_size = [n + 1] * mod  # Initialize to infinity (n+1)
    best_mask = [0] * mod
    visited = [False] * mod
    
    # Queue: (mask, current_sum, size)
    # Start with empty set (mask 0, sum 0, size 0)
    queue = deque()
    queue.append((0, 0, 0))
    
    found_count = 0
    
    while queue:
        if _elapsed() > _DEADLINE:
            break
            
        mask, curr_sum, size = queue.popleft()
        
        # If this is a non-empty set, check if it's the first time we've seen this residue
        if size > 0:
            r = curr_sum % mod
            if not visited[r]:
                visited[r] = True
                min_size[r] = size
                best_mask[r] = mask
                found_count += 1
                if found_count == mod:
                    break
        
        # Expand: add each denominator not in mask
        for i in range(1, n + 1):
            bit = 1 << (i - 1)
            if not (mask & bit):
                new_mask = mask | bit
                new_sum = (curr_sum + inv[i]) % mod
                new_size = size + 1
                queue.append((new_mask, new_sum, new_size))
                
    # Construct the result rows
    rows = []
    for r in range(mod):
        # If not visited (should not happen if deadline not hit and problem valid),
        # we leave the mask as 0, which results in an empty list.
        # However, the problem requires non-empty lists.
        # If a residue is not found, it implies an error or timeout.
        # Given the small search space, this is unlikely.
        
        mask = best_mask[r]
        terms = []
        for i in range(1, n + 1):
            if (mask >> (i - 1)) & 1:
                terms.append(i)
        
        rows.append(terms)
        
    return rows
# EVOLVE-BLOCK-END
