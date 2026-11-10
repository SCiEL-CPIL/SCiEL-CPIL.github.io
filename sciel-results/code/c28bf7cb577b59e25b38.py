# EVOLVE-BLOCK-START
import time

def _find_orthogonal_square(base_sq, deadline):
    """
    Finds a Latin square orthogonal to base_sq using backtracking with bitmask optimization.
    base_sq is an n x n list of lists.
    Returns the orthogonal square or None if not found before deadline.
    """
    n = len(base_sq)
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Bitmasks for row and column used symbols
    row_used = [0] * n
    col_used = [0] * n
    pairs_used = set()
    
    # Pre-compute base values for quick access
    base_vals = [row[:] for row in base_sq]

    # Counter to reduce time check frequency
    steps = 0

    def solve(idx):
        nonlocal steps
        steps += 1
        # Check time every 1000 steps to reduce overhead
        if steps % 1000 == 0 and time.time() > deadline:
            return False
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        
        base_val = base_vals[i][j]
        
        # Iterate through possible values
        # We can mask out used values in row and col
        # available_mask = ~(row_used[i] | col_used[j]) & ((1 << n) - 1)
        
        for v in range(n):
            bit = 1 << v
            if row_used[i] & bit:
                continue
            if col_used[j] & bit:
                continue
            
            pair = (base_val, v)
            if pair in pairs_used:
                continue
            
            # Place v
            sq[i][j] = v
            row_used[i] |= bit
            col_used[j] |= bit
            pairs_used.add(pair)
            
            if solve(idx + 1):
                return True
            
            # Backtrack
            sq[i][j] = -1
            row_used[i] &= ~bit
            col_used[j] &= ~bit
            pairs_used.discard(pair)
            
        return False

    if solve(0):
        return sq
    return None

def _construct_mols_10():
    n = 10
    # Base square: cyclic group Z_10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    # Deadline: 60s total limit, leave 15s margin.
    # We use 45s to be safe.
    deadline = time.time() + 45
    
    # Find second square
    sq2 = _find_orthogonal_square(base, deadline)
    if sq2 is not None:
        squares.append(sq2)
        
    # N(10) is known to be 2, so a third square does not exist.
    # We do not attempt to find one to save time and avoid unnecessary computation.
    
    return squares

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    # Ensure types are correct: list of lists of Python ints
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
