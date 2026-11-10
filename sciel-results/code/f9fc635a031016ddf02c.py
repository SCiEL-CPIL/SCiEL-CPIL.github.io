# EVOLVE-BLOCK-START
import time

def _find_orthogonal_square_fast(base_sq, deadline):
    """
    Finds a Latin square orthogonal to base_sq using optimized backtracking with bitmasks.
    """
    n = len(base_sq)
    sq = [[0] * n for _ in range(n)]
    
    # Use bitmasks for O(1) membership tests
    row_mask = [0] * n   # bit v set if v used in row i
    col_mask = [0] * n   # bit v set if v used in col j
    pair_mask = [0] * (n * n)  # bit (base_val * n + v) set if pair used
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        base_val = base_sq[i][j]
        
        # Try values in order; use bitmask for fast filtering
        for v in range(n):
            bit = 1 << v
            if row_mask[i] & bit:
                continue
            if col_mask[j] & bit:
                continue
            pair_bit = 1 << (base_val * n + v)
            if pair_mask[0] & pair_bit:  # We only need one pair_mask since each (base_val, v) is unique
                continue
            
            # Place v
            sq[i][j] = v
            row_mask[i] |= bit
            col_mask[j] |= bit
            pair_mask[0] |= pair_bit
            
            if solve(idx + 1):
                return True
            
            # Backtrack
            sq[i][j] = 0
            row_mask[i] &= ~bit
            col_mask[j] &= ~bit
            pair_mask[0] &= ~pair_bit
            
        return False

    if solve(0):
        return sq
    return None

def _construct_mols_10():
    n = 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    deadline = time.time() + 30  # Generous margin within 60s limit
    
    sq2 = _find_orthogonal_square_fast(base, deadline)
    if sq2 is not None:
        squares.append(sq2)
    
    return squares

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
