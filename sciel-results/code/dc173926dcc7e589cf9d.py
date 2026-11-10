import time
import sys

def _find_orthogonal_square_fast(base_sq, deadline):
    n = len(base_sq)
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pairs_used = set()
    
    # Precompute base values
    base_vals = [[base_sq[i][j] for j in range(n)] for i in range(n)]
    
    # State for backtracking
    # We will use a recursive function with MRV
    
    def get_candidates(i, j):
        base_val = base_vals[i][j]
        cands = []
        for v in range(n):
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            if (base_val, v) in pairs_used:
                continue
            cands.append(v)
        return cands

    def solve():
        if time.time() > deadline:
            return False
            
        # Find the unassigned cell with the fewest candidates (MRV)
        min_cands = None
        min_i = -1
        min_j = -1
        min_count = n + 1
        
        for i in range(n):
            for j in range(n):
                if sq[i][j] == -1:
                    cands = get_candidates(i, j)
                    if len(cands) == 0:
                        return False # Dead end
                    if len(cands) < min_count:
                        min_count = len(cands)
                        min_cands = cands
                        min_i = i
                        min_j = j
                        if min_count == 1:
                            break
            if min_count == 1:
                break
        
        if min_i == -1:
            return True # All cells assigned
            
        # Try candidates for the selected cell
        for v in min_cands:
            sq[min_i][min_j] = v
            row_used[min_i].add(v)
            col_used[min_j].add(v)
            pairs_used.add((base_vals[min_i][min_j], v))
            
            if solve():
                return True
            
            # Backtrack
            sq[min_i][min_j] = -1
            row_used[min_i].discard(v)
            col_used[min_j].discard(v)
            pairs_used.discard((base_vals[min_i][min_j], v))
            
        return False

    if solve():
        return sq
    return None

def _construct_mols_10():
    n = 10
    # Base square: cyclic group Z_10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    # Deadline: 90s limit, use 80s for safety.
    deadline = time.time() + 80
    
    # Find second square
    sq2 = _find_orthogonal_square_fast(base, deadline)
    if sq2 is not None:
        squares.append(sq2)
        
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
