# EVOLVE-BLOCK-START
import time

def _is_latin(square):
    n = len(square)
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [square[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(s1, s2):
    n = len(s1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (s1[i][j], s2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n * n

def _find_orthogonal_square(base_sq, deadline):
    n = len(base_sq)
    sq = [[-1 for _ in range(n)] for _ in range(n)]
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pairs_used = set()
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        
        base_val = base_sq[i][j]
        
        # Heuristic: Try values that are least constraining? 
        # For n=10, simple order 0-9 is fast enough.
        for v in range(n):
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            pair = (base_val, v)
            if pair in pairs_used:
                continue
            
            sq[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pairs_used.add(pair)
            
            if solve(idx + 1):
                return True
            
            sq[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            pairs_used.discard(pair)
            
        return False

    if solve(0):
        return sq
    return None

def _construct_mols_10():
    n = 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    # Use a generous deadline. n=10 is small.
    deadline = time.time() + 30.0
    
    sq2 = _find_orthogonal_square(base, deadline)
    if sq2 is not None:
        # Verify just in case
        if _is_latin(sq2) and _is_orthogonal(base, sq2):
            squares.append(sq2)
        else:
            # Should not happen, but safety check
            pass
            
    # N(10) is 2. Do not attempt 3rd.
    return squares

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
        
    # Final safety: if result is empty, return at least 1 square
    if not result:
        n = 10
        result = [[[ (i+j)%n for j in range(n)] for i in range(n)]]
        
    return result
# EVOLVE-BLOCK-END
