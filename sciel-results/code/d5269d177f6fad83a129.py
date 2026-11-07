"""Find a large set of mutually orthogonal Latin squares (MOLS) of order 10
using a highly optimized backtracking search with MRV and bitmasks."""
import time
import random

N = 10
MAX_SQUARES = 9  # Upper bound for MOLS of order 10

def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]

def _verify(S):
    # Structure check
    for sq in S:
        if not isinstance(sq, (list, tuple)):
            return False
        if len(sq) != N:
            return False
        for row in sq:
            if not isinstance(row, (list, tuple)):
                return False
            if len(row) != N:
                return False
            for v in row:
                if type(v) is not int:
                    return False
                if not (0 <= v < N):
                    return False
        # Latin property
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
    # Orthogonality
    m = len(S)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = set()
            for i in range(N):
                for j in range(N):
                    pairs.add((S[a][i][j], S[b][i][j]))
            if len(pairs) != N * N:
                return False
    return True

def _find_orthogonal(S, deadline):
    """
    Find a Latin square orthogonal to every square in S.
    Uses bitmask constraints and MRV heuristics.
    """
    K = len(S)
    # Bitmasks for constraints
    # row_mask[i] has bits for symbols already used in row i
    # col_mask[j] has bits for symbols already used in col j
    # pair_mask[k][i][j] has bits for symbols s such that (S[k][i][j], s) is used
    #   But this is not a simple mask on s for a given (i,j) because the constraint is specific.
    #   We need to check if (S[k][i][j], s) has been placed before.
    #   So for each k, we maintain a set of (a, b) pairs where a = S[k][i][j] and b is the symbol.
    #   Actually, it's better to store for each k, a set of pairs (a, b).
    
    row_mask = [0] * N
    col_mask = [0] * N
    pair_sets = [set() for _ in range(K)]
    
    grid = [[-1] * N for _ in range(N)]
    
    # Precompute S[k][i][j] for faster access
    S_vals = [[[S[k][i][j] for j in range(N)] for i in range(N)] for k in range(K)]
    
    # Heuristic: order cells by MRV (Minimum Remaining Values)
    # We'll use a priority queue or just sort at each step (N=10 is small enough)
    
    def solve():
        if time.time() > deadline:
            return None
        
        # Find the cell with minimum candidates
        best_cell = None
        best_cands = None
        min_cands_count = N + 1
        
        # Iterate over all empty cells
        for i in range(N):
            for j in range(N):
                if grid[i][j] != -1:
                    continue
                cands = []
                for s in range(N):
                    bit = 1 << s
                    if (row_mask[i] & bit) or (col_mask[j] & bit):
                        continue
                    # Check pair constraints
                    ok = True
                    for k in range(K):
                        if (S_vals[k][i][j], s) in pair_sets[k]:
                            ok = False
                            break
                    if ok:
                        cands.append(s)
                
                if len(cands) < min_cands_count:
                    min_cands_count = len(cands)
                    best_cell = (i, j)
                    best_cands = cands
                    if len(cands) == 0:
                        return None
                    if len(cands) == 1:
                        break
            else:
                continue
            break
        
        if best_cell is None:
            # All cells filled
            return [row[:] for row in grid]
        
        i, j = best_cell
        # Shuffle candidates for randomization
        random.shuffle(best_cands)
        
        for s in best_cands:
            bit = 1 << s
            grid[i][j] = s
            row_mask[i] |= bit
            col_mask[j] |= bit
            for k in range(K):
                pair_sets[k].add((S_vals[k][i][j], s))
            
            result = solve()
            if result is not None:
                return result
            
            grid[i][j] = -1
            row_mask[i] &= ~bit
            col_mask[j] &= ~bit
            for k in range(K):
                pair_sets[k].remove((S_vals[k][i][j], s))
        
        return None
    
    return solve()

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin for 60s/90s limits
    random.seed()
    
    # Start with the cyclic square
    S = [_cyclic()]
    
    # Try to add more squares
    while time.time() < deadline and len(S) < MAX_SQUARES:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Allocate a sub-deadline for this search attempt
        # Use a fraction of remaining time
        search_deadline = t0 + remaining * 0.7  # 70% of remaining time
        
        new_sq = _find_orthogonal(S, search_deadline)
        
        if new_sq is not None:
            S.append(new_sq)
        else:
            # If we couldn't find a new square within the time budget, 
            # try a shorter time or break. For now, break to avoid wasting time.
            break
    
    # Verify the result
    if _verify(S):
        return S
    
    # Fallback: return a valid prefix
    for k in range(len(S), 0, -1):
        if _verify(S[:k]):
            return S[:k]
    
    # Last resort
    return [_cyclic()]
