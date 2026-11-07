"""Directed backtracking with MRV heuristic for MOLS of order 10."""
import time
import random

N = 10
MAX_SQUARES = 9

def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]

def _verify(S):
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
                if not (0 <= v <= N - 1):
                    return False
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
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
    """Find a Latin square orthogonal to every square in S using MRV backtracking."""
    K = len(S)
    row_mask = [0] * N
    col_mask = [0] * N
    pair_sets = [set() for _ in range(K)]
    grid = [[-1] * N for _ in range(N)]
    
    # Precompute for fast access
    S_vals = [[[S[k][i][j] for j in range(N)] for i in range(N)] for k in range(K)]
    
    def get_candidates(i, j):
        """Get candidate symbols for cell (i,j)."""
        rm = row_mask[i]
        cm = col_mask[j]
        cands = []
        for s in range(N):
            bit = 1 << s
            if (rm & bit) or (cm & bit):
                continue
            ok = True
            for k in range(K):
                if (S_vals[k][i][j], s) in pair_sets[k]:
                    ok = False
                    break
            if ok:
                cands.append(s)
        return cands
    
    def solve():
        if time.time() > deadline:
            return None
        
        # MRV: find the empty cell with fewest candidates
        best_cell = None
        best_cands = None
        min_count = N + 1
        
        for i in range(N):
            for j in range(N):
                if grid[i][j] != -1:
                    continue
                cands = get_candidates(i, j)
                cnt = len(cands)
                if cnt < min_count:
                    min_count = cnt
                    best_cell = (i, j)
                    best_cands = cands
                    if cnt == 0:
                        return None
                    if cnt == 1:
                        break
            else:
                continue
            break
        
        if best_cell is None:
            # All cells filled
            return [row[:] for row in grid]
        
        i, j = best_cell
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
    deadline = t0 + 48.0  # Safety margin under 60s/90s limits
    random.seed()
    
    S = [_cyclic()]
    
    while time.time() < deadline and len(S) < MAX_SQUARES:
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Try multiple restarts for this square
        progress = False
        for attempt in range(3):
            rem = deadline - time.time()
            if rem < 0.5:
                break
            # Allocate time: first attempt gets more, subsequent less
            if attempt == 0:
                sub_deadline = t0 + rem * 0.6
            else:
                sub_deadline = t0 + rem * 0.35
            
            new_sq = _find_orthogonal(S, sub_deadline)
            if new_sq is not None:
                S.append(new_sq)
                progress = True
                break
        
        if not progress:
            break
    
    if _verify(S):
        return S
    for k in range(len(S), 0, -1):
        if _verify(S[:k]):
            return S[:k]
    return [_cyclic()]
