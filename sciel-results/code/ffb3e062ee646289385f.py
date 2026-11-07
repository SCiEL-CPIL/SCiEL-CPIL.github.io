"""
Find a large set of mutually orthogonal Latin squares (MOLS) of order 10
using a group-based construction with backtracking for remaining squares.

Key insight: Use the group Z5 x Z2 to construct Latin squares.
Each square is defined by a permutation of group elements, and orthogonality
is ensured through careful construction of the permutation structure.
"""
import time
import random

N = 10
MAX_SQUARES = 9  # Upper bound for MOLS of order 10

def _element_index(a, b):
    """Map (a, b) in Z5 x Z2 to index 0..9"""
    return a * 2 + b

def _index_element(idx):
    """Map index 0..9 to (a, b) in Z5 x Z2"""
    return (idx // 2, idx % 2)

def _group_square(permutation):
    """
    Create a Latin square from a permutation of the group Z5 x Z2.
    The square at (i, j) = permutation(i + j) where + is group addition.
    """
    # Precompute group addition table
    group_add = [[0] * N for _ in range(N)]
    for i in range(N):
        a1, b1 = _index_element(i)
        for j in range(N):
            a2, b2 = _index_element(j)
            a_sum = (a1 + a2) % 5
            b_sum = (b1 + b2) % 2
            group_add[i][j] = _element_index(a_sum, b_sum)
    
    sq = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            sq[i][j] = permutation[group_add[i][j]]
    return sq

def _verify(S):
    """Verify that S is a valid set of MOLS."""
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
    Find a Latin square orthogonal to every square in S using backtracking.
    Uses bitmask constraints and MRV heuristics.
    """
    K = len(S)
    row_mask = [0] * N
    col_mask = [0] * N
    pair_sets = [set() for _ in range(K)]
    
    grid = [[-1] * N for _ in range(N)]
    
    # Precompute S[k][i][j] for faster access
    S_vals = [[[S[k][i][j] for j in range(N)] for i in range(N)] for k in range(K)]
    
    # Fix the first row to 0..9 (WLOG for MOLS)
    for j in range(N):
        s = j
        grid[0][j] = s
        bit = 1 << s
        row_mask[0] |= bit
        col_mask[j] |= bit
        for k in range(K):
            pair_sets[k].add((S_vals[k][0][j], s))
    
    def solve():
        if time.time() > deadline:
            return None
        
        best_cell = None
        best_cands = None
        min_cands_count = N + 1
        
        # Iterate over all empty cells
        for i in range(1, N):
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

def _generate_group_squares():
    """
    Generate Latin squares from the group structure Z5 x Z2.
    Use different permutations to create orthogonal squares.
    """
    squares = []
    used = set()
    
    # Generate permutations of Z5 x Z2
    # Try to find orthogonal squares by using different group automorphisms
    for _ in range(20):  # Try up to 20 permutations
        perm = list(range(N))
        random.shuffle(perm)
        
        # Check if this permutation creates a new orthogonal square
        sq = _group_square(perm)
        
        # Check orthogonality with existing squares
        orthogonal = True
        for existing in squares:
            pairs = set()
            for i in range(N):
                for j in range(N):
                    pairs.add((sq[i][j], existing[i][j]))
            if len(pairs) != N * N:
                orthogonal = False
                break
        
        if orthogonal and sq not in used:
            squares.append(sq)
            used.add(sq)
            if len(squares) >= MAX_SQUARES:
                break
    
    return squares

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin for 60s/90s limits
    random.seed(42)  # Fixed seed for reproducibility
    
    # Strategy 1: Try group-based construction first
    group_squares = _generate_group_squares()
    
    if len(group_squares) >= 3:
        S = group_squares
    else:
        # Strategy 2: Start with cyclic square and use backtracking
        S = [[(i + j) % N for j in range(N)] for i in range(N)]
    
    # Try to add more squares using backtracking
    while time.time() < deadline and len(S) < MAX_SQUARES:
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Allocate a sub-deadline for this search attempt
        search_deadline = t0 + remaining * 0.8
        
        new_sq = _find_orthogonal(S, search_deadline)
        
        if new_sq is not None:
            S.append(new_sq)
        else:
            break
    
    # Verify the result
    if _verify(S):
        return S
    
    # Fallback: return a valid prefix
    for k in range(len(S), 0, -1):
        if _verify(S[:k]):
            return S[:k]
    
    # Last resort
    return [[(i + j) % N for j in range(N)] for i in range(N)]
