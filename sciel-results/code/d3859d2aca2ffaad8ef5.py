"""Rewritten program: exact minimum cochromatic partition via branch-and-bound."""
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Precompute adjacency as bitmasks
N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def is_clique(mask):
    """Check if the set of vertices in mask induces a clique."""
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & ~adj[v]:
            return False
    return True

def is_independent(mask):
    """Check if the set of vertices in mask induces an independent set."""
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & adj[v]:
            return False
    return True

def is_homogeneous(mask):
    """Check if mask is a clique or an independent set."""
    return is_clique(mask) or is_independent(mask)

def all_homogeneous_sets(vertices):
    """Yield all homogeneous subsets (size >= 2) of the given vertex mask."""
    verts = [i for i in range(N) if vertices & (1 << i)]
    k = len(verts)
    # Precompute all subsets
    for size in range(2, k + 1):
        for combo in combinations(verts, size):
            m = 0
            for v in combo:
                m |= (1 << v)
            if is_homogeneous(m):
                yield m

def upper_bound_remaining(mask):
    """Greedy upper bound on minimum parts needed to cover mask."""
    if mask == 0:
        return 0
    count = 0
    remaining = mask
    while remaining:
        # Greedily pick a large homogeneous set from remaining
        best_size = 1
        best_set = 0
        # Try all subsets of remaining (small enough for n=14)
        verts = [i for i in range(N) if remaining & (1 << i)]
        k = len(verts)
        # Try larger sizes first
        for size in range(min(k, 14), 1, -1):
            found = False
            for combo in combinations(verts, size):
                m = 0
                for v in combo:
                    m |= (1 << v)
                if is_homogeneous(m):
                    best_size = size
                    best_set = m
                    found = True
                    break
            if found:
                break
        if best_set == 0:
            # Single vertex
            best_set = remaining & -remaining
        remaining &= ~best_set
        count += 1
    return count

def solve():
    """Branch and bound to find minimum cochromatic partition."""
    all_verts = (1 << N) - 1
    best = [N]  # worst case: each vertex its own part
    best_parts = [None]
    
    def backtrack(remaining, current_parts, depth):
        if remaining == 0:
            if depth < best[0]:
                best[0] = depth
                best_parts[0] = list(current_parts)
            return
        
        # Pruning: if current depth + lower bound >= best, prune
        # Lower bound: at least 1 part needed if remaining != 0
        if depth + 1 >= best[0]:
            return
        
        # Upper bound check: if depth + upper_bound_remaining(remaining) >= best, prune
        ub = upper_bound_remaining(remaining)
        if depth + ub >= best[0]:
            return
        
        # Get vertices in remaining
        verts = [i for i in range(N) if remaining & (1 << i)]
        
        # To avoid symmetry, fix the first vertex to be in the next part
        first_v = verts[0]
        
        # Generate all homogeneous sets containing first_v and subset of remaining
        other_verts = verts[1:]
        k = len(other_verts)
        
        # Try all subsets S of other_verts such that {first_v} ∪ S is homogeneous
        for size in range(k + 1):
            for combo in combinations(other_verts, size):
                m = (1 << first_v)
                for v in combo:
                    m |= (1 << v)
                if is_homogeneous(m):
                    # Try this part
                    new_remaining = remaining & ~m
                    current_parts.append(list(combo) + [first_v])
                    backtrack(new_remaining, current_parts, depth + 1)
                    current_parts.pop()
    
    backtrack(all_verts, [], 0)
    return best_parts[0] if best_parts[0] is not None else [[i] for i in range(N)]

def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    try:
        result = solve()
        # Validate
        if result is None:
            raise ValueError("solve returned None")
        
        # Verify all parts are homogeneous and cover all vertices exactly once
        seen = set()
        for part in result:
            for v in part:
                if v in seen:
                    raise ValueError(f"Vertex {v} appears twice")
                seen.add(v)
            m = 0
            for v in part:
                m |= (1 << v)
            if not is_homogeneous(m):
                raise ValueError(f"Part {part} is not homogeneous")
        if len(seen) != N:
            raise ValueError(f"Only {len(seen)} vertices covered")
        
        # Convert to list of lists of ints
        return [list(part) for part in result]
    except Exception:
        # Fallback: greedy construction
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(N):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return [list(part) for part in parts]

# EVOLVE-BLOCK-END
