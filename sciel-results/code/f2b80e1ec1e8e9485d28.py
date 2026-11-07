"""Cochromatic partition solver with backtracking, iterative deepening, and memoization."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
EDGE_SET = set(tuple(e) for e in P['edges'])
EDGE_SET.update((b, a) for a, b in EDGE_SET)
ADJ = [set() for _ in range(N)]
for a, b in EDGE_SET:
    ADJ[a].add(b)
    ADJ[b].add(a)

DEADLINE = time.time() + 85.0  # safety margin before 90s kill

def is_clique(S):
    """Check if set S induces a clique."""
    for u, v in combinations(sorted(S), 2):
        if v not in ADJ[u]:
            return False
    return True

def is_independent(S):
    """Check if set S induces an independent set."""
    for u, v in combinations(sorted(S), 2):
        if v in ADJ[u]:
            return False
    return True

def can_extend_clique(S, v):
    """Check if adding v to clique S keeps it a clique."""
    for u in S:
        if v not in ADJ[u]:
            return False
    return True

def can_extend_independent(S, v):
    """Check if adding v to independent set S keeps it independent."""
    for u in S:
        if v in ADJ[u]:
            return False
    return True

def lower_bound(remaining, used_parts):
    """Lower bound on additional parts needed.
    We need at least 1 more part if remaining is non-empty.
    Stronger: partition remaining into min clique/independent parts.
    Simple bound: 1 if remaining nonempty.
    """
    if not remaining:
        return 0
    return 1

def solve(parts, remaining, depth_limit):
    """Backtracking search. parts is list of frozensets (each clique or indep).
    remaining is a frozenset of unassigned vertices.
    Returns True if we can assign all remaining using at most depth_limit more parts.
    """
    if not remaining:
        return True
    if len(parts) + 1 > depth_limit:
        return False
    if time.time() > DEADLINE:
        return False
    
    # Pick the vertex with fewest options (most constrained)
    # For each remaining vertex, count how many existing parts it can join
    best_v = None
    best_options = None
    for v in remaining:
        options = 0
        for i, part in enumerate(parts):
            if can_extend_clique(part, v) or can_extend_independent(part, v):
                options += 1
        # It can also start a new part (counts as 1 option always)
        options += 1
        if best_options is None or options < best_options:
            best_options = options
            best_v = v
            if options == 1:
                break
    
    v = best_v
    rem_after = remaining - {v}
    
    # Try adding v to existing parts
    for i, part in enumerate(parts):
        if can_extend_clique(part, v):
            new_parts = list(parts)
            new_parts[i] = part | {v}
            if solve(new_parts, rem_after, depth_limit):
                return True
        if can_extend_independent(part, v):
            new_parts = list(parts)
            new_parts[i] = part | {v}
            if solve(new_parts, rem_after, depth_limit):
                return True
    
    # Try starting a new part with v
    new_parts = list(parts)
    new_parts.append({v})
    if solve(new_parts, rem_after, depth_limit):
        return True
    
    return False

def greedy_partition():
    """Greedy: assign each vertex to first compatible part, else new part."""
    parts = []
    for v in range(N):
        placed = False
        for i, part in enumerate(parts):
            if can_extend_clique(part, v):
                parts[i] = part | {v}
                placed = True
                break
            if can_extend_independent(part, v):
                parts[i] = part | {v}
                placed = True
                break
        if not placed:
            parts.append({v})
    return parts

def construct():
    global DEADLINE
    DEADLINE = time.time() + 85.0
    
    # Start with greedy solution
    greedy = greedy_partition()
    best_count = len(greedy)
    
    # Try to improve with iterative deepening from 1 upward
    # Lower bound: chromatic number of graph and its complement
    # For n=14, try small values first
    
    for k in range(1, best_count):
        if time.time() > DEADLINE:
            break
        # Try to find a partition with exactly k parts
        # Use backtracking with depth limit k
        if solve([], frozenset(range(N)), k):
            best_count = k
            break
    
    # Now we know best_count, but we need to actually produce the partition
    # Re-run with the found count to get the actual parts
    if best_count < len(greedy):
        result_parts = None
        # We need to recover the actual partition, not just verify existence
        # Modify solve to return the partition
        def solve_with_result(parts, remaining, depth_limit):
            if not remaining:
                return parts
            if len(parts) >= depth_limit:
                return None
            if time.time() > DEADLINE:
                return None
            
            best_v = None
            best_options = None
            for v in remaining:
                options = 0
                for i, part in enumerate(parts):
                    if can_extend_clique(part, v) or can_extend_independent(part, v):
                        options += 1
                options += 1
                if best_options is None or options < best_options:
                    best_options = options
                    best_v = v
                    if options == 1:
                        break
            
            v = best_v
            rem_after = remaining - {v}
            
            for i, part in enumerate(parts):
                if can_extend_clique(part, v):
                    new_parts = list(parts)
                    new_parts[i] = part | {v}
                    res = solve_with_result(new_parts, rem_after, depth_limit)
                    if res is not None:
                        return res
                if can_extend_independent(part, v):
                    new_parts = list(parts)
                    new_parts[i] = part | {v}
                    res = solve_with_result(new_parts, rem_after, depth_limit)
                    if res is not None:
                        return res
            
            new_parts = list(parts)
            new_parts.append({v})
            res = solve_with_result(new_parts, rem_after, depth_limit)
            if res is not None:
                return res
            
            return None
        
        result_parts = solve_with_result([], frozenset(range(N)), best_count)
        if result_parts is None:
            result_parts = greedy
    else:
        result_parts = greedy
    
    # Convert to list of lists, ensuring Python int types
    final_parts = []
    for part in result_parts:
        final_parts.append(sorted([int(v) for v in part]))
    
    return final_parts
# EVOLVE-BLOCK-END
