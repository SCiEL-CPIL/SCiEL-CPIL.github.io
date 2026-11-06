import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples as (a-1, b-1, c-1) zero-indexed
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))
    
    # For each vertex, store the set of (other1, other2) pairs such that (v, other1, other2) is a bad triple
    pairs_by_vertex = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_vertex[a].add(frozenset((b, c)))
        pairs_by_vertex[b].add(frozenset((a, c)))
        pairs_by_vertex[c].add(frozenset((a, b)))
    
    # Also: for each pair (i,j), list of vertices k such that (i,j,k) is a bad triple
    pair_to_thirds = {}
    for a, b, c in bad_triples:
        for pair in [(a, b), (a, c), (b, c)]:
            p = frozenset(pair)
            if p not in pair_to_thirds:
                pair_to_thirds[p] = set()
            other = (a, b, c)
            for x in other:
                if x not in pair:
                    pair_to_thirds[p].add(x)
                    break
    
    # Precompute for fast "can add" check:
    # can_add(selected_mask, v) = for all (u,w) in pairs_by_vertex[v], NOT (u in selected AND w in selected)
    # Equivalently: no pair (u,w) in pairs_by_vertex[v] has both bits set in selected_mask
    
    # Build adjacency for upper bound: vertex v is "adjacent" to u if they share a bad triple
    adj = [0] * n
    for a, b, c in bad_triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Degree in conflict graph
    degree = [bin(adj[i]).count('1') for i in range(n)]
    
    # Memoization: key = (available_mask, selected_mask) -> upper bound on additional vertices
    # This is expensive in memory, so we use a limited cache
    memo = {}
    MEMO_LIMIT = 200000
    
    def can_add(selected_mask, v):
        """Check if adding vertex v to selected_mask creates no bad triple."""
        for pair in pairs_by_vertex[v]:
            u, w = tuple(pair)
            if (selected_mask >> u) & 1 and (selected_mask >> w) & 1:
                return False
        return True
    
    def count_safe(available_mask, selected_mask):
        """Count vertices in available_mask that can be added to selected_mask."""
        count = 0
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if can_add(selected_mask, v):
                count += 1
            temp ^= lsb
        return count
    
    def upper_bound(available_mask, selected_mask, current_size):
        """Compute an upper bound on the final size."""
        # Simple bound: current_size + number of safe vertices
        safe = count_safe(available_mask, selected_mask)
        return current_size + safe
    
    def branch_and_bound(available_mask, selected_mask, current_size):
        nonlocal best_size, best_mask, time_limit_exceeded, memo
        
        if time.time() > deadline:
            time_limit_exceeded = True
            return
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        # Upper bound check
        ub = upper_bound(available_mask, selected_mask, current_size)
        if ub <= best_size:
            return
        
        # Memoization key (limited size)
        key = (available_mask, selected_mask)
        if len(memo) < MEMO_LIMIT:
            if key in memo:
                if memo[key] <= best_size - current_size:
                    return
            else:
                memo[key] = ub - current_size
        else:
            memo.clear()
            memo[key] = ub - current_size
        
        # Find best vertex to branch on: prefer high-degree, and prefer vertices that are "safe"
        # We want to branch on a vertex that is likely to be in the optimal solution
        # Strategy: pick a safe vertex with high degree in the remaining subgraph
        candidates = []
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if can_add(selected_mask, v):
                # Count how many other available vertices it conflicts with
                deg_in_sub = bin(adj[v] & available_mask).count('1')
                candidates.append((deg_in_sub, v))
            temp ^= lsb
        
        if not candidates:
            # No safe vertices; we can't add anything more
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        # Pick the candidate with highest degree (most constraining)
        candidates.sort(reverse=True)
        best_deg, best_v = candidates[0]
        
        # Branch 1: include best_v
        new_selected = selected_mask | (1 << best_v)
        # Remove best_v and all vertices that would form a bad triple with best_v given current selection
        # A vertex u in available_mask can be removed if (best_v, u, w) is a bad triple for some w in selected_mask
        # OR if (u, w1, w2) is a bad triple for w1, w2 in selected_mask (but that's already handled by can_add)
        # Actually, we should remove vertices that are now "unsafe" - but that's handled by can_add in recursion
        # For pruning: remove best_v from available
        new_available = available_mask & ~(1 << best_v)
        branch_and_bound(new_available, new_selected, current_size + 1)
        
        if time_limit_exceeded:
            return
        
        # Branch 2: exclude best_v
        new_available = available_mask & ~(1 << best_v)
        branch_and_bound(new_available, selected_mask, current_size)
    
    # Initial greedy solution
    best_size = 0
    best_mask = 0
    time_limit_exceeded = False
    
    # Multiple greedy orderings
    orderings = [
        sorted(range(n), key=lambda i: degree[i]),
        sorted(range(n), key=lambda i: degree[i], reverse=True),
        list(range(n)),
        list(range(n - 1, -1, -1)),
        sorted(range(n), key=lambda i: len(pairs_by_vertex[i])),
        sorted(range(n), key=lambda i: len(pairs_by_vertex[i]), reverse=True),
    ]
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        selected_mask = 0
        for node in order:
            if can_add(selected_mask, node):
                selected_mask |= (1 << node)
        sz = bin(selected_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = selected_mask
    
    # Run branch and bound
    all_mask = (1 << n) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # Local search: try to add vertices
    if not time_limit_exceeded and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for node in range(n):
                if time.time() > deadline - 5:
                    break
                if not (best_mask & (1 << node)):
                    if can_add(best_mask, node):
                        best_mask |= (1 << node)
                        improved = True
    
    # Swap local search: try replacing one vertex with two
    if not time_limit_exceeded and time.time() < deadline - 3:
        for _ in range(3):  # a few passes
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                if best_mask & (1 << i):
                    # Try removing i and adding two others
                    temp_mask = best_mask & ~(1 << i)
                    added = 0
                    for j in range(n):
                        if time.time() > deadline - 3:
                            break
                        if j != i and not (temp_mask & (1 << j)):
                            if can_add(temp_mask, j):
                                new_mask = temp_mask | (1 << j)
                                for k in range(n):
                                    if time.time() > deadline - 3:
                                        break
                                    if k != i and k != j and not (new_mask & (1 << k)):
                                        if can_add(new_mask, k):
                                            # Check size improvement
                                            new_size = bin(new_mask).count('1')
                                            old_size = bin(best_mask).count('1')
                                            if new_size > old_size:
                                                best_mask = new_mask
                                                improved = True
                                                added += 1
                                                break
                                if added > 0:
                                    break
                    if improved:
                        break
            if not improved:
                break
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]
