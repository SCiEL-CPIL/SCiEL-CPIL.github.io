"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """For each element x, find all pairs (y,z) such that {x,y,z} is a bad triple.
    conflicts[x] = set of frozenset({y,z}) for each bad triple containing x."""
    conflicts = {x: set() for x in range(1, 129)}
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))
    return conflicts

def _build_bitset_conflicts():
    """Build bitset-based conflict structure for branch-and-bound.
    conflict_bits[i] is a 128-bit integer where bit j is set if i and j cannot both be in an independent set.
    """
    n = 128
    # First build pairwise conflicts: two elements conflict if there exists a third that forms a bad triple with both
    # Actually, for maximum independent set in a 3-uniform hypergraph, we need:
    # A set S is independent if no hyperedge is fully contained in S.
    # So for branch-and-bound, when we consider adding vertex v, we need to check if any hyperedge 
    # that contains v has its other two vertices already selected.
    
    # Let's build: for each vertex v, the set of pairs (a,b) such that {v,a,b} is a hyperedge
    # Then when building the solution, if we've selected both a and b, we cannot select v.
    
    # For efficient branch-and-bound with bitsets:
    # We'll use a different approach. Represent the problem as:
    # We want to select max subset S of {1..128} such that for no triple {a,b,c} in hyperedges are all three in S.
    
    # Bitset approach: 
    # For each element i, store bitset of elements that "conflict" with i in the sense that
    # if i is selected, certain combinations of other elements become forbidden.
    
    # Actually, let's think differently. In a 3-uniform hypergraph, the independence condition is:
    # For every hyperedge {a,b,c}, not all of a,b,c are in S.
    
    # For branch-and-bound, we can use:
    # - Include vertex v: then for every hyperedge {v, a, b}, we must exclude at least one of a, b.
    #   This means we can't simply mark a and b as forbidden; we need to ensure at least one is excluded.
    #   This makes simple bitset pruning harder.
    
    # Alternative: convert to a graph problem. Two vertices conflict if they appear together in some hyperedge
    # AND there's a third vertex that also appears with both. But that's not quite right either.
    
    # Let me reconsider. The key insight for 3-uniform hypergraph independent set:
    # When we include vertex v, we need to ensure that for every hyperedge {v, a, b}, 
    # at least one of a or b is not in our final set.
    
    # For branch-and-bound with memoization, we can use:
    # state = (remaining_vertices_bitset, ...)
    # But the state space might be too large.
    
    # Let's try a different approach: use the fact that the hypergraph is sparse.
    # Build for each vertex v: the list of hyperedges containing v.
    # Then use recursive branch-and-bound with pruning.
    
    hyperedges = []
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            hyperedges.append((a, b, c))
    
    # For each vertex, store indices of hyperedges containing it
    vertex_to_hyperedges = {v: [] for v in range(1, 129)}
    for idx, (a, b, c) in enumerate(hyperedges):
        vertex_to_hyperedges[a].append(idx)
        vertex_to_hyperedges[b].append(idx)
        vertex_to_hyperedges[c].append(idx)
    
    return hyperedges, vertex_to_hyperedges

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    for pair in conflicts[x]:
        if pair & selected_set == pair:
            return False
    return True

def _greedy(conflicts, order):
    """Greedy: add elements in given order if they don't create a bad triple."""
    selected = []
    selected_set = set()
    for x in order:
        if _can_add(x, selected_set, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, conflicts, time_limit=10):
    """Local search: try removing one element and adding another to improve size."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try all single removals
        for rem in list(selected_set):
            candidate = selected_set - {rem}
            # Try to add any element not in candidate
            for x in range(1, 129):
                if x in candidate:
                    continue
                if _can_add(x, candidate, conflicts):
                    new_set = candidate | {x}
                    if len(new_set) > best_size:
                        best = list(new_set)
                        best_size = len(best)
                        selected_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try double removals
            improved = False
            for rem1, rem2 in combinations(list(selected_set), 2):
                candidate = selected_set - {rem1, rem2}
                for x in range(1, 129):
                    if x in candidate:
                        continue
                    if _can_add(x, candidate, conflicts):
                        new_set = candidate | {x}
                        if len(new_set) > best_size:
                            best = list(new_set)
                            best_size = len(best)
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    return best

def _branch_and_bound(hyperedges, vertex_to_hyperedges, deadline, best_size_ref):
    """Branch-and-bound for maximum independent set in 3-uniform hypergraph.
    
    Uses recursive search with pruning:
    - If current_size + remaining_count <= best_size, prune.
    - For each vertex, decide include or exclude.
    - When including vertex v, for each hyperedge {v, a, b}, we must exclude at least one of a, b.
    
    Returns (best_solution, best_size)
    """
    n = 128
    # Map vertices 1-128 to indices 0-127 for bitset operations
    # Use Python integers as bitsets (arbitrary precision)
    
    # Precompute: for each vertex v, the set of hyperedges containing v
    # and for each such hyperedge, the other two vertices
    
    # For efficient pruning, precompute for each vertex v:
    # - The count of hyperedges containing v
    # - The "constraint" that if v is included, certain pairs must be broken
    
    # Let's use a simpler approach: recursive with memoization on (available_bitset)
    # But available_bitset has 2^128 states, too many.
    
    # Instead, use branch-and-bound with good ordering and pruning.
    
    # Order vertices by degree (number of hyperedges they're in) for better pruning
    vertex_order = sorted(range(1, 129), key=lambda v: len(vertex_to_hyperedges[v]), reverse=True)
    
    # Convert to 0-indexed
    vertex_order_0 = [v - 1 for v in vertex_order]
    
    best_solution = set()
    best_size = 0
    current_solution = set()
    
    # For fast checking, maintain:
    # - selected: set of selected vertices
    # - For each hyperedge, track how many of its vertices are selected
    
    # Actually, let's use a different representation.
    # available: set of vertices that can still be selected
    # When we select vertex v, we need to ensure no hyperedge becomes fully selected.
    # A hyperedge {a,b,c} becomes fully selected only if all three are in our solution.
    # So when adding v, we check: is there a hyperedge {v, a, b} where both a and b are already selected?
    
    # For efficiency, maintain for each vertex v: the set of pairs (a,b) such that {v,a,b} is a hyperedge
    # and a, b are both in the current partial solution.
    # If this set is non-empty, v cannot be added.
    
    # Let's precompute pair_constraints[v] = set of (a,b) where {v,a,b} is a hyperedge
    pair_constraints = {v: set() for v in range(1, 129)}
    for a, b, c in hyperedges:
        pair_constraints[a].add(frozenset((b, c)))
        pair_constraints[b].add(frozenset((a, c)))
        pair_constraints[c].add(frozenset((a, b)))
    
    # Now use branch-and-bound
    # State: (available_vertices, selected_vertices)
    # Prune if len(selected) + len(available) <= best_size
    
    def _bbsolve(available, selected, current_size):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: upper bound
        if current_size + len(available) <= best_size:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_solution = set(selected)
            return
        
        # Choose vertex to branch on: pick the one with highest degree in the induced subgraph
        # For simplicity, pick the first available vertex in our ordering
        # Better: pick the vertex that appears in the most hyperedges with other available vertices
        
        # Simple heuristic: pick vertex with most pair_constraints that are "active"
        # (i.e., both other vertices are in available)
        best_v = None
        best_score = -1
        for v in available:
            score = 0
            for pair in pair_constraints[v]:
                a, b = tuple(pair)
                if a in available and b in available:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v is None:
            best_v = next(iter(available))
        
        # Branch 1: Include best_v
        # Check if we can include it: no pair in pair_constraints[best_v] should have both elements in selected
        can_include = True
        for pair in pair_constraints[best_v]:
            a, b = tuple(pair)
            if a in selected and b in selected:
                can_include = False
                break
        
        if can_include:
            new_selected = selected | {best_v}
            new_available = available - {best_v}
            _bbsolve(new_available, new_selected, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_available = available - {best_v}
        _bbsolve(new_available, selected, current_size)
    
    _bbsolve(set(range(1, 129)), set(), 0)
    
    return list(best_solution), best_size

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute conflict structure
    conflicts = _build_conflicts()
    
    best = []
    best_size = 0
    
    # Strategy 1: Greedy with natural order
    sel = _greedy(conflicts, list(range(1, 129)))
    best = sel
    best_size = len(sel)
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline - 30:
        sel = _greedy(conflicts, list(range(128, 0, -1)))
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Strategy 3: Local search on best
    if time.time() < deadline - 25:
        improved = _local_search(best, conflicts, time_limit=8)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Strategy 4: Multiple random restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 20 and restarts < 200:
        restarts += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel = _greedy(conflicts, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Improve if close to best
        if len(sel) >= best_size - 3 and time.time() < deadline - 25:
            improved = _local_search(sel, conflicts, time_limit=3)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
    
    # Strategy 5: Branch-and-bound with time limit
    if time.time() < deadline - 5:
        hyperedges, vertex_to_hyperedges = _build_bitset_conflicts()
        bbsol, bb_size = _branch_and_bound(hyperedges, vertex_to_hyperedges, deadline - 2, best_size)
        if bb_size > best_size:
            best = bbsol
            best_size = bb_size
    
    # Strategy 6: Final local search
    if time.time() < deadline - 3 and len(best) > 0:
        improved = _local_search(best, conflicts, time_limit=3)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    if not best:
        return [1]
    
    return best
# EVOLVE-BLOCK-END
