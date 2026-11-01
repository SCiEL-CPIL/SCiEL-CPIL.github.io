import numpy as np
import itertools
import time
from collections import deque

def construct():
    D = 11
    deadline = time.time() + 140  # Safety margin
    
    candidates = []
    
    # Type 1: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 2: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Conflict if dot > 2 (since norm^2 is 4 for all, dist^2 = 4+4-2dot = 8-2dot >= 4 => dot <= 2)
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    # Convert to list of conflicts for faster lookup
    conflict_sets = [set(np.where(conflict[i])[0]) for i in range(n)]
    
    # Build adjacency list as sorted arrays for faster operations
    conflict_adj = [sorted(conflict_sets[i]) for i in range(n)]
    
    # Initial greedy solution to establish a lower bound
    def greedy_solution(order=None):
        if order is None:
            order = list(range(n))
        selected = []
        selected_set = set()
        for idx in order:
            if not conflict_sets[idx].intersection(selected_set):
                selected.append(idx)
                selected_set.add(idx)
        return selected, selected_set
    
    # Get initial lower bound
    init_selected, init_set = greedy_solution()
    best_size = len(init_selected)
    best_selected = init_selected.copy()
    
    # Branch and bound with dynamic variable ordering
    # State: (selected_set, available_nodes, current_size)
    # We use a priority queue ordered by potential (upper bound)
    
    # Upper bound: maximum independent set size in the induced subgraph on available nodes
    # Simple bound: number of available nodes
    # Better bound: greedy coloring of conflict graph on available nodes gives an upper bound
    # For now, use a simple bound and improve if time permits
    
    def compute_upper_bound(available):
        """Compute an upper bound on the independent set size in the subgraph induced by available nodes."""
        if not available:
            return 0
        # Simple bound: |available|
        # Better: use greedy coloring
        # Even better: use the fact that we can greedily pick non-conflicting nodes
        # Let's use a simple greedy coloring approach
        available_set = set(available)
        color_class_sizes = []
        remaining = set(available)
        while remaining:
            # Pick a random node from remaining
            node = next(iter(remaining))
            color_class = {node}
            remaining.remove(node)
            # Add all non-conflicting nodes to this color class
            for other in list(remaining):
                if other not in conflict_sets[node] and not (color_class & conflict_sets[other]):
                    color_class.add(other)
                    remaining.remove(other)
            color_class_sizes.append(len(color_class))
        # Upper bound is the sum of the largest color classes... actually for independent set,
        # each color class can contribute at most 1 to an independent set? No, that's for clique cover.
        # For independent set, the upper bound from coloring is the number of colors used.
        # Actually, a proper coloring gives an upper bound on the clique number, not independent set.
        # For independent set, we need a different bound.
        # Simple bound: |available| is valid but weak.
        # Better: use the maximum degree to bound.
        # Let's just use |available| for now and add pruning based on current best.
        return len(available)
    
    # Branch and bound search
    # We'll use a stack-based DFS with pruning
    # State: (available_nodes_list, selected_count, selected_set)
    # We order variables by "conflict freedom" = number of available nodes they don't conflict with
    
    # Precompute for each node its conflict set as a bitset for faster operations?
    # n is ~5300, so we can use Python sets.
    
    # Better approach: use a recursive B&B with memoization-like pruning
    # But n is too large for full memoization.
    
    # Let's use a different strategy: iterative deepening with pruning
    # Or: use the fact that the problem has structure (Leech lattice minimal vectors)
    
    # Actually, let's try a smarter greedy + local search with better heuristics
    
    # Phase 1: Multiple greedy restarts with different orderings
    rng = np.random.RandomState(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if attempt % 3 == 0:
            order = rng.permutation(n)
        elif attempt % 3 == 1:
            # Order by degree (fewest conflicts first)
            degrees = [len(conflict_sets[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: degrees[x])
        else:
            # Order by random perturbation of degree
            degrees = [len(conflict_sets[i]) + rng.uniform(-10, 10) for i in range(n)]
            order = sorted(range(n), key=lambda x: degrees[x])
        
        selected, selected_set = greedy_solution(order)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
        
        # Local search: try to improve by swaps
        if time.time() > deadline:
            break
        
        improved = True
        max_swaps = 50
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            swap_count += 1
            
            # Try to find a swap that increases the set size
            # Pick a random selected node to remove
            if not selected:
                break
            
            to_remove = selected[rng.randint(len(selected))]
            selected_set.remove(to_remove)
            
            # Try to add as many compatible nodes as possible
            added = 0
            for idx in range(n):
                if idx in selected_set:
                    continue
                if not conflict_sets[idx].intersection(selected_set):
                    selected_set.add(idx)
                    added += 1
                    if time.time() > deadline:
                        break
            
            # Rebuild selected list
            selected = list(selected_set)
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected.copy()
                improved = True
            elif len(selected) < len(init_selected) if init_selected else False:
                # Revert if we got worse
                selected_set = set(init_selected)
                selected = init_selected.copy()
    
    return candidates[best_selected]
