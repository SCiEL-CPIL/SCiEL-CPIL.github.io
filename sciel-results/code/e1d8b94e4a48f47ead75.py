import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590
    
    # Generate all integer vectors with squared norm 4 in Z^11
    # These are vectors with exactly 4 non-zero entries, each being +/-1
    candidates = []
    for comb in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[comb[0]] = s1
                        v[comb[1]] = s2
                        v[comb[2]] = s3
                        v[comb[3]] = s4
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # Compute dot product matrix
    dots = candidates @ candidates.T
    
    # Conflict: squared distance < 4, i.e., ||u-v||^2 < 4
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # So conflict if 8 - 2*u.v < 4, i.e., u.v > 2
    # Also need to exclude self (u.v = 4)
    
    conflicts = (dots > 2)
    np.fill_diagonal(conflicts, False)
    
    # Convert to adjacency list for branch-and-bound
    adj = [set(np.where(conflicts[i])[0]) for i in range(N)]
    
    # Greedy coloring for upper bound (DSATUR-style heuristic)
    # We use a simple greedy coloring to get an upper bound on the independent set
    # Actually, for maximum independent set, we want to color the complement graph
    # or use the fact that the independence number <= number of colors in any coloring
    
    # Let's use a branch and bound approach with a good upper bound
    
    # Upper bound: for a set of vertices, the max independent set is at most
    # the number of colors needed to color the subgraph induced by those vertices
    # A simple upper bound is the size of the set itself, but we can do better
    
    # Let's use a recursive branch and bound with memoization-like pruning
    
    best_set = []
    
    # Greedy initial solution
    def greedy_independent_set(order):
        selected = []
        remaining = set(order)
        for v in order:
            if v not in remaining:
                continue
            # Check if v can be added
            can_add = True
            for u in selected:
                if u in adj[v]:
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                remaining.discard(v)
        return selected
    
    # Multiple greedy restarts with different orderings
    for restart in range(50):
        if time.time() > deadline - 100:
            break
        order = list(range(N))
        np.random.shuffle(order)
        sol = greedy_independent_set(order)
        if len(sol) > len(best_set):
            best_set = sol[:]
    
    # Local search / improvement
    def try_improve(current_set):
        current_set = set(current_set)
        improved = True
        while improved and time.time() < deadline - 100:
            improved = False
            for u in list(current_set):
                current_set.discard(u)
                # Find candidates that can be added
                candidates_to_add = []
                for v in range(N):
                    if v in current_set:
                        continue
                    # Check if v conflicts with any in current_set
                    can_add = True
                    for w in current_set:
                        if v in adj[w]:
                            can_add = False
                            break
                    if can_add:
                        candidates_to_add.append(v)
                if candidates_to_add:
                    # Try adding the best candidate (one that conflicts with fewest others)
                    best_v = None
                    min_conflicts = float('inf')
                    for v in candidates_to_add:
                        # Count how many of the remaining non-selected vertices this would conflict with
                        # Actually, just add a random one or the one with fewest adjacencies in the pool
                        # For simplicity, pick one at random
                        v = candidates_to_add[np.random.randint(len(candidates_to_add))]
                        current_set.add(v)
                        improved = True
                        break
                else:
                    # Couldn't add anything, put u back
                    current_set.add(u)
        return current_set
    
    improved_set = try_improve(set(best_set))
    if len(improved_set) > len(best_set):
        best_set = list(improved_set)
    
    # Branch and bound for maximum independent set
    # This is the key structural change
    
    # We'll use a recursive approach with upper bounds
    # Upper bound for a set S: use greedy coloring of the induced subgraph
    
    def greedy_color(vertices):
        """Greedy coloring of the subgraph induced by vertices. Returns number of colors."""
        vertices = list(vertices)
        color = {}
        num_colors = 0
        # Order vertices by degree in the induced subgraph (descending)
        def degree_in(v):
            return len(adj[v] & set(vertices))
        vertices.sort(key=degree_in, reverse=True)
        for v in vertices:
            # Find smallest color not used by neighbors
            used = set()
            for u in adj[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        return num_colors
    
    # Branch and bound
    # State: current independent set, candidate set (vertices not yet decided)
    # Upper bound: len(current) + greedy_color(candidate_set)
    
    best_size = len(best_set)
    best_solution = set(best_set)
    
    def branch_and_bound(current, candidates_set):
        nonlocal best_size, best_solution
        if time.time() > deadline - 50:
            return
        
        if len(current) + len(candidates_set) <= best_size:
            return
        
        if not candidates_set:
            if len(current) > best_size:
                best_size = len(current)
                best_solution = set(current)
            return
        
        # Upper bound using greedy coloring
        ub = len(current) + greedy_color(candidates_set)
        if ub <= best_size:
            return
        
        # Choose a vertex to branch on
        # Use the vertex with highest degree in the candidate subgraph
        def degree_in(v):
            return len(adj[v] & candidates_set)
        
        # Pick vertex with max degree
        v = max(candidates_set, key=degree_in)
        
        # Branch 1: include v
        new_current = current | {v}
        new_candidates = candidates_set - {v} - adj[v]
        branch_and_bound(new_current, new_candidates)
        
        # Branch 2: exclude v
        new_candidates = candidates_set - {v}
        branch_and_bound(current, new_candidates)
    
    # Initialize with a good candidate set
    # Start with all vertices, but use the greedy solution as a starting point for pruning
    # Actually, let's just run B&B on the full graph with the greedy solution as initial best
    
    # To make it more efficient, we can order the vertices and use a smarter approach
    # Let's try a different strategy: use the greedy solution to get a lower bound,
    # then use B&B with that lower bound
    
    # The B&B above may be too slow for N=5302. Let's use a simpler approach:
    # Iterated local search with simulated annealing-like moves
    
    # Actually, let's try a different structural approach:
    # Use a maximum independent set algorithm based on the complement graph
    # and use a library-like implementation of a known efficient algorithm
    
    # For now, let's do more greedy + local search iterations
    
    for restart in range(200):
        if time.time() > deadline - 50:
            break
        
        # Random restart with different perturbation
        order = list(range(N))
        np.random.shuffle(order)
        sol = set(greedy_independent_set(order))
        
        # Local search: try to add/remove/swap vertices
        for iteration in range(100):
            if time.time() > deadline - 50:
                break
            improved = False
            
            # Try adding a vertex
            addable = []
            for v in range(N):
                if v in sol:
                    continue
                can_add = True
                for u in sol:
                    if v in adj[u]:
                        can_add = False
                        break
                if can_add:
                    addable.append(v)
            
            if addable:
                v = addable[np.random.randint(len(addable))]
                sol.add(v)
                improved = True
            else:
                # Try swapping: remove one, add another
                if sol:
                    u = list(sol)[np.random.randint(len(sol))]
                    sol.discard(u)
                    addable2 = []
                    for v in range(N):
                        if v in sol:
                            continue
                        can_add = True
                        for w in sol:
                            if v in adj[w]:
                                can_add = False
                                break
                        if can_add:
                            addable2.append(v)
                    if addable2:
                        v = addable2[np.random.randint(len(addable2))]
                        sol.add(v)
                        improved = True
                    else:
                        sol.add(u)
            
            if not improved:
                break
        
        if len(sol) > len(best_set):
            best_set = list(sol)
    
    # Final check
    if not best_set:
        # Fallback to a small valid set
        return np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    points = candidates[best_set]
    return np.around(points).astype(np.int64)
