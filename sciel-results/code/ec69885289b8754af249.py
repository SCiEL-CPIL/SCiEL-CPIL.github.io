import time
import random
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_k4(adj, n):
    """Check if graph contains a K4."""
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w in adj[v]:
                    # u, v, w form a triangle
                    # Check for 4th vertex
                    common_vw = adj[v].intersection(adj[w])
                    for x in common_vw:
                        if x in adj[u] and x != v and x != w:
                            return True
    return False

def has_independent_set(adj, n, k):
    """Check if graph contains an independent set of size k."""
    if n > 30 and k > 5:
        pass
        
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    return False
        return True

    for comb in combinations(range(n), k):
        if is_independent(comb):
            return True
    return False

def get_independent_sets_size_k(adj, n, k):
    """Return list of independent sets of size k."""
    if n < k:
        return []
    
    sets = []
    # Simple backtracking to find independent sets of size k
    # This is faster than combinations for sparse graphs or small k
    
    def backtrack(start, current_set, remaining):
        if len(current_set) == k:
            sets.append(tuple(current_set))
            return
        if remaining < k - len(current_set):
            return
        
        for v in range(start, n):
            # Check if v can be added to current_set
            # v must not be connected to any in current_set
            valid = True
            for u in current_set:
                if v in adj[u]:
                    valid = False
                    break
            if valid:
                current_set.append(v)
                backtrack(v + 1, current_set, remaining - 1)
                current_set.pop()
                
    backtrack(0, [], n)
    return sets

def try_add_vertex(adj, n, deadline):
    """
    Tries to find a valid set of neighbors N for a new vertex v_new.
    Returns the set N if found, else None.
    Constraints on N:
    1. N is triangle-free in G.
    2. Non-neighbors (V \ N) do not contain an independent set of size 5.
       Equivalent: N intersects every independent set of size 5 in G.
    """
    # Step 1: Find all independent sets of size 5 in G
    ind_sets_5 = get_independent_sets_size_k(adj, n, 5)
    
    if not ind_sets_5:
        # No independent sets of size 5, so constraint 2 is always satisfied.
        # We just need a triangle-free set N.
        # A maximal triangle-free set is good.
        # Greedy: pick vertices that don't form triangles.
        # Since we want to maximize n, the specific N doesn't matter much as long as it's valid.
        # Let's pick a random maximal triangle-free set.
        N = set()
        remaining = set(range(n))
        random.shuffle(list(remaining))
        for v in remaining:
            # Check if adding v to N creates a triangle
            # v creates a triangle if there is an edge between two neighbors of v in N
            # i.e., if adj[v] intersection N contains an edge
            neigh_in_N = adj[v].intersection(N)
            triangle_formed = False
            for u in neigh_in_N:
                for w in neigh_in_N:
                    if u < w and w in adj[u]:
                        triangle_formed = True
                        break
                if triangle_formed:
                    break
            if not triangle_formed:
                N.add(v)
        return N

    # Step 2: Backtracking search for N
    # N must hit all sets in ind_sets_5
    # N must be triangle-free
    
    # Heuristic: Sort vertices by "utility" (how many ind sets they hit)
    # But utility changes as we add to N.
    
    # We will use a randomized DFS with backtracking.
    # State: current N, list of active ind sets (those not hit by N)
    
    active_sets = [set(s) for s in ind_sets_5]
    
    # Pre-compute for each vertex, which sets it hits?
    # Or just filter active sets.
    
    def is_triangle_free_subset(N_subset):
        # Check if N_subset is triangle-free in G
        # Optimization: only check new vertex against existing
        # But here we check from scratch or maintain state.
        # Let's just check the whole set for simplicity in the inner loop, 
        # or better, pass the last added vertex and check only pairs involving it.
        pass

    # We'll implement a recursive function
    # To avoid deep recursion limits or slow performance, we'll limit depth or use iterative.
    # n is up to 35. Depth 35 is fine.
    
    # Randomize vertex order for diversity
    vertices = list(range(n))
    
    def backtrack(idx, N, active_sets_copy):
        if not active_sets_copy:
            return N.copy()
        
        if time.time() > deadline:
            return None
            
        # Heuristic: Pick a set from active_sets_copy, and try to pick a vertex from it
        # This ensures we make progress on hitting sets.
        # Pick a random active set
        random.shuffle(active_sets_copy)
        target_set = active_sets_copy[0]
        
        # Try each vertex in target_set
        # Order: Random or by degree in N?
        candidates = list(target_set)
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if adding v keeps N triangle-free
            # v must not form a triangle with any edge in N
            # i.e., for any u, w in N, if u-w is edge, v cannot be connected to both u and w.
            # Actually, v forms a triangle with u, w in N if u-w is edge AND v-u is edge AND v-w is edge.
            
            valid = True
            # Optimization: Pre-check
            # If v is connected to two neighbors in N that are connected to each other, invalid.
            
            # Iterate over neighbors of v in N
            neigh_v_in_N = adj[v].intersection(N)
            for u in neigh_v_in_N:
                # Check if u is connected to any other vertex in neigh_v_in_N
                # If yes, then u, w, v form a triangle.
                # We need to check if there exists w in neigh_v_in_N such that w != u and w in adj[u].
                for w in neigh_v_in_N:
                    if w != u and w in adj[u]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Add v to N
            new_N = N | {v}
            
            # Update active sets: remove any set that contains v
            new_active_sets = []
            for s in active_sets_copy:
                if v not in s:
                    new_active_sets.append(s)
            
            # Recurse
            result = backtrack(idx + 1, new_N, new_active_sets)
            if result is not None:
                return result
                
        return None

    # Start with empty N
    # To improve success rate, we can try multiple random restarts or different orderings
    # But the "pick from target set" strategy is deterministic given the shuffle.
    
    # Let's try a few times
    for _ in range(20):
        res = backtrack(0, set(), [set(s) for s in ind_sets_5])
        if res is not None:
            return res
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with Paley 17
    g17 = generate_paley(17)
    if not g17:
        return {'n': 1, 'edges': []}
        
    adj = build_adjacency(17, g17['edges'])
    n = 17
    
    # Verify initial graph
    if has_k4(adj, n) or has_independent_set(adj, n, 6):
        # Fallback to smaller if P17 is invalid (it shouldn't be)
        return {'n': 1, 'edges': []}
        
    # Main loop: try to add vertices
    while n < 35 and time.time() < deadline:
        added = False
        # Try to add a vertex
        # We might need to "prepare" the graph by swapping edges if we get stuck
        # For now, just try to add.
        
        new_N = try_add_vertex(adj, n, deadline)
        
        if new_N is not None:
            # Add new vertex n with neighbors new_N
            new_edges = []
            for u in new_N:
                new_edges.append([n, u])
            
            # Update adj
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            n += 1
            added = True
            
        if not added:
            # Try to modify the graph to make it easier to add a vertex
            # Perform a few random valid edge swaps
            swaps_made = 0
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Pick a random edge (u, v) and a random non-edge (w, x)
                # Such that swapping maintains K4-free and alpha <= 5
                # This is complex. Simpler: just pick random u, v, w, x.
                # Actually, let's just break and return current best if we can't add.
                # Or try a different randomization in try_add_vertex.
                pass
            
            # If we can't add, we might be stuck.
            # Let's try a few more times with different random seeds implicitly by the shuffle
            # If still stuck, break.
            # To avoid infinite loop, we break.
            break
            
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

# Quick sanity check for has_independent_set and get_independent_sets_size_k
# The backtrack in get_independent_sets_size_k might be slow if n is large and k is 5.
# C(35, 5) is ~ 324,632.
# The backtracking prunes based on adjacency.
# If the graph is dense, it prunes fast.
# If the graph is sparse, there are many independent sets.
# But we stop when we find them? No, we need ALL of them for the hitting set constraint.
# If there are many, the list `ind_sets_5` will be large.
# Then `try_add_vertex` has to hit all of them.
# If the number of independent sets is huge, the backtracking in `try_add_vertex` might be slow.

# Optimization:
# Instead of enumerating all independent sets, we can use a SAT-like approach or just rely on the fact that
# for K4-free graphs, the structure is constrained.
# However, for n=20-30, it should be manageable.

# Let's refine `try_add_vertex` to handle the case where `ind_sets_5` is very large.
# If len(ind_sets_5) > 1000, we might timeout.
# We can add a check.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    g17 = generate_paley(17)
    if not g17:
        return {'n': 1, 'edges': []}
        
    adj = build_adjacency(17, g17['edges'])
    n = 17
    
    if has_k4(adj, n) or has_independent_set(adj, n, 6):
        return {'n': 1, 'edges': []}
        
    while n < 35 and time.time() < deadline:
        # Check time
        if time.time() > deadline - 10:
            break
            
        new_N = try_add_vertex(adj, n, deadline)
        
        if new_N is not None:
            new_edges = []
            for u in new_N:
                new_edges.append([n, u])
            
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            n += 1
        else:
            # Stuck. Try to perturb.
            # Randomly swap an edge with a non-edge, if valid.
            perturbed = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Pick u, v edge
                # Pick w, x non-edge
                # Try swap
                # This is risky and slow.
                # Just break.
                break
            break

    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
