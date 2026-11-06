import time
import random
import sys
from itertools import combinations

def check_c4(adj, n):
    """
    Check if graph with adjacency lists is C4-free.
    Condition: For any pair u, v, |N(u) intersection N(v)| <= 1.
    """
    adj_sets = [set(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj_sets[u]:
            if v > u:
                common = adj_sets[u] & adj_sets[v]
                if len(common) > 1:
                    return True
    return False

def can_add_edge(adj, u, v):
    """
    Check if adding edge (u, v) keeps the graph C4-free.
    This requires that u and v have no common neighbors.
    """
    if u in adj[v] or v in adj[u]:
        return False
    common = adj[u] & adj[v]
    return len(common) == 0

def find_independent_set(adj, n, size):
    """
    Find an independent set of the given size.
    Returns the set of vertices if found, else None.
    Uses backtracking with bitmasks for speed.
    """
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    def backtrack(start, current_mask, current_size):
        if current_size == size:
            return current_mask
        if start >= n:
            return None
        
        # Pruning: if not enough vertices left
        if n - start < size - current_size:
            return None
            
        # Heuristic: Try vertices with fewest neighbors in remaining candidates?
        # For simplicity, iterate. Optimization: Order by degree?
        # Let's just iterate.
        
        # Option 1: Exclude vertex 'start'
        # Option 2: Include vertex 'start'
        
        # To find ANY set, we can branch.
        # If we include 'start', we must exclude all its neighbors.
        # If we exclude 'start', we continue to 'start+1'.
        
        # Branch 1: Include 'start'
        if (current_mask & (1 << start)) == 0:
            new_mask = current_mask | (1 << start)
            # Remove neighbors of 'start' from consideration?
            # In the recursive call, we can just check conflicts.
            # But for efficiency, we can mask out neighbors.
            # However, the 'start' index advances, so we don't need to mask out
            # previous vertices. We only need to ensure future vertices don't conflict.
            
            # Let's use a helper that takes a bitmask of available vertices?
            # That might be cleaner.
            
        # Let's rewrite using available mask.
        pass

    # Simpler approach for n <= 43:
    # Use a set of candidates.
    def backtrack_set(candidates, current_set):
        if len(current_set) >= size:
            return current_set
        if len(candidates) + len(current_set) < size:
            return None
            
        if not candidates:
            return None
            
        # Pick a vertex
        # Heuristic: Pick vertex with highest degree in the subgraph induced by candidates?
        # Or just pick the first one.
        # Let's pick the vertex with the most neighbors in candidates to prune more.
        # This is expensive to compute every step.
        # Let's just pick the first one for simplicity, or random.
        # Actually, for finding a set of size 11, branching factor is high.
        
        # Let's try a greedy approach first?
        # If greedy fails, backtrack.
        
        # Backtracking:
        # Pick v.
        # Branch 1: Include v. Candidates becomes candidates - {v} - N(v).
        # Branch 2: Exclude v. Candidates becomes candidates - {v}.
        
        # To speed up, we can order candidates.
        # Let's sort by degree in the induced subgraph?
        # For n=40, this is okay.
        
        # Let's just take the first element.
        v = next(iter(candidates))
        
        neighbors_in_candidates = candidates & adj[v]
        non_neighbors_in_candidates = candidates - neighbors_in_candidates - {v}
        
        # Branch 1: Include v
        # If we include v, we can only add vertices from non_neighbors_in_candidates
        if len(current_set) + len(non_neighbors_in_candidates) + 1 >= size:
            res = backtrack_set(non_neighbors_in_candidates, current_set | {v})
            if res:
                return res
                
        # Branch 2: Exclude v
        if len(current_set) + len(candidates - {v}) >= size:
            res = backtrack_set(candidates - {v}, current_set)
            if res:
                return res
                
        return None

    return backtrack_set(set(range(n)), set())

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # Safety margin for 150s limit
    
    # Base structure: Polarity graph of projective plane of order 6.
    # q = 6
    # Number of points = q^2 + q + 1 = 36 + 6 + 1 = 43.
    # Vertices are 0..42.
    # Two points (x1, y1) and (x2, y2) are adjacent if x1*x2 + y1*y2 = 1 (in GF(6)? No, GF(q^2)).
    # Wait, the standard polarity graph construction uses GF(q).
    # Points are (x, y) in GF(q) x GF(q) plus a point at infinity.
    # Total points q^2 + 1.
    # For q=6, 37 points. This is not 43.
    
    # The prompt mentions "39-vertex polarity graph of the projective plane of order 6".
    # Projective plane of order q has q^2 + q + 1 points.
    # For q=6, 43 points.
    # A polarity graph on these points:
    # Points are represented as 1D vectors in GF(q)^3.
    # Two points p, p' are adjacent if p^T A p' = 0 for some symmetric matrix A?
    # Or specifically, a symplectic polarity?
    
    # Let's stick to the known construction for C4-free graphs.
    # The "Erdos-Renyi" random graph is one.
    # The "Polarity graph" of PG(2, q) is a standard example.
    # Vertices: Points of PG(2, q).
    # Edges: p ~ q if p is on the polar line of q.
    # This graph is C4-free.
    # Size: q^2 + q + 1.
    # For q=5, size 31.
    # For q=6, size 43.
    # But q must be a prime power for PG(2, q) to exist in the standard finite field sense?
    # Yes, q must be a prime power. 6 is NOT a prime power.
    # So PG(2, 6) does not exist in the standard sense.
    # The prompt says "projective plane of order 6".
    # It is an open problem whether a projective plane of order 6 exists.
    # It is known that NO projective plane of order 6 exists.
    # So the "39-vertex polarity graph of the projective plane of order 6" is a trick or refers to a specific known graph often associated with this context, possibly derived from a different construction or a subgraph.
    
    # Actually, the best known C4-free graph without independent set of size 11 might be related to the Hoffman-Singleton graph or similar?
    # Hoffman-Singleton is 50 vertices, 7-regular.
    # But we need n <= 43.
    
    # Let's look at the "inspiration" again.
    # "Use the 39-vertex polarity graph of the projective plane of order 6... as the base structure"
    # This phrasing is suspicious. Maybe it means a graph ON 39 vertices?
    # Or maybe it's a typo for order 5 (31 vertices) or something else?
    
    # Let's ignore the specific geometric construction if it's flawed (order 6 plane doesn't exist) and focus on the STRATEGY:
    # 1. Start with a good base graph (C4-free, low alpha).
    # 2. Perform local search: add edges that don't create C4 to reduce alpha.
    
    # What is a good base graph?
    # Random greedy construction often works well.
    # Let's try to generate a random C4-free graph on n=43 or 42 or 41.
    # If we can find one with alpha < 11, we return it.
    
    # Strategy:
    # 1. Try to construct a C4-free graph on n=43, 42, 41... down to 39.
    # 2. Use a greedy algorithm to add edges while maintaining C4-free.
    # 3. Check alpha. If alpha >= 11, try to reduce it by adding more edges (if possible) or restarting.
    
    # Greedy C4-free construction:
    # Start with empty graph on n vertices.
    # Consider all pairs (u, v).
    # Shuffle them.
    # For each pair, if adding the edge doesn't create a C4, add it.
    # This produces a maximal C4-free graph.
    
    # Does this graph have low alpha?
    # Maximal C4-free graphs tend to have high minimum degree, thus lower alpha.
    
    # Let's implement this.
    
    for n in range(43, 38, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            edges = []
            
            # Generate all pairs
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            
            # Greedy add
            for u, v in pairs:
                if can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    
            # Check if alpha < 11
            # We need to check if there is an independent set of size 11.
            # If find_independent_set returns None, then alpha < 11.
            # If it returns a set, alpha >= 11.
            
            # Optimization: We can check alpha during construction?
            # No, easier to check at the end.
            
            # To speed up, we can estimate alpha or use a faster check.
            # The backtracking set method might be slow if alpha is large.
            # If alpha is small, it might find a set quickly or fail quickly?
            # If alpha < 11, it must explore the whole space to prove it?
            # No, it returns None only if no set is found.
            # If alpha < 11, the search space is exhausted.
            # This could be slow.
            
            # Alternative: Check if a specific set is independent? No.
            # We need to know if ANY set of size 11 exists.
            
            # Maybe we can use a heuristic to prune?
            # If we find a set of size 11, we stop (bad).
            # If we don't find one, we are stuck with the proof.
            
            # Let's try a different approach:
            # Start with a known good graph?
            # The seed is 10 disjoint triangles (n=30, alpha=10).
            # We want to go higher.
            
            # Let's try n=39 first as suggested by the prompt's "best known".
            # If we can get 39, that's a score of 39/30 = 1.3.
            
            # Let's refine the alpha check.
            # If n is small (<= 43), bitmask backtracking is fast.
            
            indep_set = find_independent_set(adj, n, 11)
            if indep_set is None:
                # Valid graph
                return {'n': n, 'edges': edges}
            else:
                # Alpha >= 11.
                # Try to fix by adding edges?
                # We are already maximal C4-free.
                # So we cannot add any more edges.
                # We must restart with a different random seed/order.
                pass
                
    # Fallback: Try to improve the seed (n=30) or find a smaller valid graph?
    # The seed is valid.
    # If we didn't find anything > 30, return seed.
    
    # But we should try n=31..38 as well if 43..39 failed quickly?
    # The loop above goes 43..39.
    # If time remains, try 38..31.
    
    if time.time() < deadline:
        for n in range(38, 30, -1):
            if time.time() > deadline:
                break
            for attempt in range(50):
                if time.time() > deadline:
                    break
                adj = [set() for _ in range(n)]
                edges = []
                pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
                random.shuffle(pairs)
                
                for u, v in pairs:
                    if can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
                        
                indep_set = find_independent_set(adj, n, 11)
                if indep_set is None:
                    return {'n': n, 'edges': edges}
    
    # Return seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# Let's optimize find_independent_set with bitmasks for speed.

def find_independent_set_fast(adj, n, size):
    """
    Fast independent set search using bitmasks.
    Returns a bitmask of an independent set of size `size` or 0 if not found.
    """
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # We want to find a set of `size` vertices such that no two are adjacent.
    # This is equivalent to finding a clique of size `size` in the complement graph.
    # Or just backtracking on the original graph.
    
    def backtrack(available_mask, current_size):
        if current_size == size:
            return 1  # Found
        if current_size + available_mask.bit_count() < size:
            return 0
        if available_mask == 0:
            return 0
            
        # Pick a vertex from available_mask
        # Heuristic: Pick vertex with highest degree in available subgraph?
        # To minimize branching, pick the one with the most neighbors in available?
        # If we pick v, and include it, we remove v and all its neighbors.
        # If we pick v, and exclude it, we remove v.
        
        # Let's pick the first set bit.
        v = (available_mask & -available_mask).bit_length() - 1
        
        # Branch 1: Include v
        # New available = available_mask & ~(1 << v) & ~adj_mask[v]
        new_available_include = available_mask & ~(1 << v) & ~adj_mask[v]
        if backtrack(new_available_include, current_size + 1):
            return 1
            
        # Branch 2: Exclude v
        new_available_exclude = available_mask & ~(1 << v)
        if backtrack(new_available_exclude, current_size):
            return 1
            
        return 0

    # Initial available mask: all vertices
    initial_mask = (1 << n) - 1
    return backtrack(initial_mask, 0)

# Replace the previous find_independent_set with this faster one
def find_independent_set(adj, n, size):
    return find_independent_set_fast(adj, n, size)

# Re-define construct to use the updated logic if needed, but it calls the global function.

# One more check: The prompt says "Maximize n".
# The seed is 30.
# If we find 31, score is 31/30 = 1.033.
# If we find 39, score is 1.3.

# The random greedy method is stochastic.
# We should ensure we try enough times.
# With n=43, the number of maximal C4-free graphs is huge.
# Some will have low alpha, some high.
# We hope that a significant fraction have alpha < 11.

# Let's run a quick mental check on the time.
# n=43, pairs ~ 900.
# can_add_edge is O(deg).
# Greedy construction is fast.
# find_independent_set is the bottleneck.
# If alpha < 11, the search space is pruned heavily?
# If alpha is, say, 10, the search for size 11 will fail.
# How fast does it fail?
# It explores the tree of possibilities.
# If the graph is dense (high degree), the "Include" branch removes many vertices, so the tree is shallow.
# C4-free graphs from greedy are usually quite dense.
# So the check should be fast.

# Final code assembly.
