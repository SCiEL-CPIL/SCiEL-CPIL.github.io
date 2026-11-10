"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 58.0  # safety margin within 60s
    
    # Strategy: We need to maximize edges in a graph on 15 vertices
    # such that no simple path of length 3 (s-u-v-t) has deg(s) == deg(t).
    
    # Key insight: If we can make all degrees distinct, the condition is automatically satisfied.
    # Max degree is 14. Min degree is 0. We have 15 vertices, so we can assign degrees 0,1,2,...,14.
    # But can we realize a graph with degrees 0,1,2,...,14?
    # By Erdős–Gallai, a sequence is graphical iff it satisfies the conditions.
    # The sequence (0, 1, 2, ..., 14) has sum = 105, which is odd. So it's NOT graphical.
    # So we can't have all degrees distinct.
    
    # Next best: minimize the number of pairs of vertices with equal degrees.
    # If two vertices have equal degree, they cannot be endpoints of a 3-edge path.
    # That is, if deg(s) == deg(t), then there should be no path s-u-v-t (i.e., distance 3 between s and t).
    
    # Alternative: Use a near-regular graph but ensure that vertices with same degree are not at distance 3.
    # Or, use a graph where the degree sequence has very few repeated degrees, and those repeated-degree
    # vertices are placed such that they're never at distance 3 from each other.
    
    # Let's think about a different approach:
    # Complete graph K_15 has 105 edges, all degrees 14. Then ANY 3-path has endpoints of degree 14 == 14. Invalid.
    # So we need to remove edges to break this.
    
    # Let's try: Start with K_15 and remove a set of edges so that the degree sequence has
    # the property that for any two vertices with equal degree, they are NOT at distance 3.
    
    # Better idea: Use a graph where most vertices have distinct degrees.
    # Since sum of degrees must be even, and we have 15 vertices, we need an even number of vertices
    # with "paired" degrees (or one vertex with a degree that makes the total even).
    
    # Actually, let's just do a smart search:
    # 1. Start with a high-edge-count base (like K_15 minus some edges)
    # 2. Check validity
    # 3. Use local search to add/remove edges to maximize count while maintaining validity
    
    # First, let's understand the constraint better:
    # For edge (u,v), for all s in N(u)\{v} and t in N(v)\{u} with s != t: deg[s] != deg[t]
    
    # This is a strong constraint. Let's try to build a graph where:
    # - Degrees are as "spread out" as possible
    # - Vertices with the same degree are far apart (not at distance 3)
    
    # Let's try a construction:
    # Partition vertices into groups by degree. Within each group, ensure no two vertices are at distance 3.
    
    # Actually, let me just do a randomized local search from a good starting point.
    
    def get_degrees(edge_set, n):
        deg = [0] * n
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edge_set, n):
        deg = get_degrees(edge_set, n)
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Start with K_15 minus a random set of edges to get a valid starting point
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    random.seed(42)
    
    best_edges = None
    best_count = 0
    
    # Try multiple random starts
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Start with K_15 and randomly remove edges until valid
        current = set(all_edges)
        # Remove edges randomly until valid or too few
        edges_list = list(current)
        random.shuffle(edges_list)
        
        # Try removing a random subset
        removed = set()
        for e in edges_list:
            if time.time() > deadline:
                break
            if is_valid(current, n):
                break
            current.discard(e)
            removed.add(e)
        
        if not is_valid(current, n):
            continue
        
        # Local search: try adding back removed edges
        for e in removed:
            if time.time() > deadline:
                break
            test = current | {e}
            if is_valid(test, n):
                current.add(e)
        
        # Also try: for each non-edge, see if we can add it
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                e = (i, j)
                if e in current:
                    continue
                test = current | {e}
                if is_valid(test, n):
                    current.add(e)
        
        count = len(current)
        if count > best_count:
            best_count = count
            best_edges = set(current)
    
    # Additional local search on the best found
    if best_edges is not None:
        current = set(best_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    e = (i, j)
                    if e in current:
                        continue
                    test = current | {e}
                    if is_valid(test, n):
                        current.add(e)
                        improved = True
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current
    
    # Convert to list format
    if best_edges is None:
        # Fallback: empty graph
        best_edges = set()
    
    result = [list(e) for e in best_edges]
    result.sort()
    
    # Ensure types are correct: integers must be Python int
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
# EVOLVE-BLOCK-END
