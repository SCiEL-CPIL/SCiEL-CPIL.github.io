import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 45.0  # seconds, safety margin
    
    n = 15
    
    def get_edges_set():
        return set()

    def is_valid(edges_list):
        """
        Check if the graph represented by edges_list is valid.
        Violation: Simple path s-u-v-t where d(s) == d(t).
        """
        # Build adjacency and degrees
        adj = {i: set() for i in range(n)}
        degrees = [0] * n
        
        for u, v in edges_list:
            if u == v: continue
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        # Check all pairs s, t at distance 3
        # A path s-u-v-t exists if:
        # u is neighbor of s
        # v is neighbor of u, v != s
        # t is neighbor of v, t != u, t != s
        # AND d(s) == d(t)
        
        # Optimization: Precompute neighbors
        # For each s, t:
        #   if d(s) != d(t): continue
        #   if there exists u in N(s), v in N(u)\{s} such that t in N(v)\{u, s}:
        #       return False
         
        # This is O(N^4) or O(N * E^2) roughly. N=15, so 15^4 = 50625 checks. Very fast.
        
        for s in range(n):
            for t in range(n):
                if s == t: continue
                if degrees[s] != degrees[t]: continue
                
                # Check if path s-u-v-t exists
                found_path = False
                for u in adj[s]:
                    if u == t: continue # path length 1, not 3. But wait, s-u-v-t requires u!=t, v!=s, t!=u, v!=t?
                    # Simple path: vertices distinct. s, u, v, t distinct.
                    for v in adj[u]:
                        if v == s: continue
                        if v == t: continue
                        if v == u: continue # no loops
                        if t in adj[v]:
                            if u != t and s != v: # distinctness already checked mostly
                                found_path = True
                                break
                    if found_path:
                        break
                if found_path:
                    return False
        return True

    def get_degree_counts(edges_list):
        degrees = [0] * n
        for u, v in edges_list:
            degrees[u] += 1
            degrees[v] += 1
        return degrees

    best_edges = None
    best_score = -1

    # Strategy: Local Search with swaps
    # We try multiple random starts or perturbations.
    
    def local_search(initial_edges):
        edges = set(tuple(sorted(e)) for e in initial_edges)
        # Ensure validity first
        if not is_valid(list(edges)):
             # Try to fix by removing edges
             valid = False
             attempts = 0
             while not is_valid(list(edges)) and attempts < 1000:
                 # Find a violating path
                 # Simplified: just remove a random edge until valid
                 # This is slow but works for small N
                 e_to_remove = random.choice(list(edges))
                 edges.remove(e_to_remove)
                 attempts += 1
             if not is_valid(list(edges)):
                 return None, 0
        
        # Now edges is valid (or we failed)
        if not is_valid(list(edges)):
             return None, 0

        current_score = len(edges)
        
        # Try to add edges
        all_possible = set()
        for i in range(n):
            for j in range(i+1, n):
                all_possible.add((i, j))
                
        # Phase 1: Greedy addition
        improved = True
        while improved:
            improved = False
            # Shuffle missing edges
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            
            for e in missing:
                # Check if adding e makes it invalid
                new_edges = edges | {e}
                if is_valid(list(new_edges)):
                    edges = new_edges
                    current_score = len(edges)
                    improved = True
                    break
            
        # Phase 2: Swap moves to escape local optima
        swap_improved = True
        max_swap_iterations = 50
        swap_iter = 0
        while swap_improved and swap_iter < max_swap_iterations:
            swap_improved = False
            swap_iter += 1
            
            # Get current edges and missing edges
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            
            # Try swaps: remove one edge, add one missing edge
            existing = list(edges)
            random.shuffle(existing)
            
            for e_remove in existing:
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                edges_without = edges - {e_remove}
                
                for e_add in missing:
                    if e_add == e_remove:
                        continue
                    new_edges = edges_without | {e_add}
                    if is_valid(list(new_edges)):
                        edges = new_edges
                        current_score = len(edges)
                        swap_improved = True
                        break
                if swap_improved:
                    break
            
            # After a swap, try to add more edges greedily
            if swap_improved:
                add_improved = True
                while add_improved:
                    add_improved = False
                    missing = [e for e in all_possible if e not in edges]
                    random.shuffle(missing)
                    for e in missing:
                        new_edges = edges | {e}
                        if is_valid(list(new_edges)):
                            edges = new_edges
                            current_score = len(edges)
                            add_improved = True
                            break
        
        return list(edges), current_score

    # Run several iterations
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Generate a random initial graph
        # Start with a random subset of edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Try different densities, biased toward higher densities
        # Higher density starts are more likely to find dense valid graphs
        if i < num_restarts // 2:
            density = 0.7 + 0.3 * random.random()
        else:
            density = 0.5 + 0.5 * random.random()
        
        num_initial = int(len(all_edges) * density)
        initial_edges = random.sample(all_edges, num_initial)
        
        res_edges, score = local_search(initial_edges)
        
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges
            
        # If we found a very high score, we can stop early?
        # Max is 105.
        if best_score > 95:
             break

    if best_edges is None:
        # Fallback: Empty graph is valid
        return []
        
    # Convert to required format: list of lists [u, v]
    # Ensure all values are Python int (not numpy int or bool)
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

if __name__ == "__main__":
    # Just in case, but task says construct() is the entry
    pass
