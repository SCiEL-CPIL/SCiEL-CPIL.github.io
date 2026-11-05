import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Target: Find a 5-uniform hypergraph on n vertices with minimal edges
    # that is NOT 2-colorable.
    # Known best is 51 edges.
    
    # We will search over n=9 to n=15.
    # For each n, we start with a random hypergraph and try to reduce edges.
    
    best_edges = None
    best_n = 0
    
    # Fast 2-colorability checker using bitmasks
    # n <= 26, so we can use Python ints as bitsets.
    # A coloring is a bitmask of n bits.
    # An edge is monochromatic if (edge_mask & coloring) == 0 or (edge_mask & coloring) == edge_mask.
    
    def is_2_colorable_fast(n, edge_masks):
        # Iterate all 2^n colorings
        # n is small (9-15), so 2^n is at most 32768.
        # For each coloring, check all edges.
        # Optimization: Precompute for each edge mask, the condition for being monochromatic.
        
        # Actually, for n=15, 32768 * E checks. If E=50, that's ~1.6M ops. Very fast in C/NumPy, but Python loop is slower.
        # Let's optimize the inner loop.
        
        # Alternative: Use a SAT-like backtracking?
        # For n<=12, brute force is fine.
        # For n=15, brute force might be ~0.1s per check. Too slow for many checks.
        
        # Let's stick to n=9 and n=10 primarily, where brute force is instant.
        # n=9: 512 colorings.
        # n=10: 1024 colorings.
        # n=11: 2048 colorings.
        
        # If we restrict to n=9, max edges is 126.
        # Can we get < 51 edges on n=9?
        # The complete graph on 9 vertices has 126 edges.
        # Is it possible to have a non-2-colorable 5-uniform hypergraph on 9 vertices with < 51 edges?
        # Random hypergraphs become non-2-colorable around certain densities.
        
        # Let's try n=9 first.
        
        for mask in range(1 << n):
            # Check if any edge is monochromatic
            for em in edge_masks:
                # em is an int bitmask of the edge vertices
                # If (mask & em) == 0: all 0s
                # If (mask & em) == em: all 1s
                if (mask & em) == 0 or (mask & em) == em:
                    break
            else:
                # No monochromatic edge found for this mask
                return True
        return False

    # Search strategy:
    # 1. Try n=9. Generate random hypergraphs with decreasing edge count.
    # 2. Use Simulated Annealing to minimize edges while keeping non-2-colorable.
    
    def run_sa(n, initial_edges, max_iter=5000, t_max=50):
        # initial_edges: list of 5-tuples
        # Convert to set for fast lookup
        edge_set = set(initial_edges)
        current_edges = list(edge_set)
        
        # Precompute masks for current edges? 
        # Masks depend only on vertex indices, which are fixed for n.
        # But the set of edges changes.
        
        # To check 2-colorability, we need the list of edge masks.
        
        def get_masks(edges):
            return [sum(1 << v for v in e) for e in edges]
            
        current_masks = get_masks(current_edges)
        
        # Check if current is valid (non-2-colorable)
        if is_2_colorable_fast(n, current_masks):
            return None # Invalid start
            
        best_edges = current_edges[:]
        best_count = len(current_edges)
        
        T = 1.0
        cooling = 0.999
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            if T < 0.001:
                break
                
            # Random move: Add an edge, Remove an edge, or Swap
            move_type = random.random()
            
            new_edges = current_edges[:]
            
            if move_type < 0.4:
                # Remove a random edge
                if len(new_edges) > 1:
                    idx = random.randint(0, len(new_edges) - 1)
                    new_edges.pop(idx)
            elif move_type < 0.8:
                # Add a random edge
                # Generate random 5-subset
                new_edge = tuple(sorted(random.sample(range(n), 5)))
                if new_edge not in new_edges:
                    new_edges.append(new_edge)
            else:
                # Swap: Remove one, add one
                if len(new_edges) > 1:
                    idx = random.randint(0, len(new_edges) - 1)
                    removed = new_edges.pop(idx)
                    new_edge = tuple(sorted(random.sample(range(n), 5)))
                    if new_edge not in new_edges:
                        new_edges.append(new_edge)
                    else:
                        new_edges.append(removed) # Revert swap if collision
            
            # Check validity
            new_masks = get_masks(new_edges)
            
            # If it became 2-colorable, it's invalid.
            if is_2_colorable_fast(n, new_masks):
                # Reject
                continue
                
            # It is valid (non-2-colorable).
            # Energy is -len(new_edges) (we want to minimize edges, so lower is better)
            # Actually, let's define cost = len(new_edges). We want to minimize cost.
            old_cost = len(current_edges)
            new_cost = len(new_edges)
            
            delta = new_cost - old_cost
            
            if delta < 0:
                # Improvement
                current_edges = new_edges
                current_masks = new_masks
                if new_cost < best_count:
                    best_count = new_cost
                    best_edges = new_edges[:]
            else:
                # Worse or equal
                # Accept with probability exp(-delta / T)
                if delta == 0:
                    current_edges = new_edges
                    current_masks = new_masks
                else:
                    if random.random() < pow(2.71828, -delta / T):
                        current_edges = new_edges
                        current_masks = new_masks
                        
            T *= cooling
            
        return best_edges

    # Try different n values
    # n=9 is the smallest possible for non-2-colorable 5-uniform?
    # C(8,5) = 56. Complete graph on 8 is 2-colorable (4-4 split).
    # So n must be at least 9.
    
    for n in [9, 10, 11]:
        if time.time() > deadline - 20:
            break
            
        # Start with a random hypergraph that is likely non-2-colorable
        # Dense random hypergraphs are non-2-colorable.
        
        # For n=9, C(9,5)=126.
        # Let's start with ~60% of edges.
        
        all_possible = list(combinations(range(n), 5))
        
        # Try a few random starts
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            # Random subset
            size = int(len(all_possible) * 0.6)
            initial = random.sample(all_possible, size)
            
            # Check if non-2-colorable
            masks = [sum(1 << v for v in e) for e in initial]
            if is_2_colorable_fast(n, masks):
                continue
                
            # Run SA
            result = run_sa(n, initial, max_iter=2000, t_max=20)
            
            if result:
                if best_edges is None or len(result) < len(best_edges):
                    best_edges = result
                    best_n = n
                    
            # If we found a very small one, we can stop early
            if best_edges and len(best_edges) <= 51:
                break
                
    if best_edges is None:
        # Fallback: Return complete graph on 9 vertices (126 edges)
        # This is guaranteed non-2-colorable.
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n

    # Final verification
    # Convert to list of lists
    final_edges = [list(e) for e in best_edges]
    
    # Ensure vertices are 0..n-1
    # Ensure distinct
    seen = set()
    unique_edges = []
    for e in final_edges:
        t = tuple(sorted(e))
        if t not in seen:
            seen.add(t)
            unique_edges.append(e)
            
    # Check non-2-colorable one last time to be safe
    if is_2_colorable_fast(best_n, [sum(1 << v for v in e) for e in unique_edges]):
        # It is 2-colorable! This should not happen if SA worked correctly, 
        # but if it does, we must return a valid one.
        # Fallback to complete graph on 9
        n = 9
        unique_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': unique_edges}

# Run construct to test
if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
