import random
import time
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    d = 7
    n = 1 << d
    total_edges = n * d  # 448
    
    # Generate all edges
    edges = []
    edge_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_set.add((x, y))
    
    # For each edge, precompute which squares contain it
    # A square is defined by two distinct dimensions i, j and a base vertex with bits i,j = 0
    # Edge (x, y) where y = x ^ (1<<i) is in squares with dimension i and any other dimension j != i
    # The square has vertices: x, x^(1<<j), y, y^(1<<j) where j != i
    # We need x to have bit j = 0 for the square to be "canonical" but actually any 4-cycle works
    
    # Better: for each edge (x, x^(1<<i)), the squares containing it are those where
    # we pick another dimension j != i, and the square is {x, x^(1<<j), x^(1<<i), x^(1<<i)^(1<<j)}
    # But we need to ensure this is a valid square (all 4 vertices distinct, which they are if i != j)
    
    # Precompute: for each edge, list of squares (as frozensets of 4 edges) that contain it
    # Actually, let's precompute for each pair of dimensions (i,j) with i<j, 
    # and for each "base" where bits i and j are 0, the 4 edges of that square.
    
    # Square indexed by (i, j, base) where i < j, base has bits i and j set to 0
    # Vertices: base, base^(1<<i), base^(1<<j), base^(1<<i)^(1<<j)
    # Edges: (base, base^(1<<i)), (base, base^(1<<j)), 
    #         (base^(1<<i), base^(1<<i)^(1<<j)), (base^(1<<j), base^(1<<j)^(1<<i))
    
    # Let's precompute all squares and for each edge, which squares it's in
    edge_to_squares = {e: [] for e in edges}
    squares = []  # list of tuples of 4 edges
    
    for i in range(d):
        for j in range(i+1, d):
            # For each base with bits i and j = 0
            mask_i = 1 << i
            mask_j = 1 << j
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_i ^ mask_j
                    e0 = (min(v0, v1), max(v0, v1))
                    e1 = (min(v0, v2), max(v0, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v3), max(v2, v3))
                    sq = (e0, e1, e2, e3)
                    squares.append(sq)
                    for e in sq:
                        edge_to_squares[e].append(len(squares) - 1)
    
    def is_valid(chosen_set):
        """Check if chosen_set has no C4"""
        for sq in squares:
            if all(e in chosen_set for e in sq):
                return False
        return True
    
    def count_c4s(chosen_set):
        """Count number of C4s in chosen_set"""
        count = 0
        for sq in squares:
            if all(e in chosen_set for e in sq):
                count += 1
        return count
    
    # Local search helpers
    def try_add_edge(chosen_set, edge):
        """Try to add edge, return True if valid after adding"""
        # Check all squares containing this edge
        for sq_idx in edge_to_squares[edge]:
            sq = squares[sq_idx]
            if all(e in chosen_set for e in sq if e != edge):
                return False
        return True
    
    def try_remove_edge(chosen_set, edge):
        """Remove edge from chosen_set"""
        chosen_set.discard(edge)
    
    # Greedy initialization with random edge ordering
    def greedy_init(seed):
        rng = random.Random(seed)
        edge_order = list(edges)
        rng.shuffle(edge_order)
        chosen = set()
        for e in edge_order:
            if try_add_edge(chosen, e):
                chosen.add(e)
        return chosen
    
    # Local search: add, remove, swap
    def local_search(chosen, rng, time_limit):
        """Improve chosen set via local search"""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding edges
            edges_to_try = [e for e in edges if e not in chosen]
            rng.shuffle(edges_to_try)
            for e in edges_to_try:
                if time.time() >= time_limit:
                    break
                if try_add_edge(chosen, e):
                    chosen.add(e)
                    improved = True
            
            # Try removing edges that are in many C4s, then see if we can add more
            # Actually, just try 1-1 swaps: remove one, add another
            # First, try pure removals to see if it helps (unlikely)
            # Then try swaps
            edges_in = list(chosen)
            edges_out = [e for e in edges if e not in chosen]
            rng.shuffle(edges_in)
            rng.shuffle(edges_out)
            
            for e_remove in edges_in:
                if time.time() >= time_limit:
                    break
                for e_add in edges_out:
                    if time.time() >= time_limit:
                        break
                    if e_add == e_remove:
                        continue
                    # Try swap: remove e_remove, add e_add
                    if try_add_edge(chosen - {e_remove}, e_add):
                        chosen.discard(e_remove)
                        chosen.add(e_add)
                        improved = True
                        break  # restart outer loop after one swap
                if improved:
                    break
            
            # Also try 2-for-1 or 1-for-2 swaps? Keep it simple for now.
            # Try removing one edge to allow adding two
            edges_in = list(chosen)
            edges_out = [e for e in edges if e not in chosen]
            rng.shuffle(edges_in)
            for e_remove in edges_in:
                if time.time() >= time_limit:
                    break
                reduced = chosen - {e_remove}
                # Try to add as many as possible from edges_out
                added = 0
                for e_add in edges_out:
                    if time.time() >= time_limit:
                        break
                    if try_add_edge(reduced, e_add):
                        reduced.add(e_add)
                        added += 1
                if added > 1:
                    chosen = reduced
                    improved = True
                    break
        
        return chosen
    
    # Run multiple random restarts
    best_chosen = set()
    best_size = 0
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() >= deadline - 30:  # leave time for final local search
            break
        
        seed = restart * 12345 + 67890
        chosen = greedy_init(seed)
        size = len(chosen)
        
        if size > best_size:
            best_size = size
            best_chosen = set(chosen)
        
        # Local search on this solution
        rng = random.Random(seed)
        local_time = deadline - 30 - (time.time() - (deadline - 140)) * 0  # just use deadline - 30
        local_time = min(deadline - 20, time.time() + 10)  # limit local search per restart
        chosen = local_search(chosen, rng, local_time)
        size = len(chosen)
        
        if size > best_size:
            best_size = size
            best_chosen = set(chosen)
    
    # Final polish on best solution
    if time.time() < deadline - 5 and len(best_chosen) > 0:
        rng = random.Random(99999)
        best_chosen = local_search(best_chosen, rng, deadline - 2)
    
    # Convert to list of [u, v]
    result = [[u, v] for u, v in best_chosen]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
