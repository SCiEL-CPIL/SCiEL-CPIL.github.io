"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    n = 256
    # Build conflict graph: edge (a,b) if (a*b) % (a+b) == 0
    # Use 0-indexed: vertex i represents number i+1
    # adjacency as list of bitmasks (using Python ints as bitsets)
    
    # Precompute conflicts
    conflicts = [0] * (n + 1)  # 1-indexed
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)
    
    # Maximum independent set via branch and bound
    # We want to select a subset with no internal conflicts
    
    # Use recursive branch-and-bound with bitsets
    # State: available candidates (bitmask), current selected set size
    # We'll use a greedy upper bound: for remaining candidates, 
    # upper bound = current_size + max possible from remaining
    
    # Better: use the fact that in a graph, MIS size <= n - min_vertex_cover
    # Or use: for a set of vertices, MIS <= sum over connected components
    
    # Simple approach: branch on vertices, use coloring-based upper bound
    # For n=256, let's use a simpler bound: greedy coloring of the conflict graph
    # on the remaining candidates gives an upper bound on independent set size
    
    # Actually, for maximum independent set, we can use:
    # Upper bound = size of a greedy coloring of the subgraph induced by candidates
    # (each color class is an independent set, so MIS >= max color class size, 
    #  but actually we need: MIS <= number of colors in a proper coloring)
    # Wait: a proper coloring partitions vertices into independent sets, so 
    # MIS <= max color class? No, MIS can span multiple color classes.
    # Correct: MIS of G <= alpha(G) and any proper coloring with k colors gives 
    # alpha(G) <= n, but the bound is: if we color with k colors, then the 
    # maximum independent set can have at most... hmm.
    
    # Better upper bound: greedy set cover / domination
    # Or: for the remaining candidate set S, compute a greedy independent set 
    # and use |greedy_IS| as a lower bound, and for upper bound use 
    # |S| (trivial) or a clique cover bound.
    
    # Let me use a different approach: since n=256 is moderate, 
    # I'll use a branch-and-bound that picks a vertex and either includes or excludes it.
    # Upper bound: for remaining vertices, the max independent set size is at most 
    # the size of a maximum matching complement... too complex.
    
    # Simpler: use the bound that in any graph, alpha(G) <= n - |E|/delta... 
    # Let me just use: upper bound = current_size + popcount(remaining)
    # and prune when upper_bound <= best_found.
    
    # This may be slow. Let me try a smarter approach.
    
    # Key observation: the conflict graph is likely very sparse. 
    # Let's count edges first.
    
    edge_count = 0
    for a in range(1, n + 1):
        edge_count += bin(conflicts[a]).count('1')
    edge_count //= 2
    
    # For sparse graphs, branch and bound works well.
    # Let's implement a proper branch and bound.
    
    # Convert to 0-indexed for bit operations
    # Vertex i (0-indexed) represents number i+1
    # conflicts0[i] is bitmask of conflicts for vertex i
    
    conflicts0 = [0] * n
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a != b and (conflicts[a] >> b) & 1:
                conflicts0[a-1] |= (1 << (b-1))
    
    # Actually let me rebuild cleanly
    conflicts0 = [0] * n
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts0[a-1] |= (1 << (b-1))
                conflicts0[b-1] |= (1 << (a-1))
    
    # Branch and bound for maximum independent set
    # We'll use a recursive approach with memoization on the candidate set (as a frozenset or bitmask)
    # But bitmask of 256 bits is large for memoization. Let's use recursion with pruning.
    
    best_size = 0
    best_set = [0] * n  # 1 if selected
    
    # Greedy initial solution: pick vertices in order, skip if conflicts with already picked
    greedy_selected = [False] * n
    for i in range(n):
        can_pick = True
        for j in range(i):
            if greedy_selected[j] and (conflicts0[i] >> j) & 1:
                can_pick = False
                break
        if can_pick:
            greedy_selected[i] = True
            best_size += 1
    
    # Store best solution
    best_solution = [i for i in range(n) if greedy_selected[i]]
    
    # Now branch and bound
    # To speed up, order vertices by degree (descending) for better pruning
    degrees = [bin(conflicts0[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    # Map: position in order -> vertex
    # For the algorithm, we'll process vertices in this order
    
    # Actually, let me use a different strategy: 
    # Pick a vertex v from remaining. Branch: include v (remove v and its neighbors) or exclude v.
    # Upper bound: current_count + greedy_IS_size(remaining)
    
    # Let me implement with a time limit check
    import time
    start_time = time.time()
    TIMEOUT = 55  # seconds, leave margin
    
    # Use iterative deepening or just straightforward B&B
    # Given the sparsity, let's try.
    
    # For efficiency, represent the remaining candidate set as a Python int (bitmask)
    # n=256, so we need 256-bit integers. Python handles this.
    
    # Precompute: for each vertex, its neighbor mask
    # Already have conflicts0[i] as the neighbor mask for vertex i
    
    # Greedy independent set size on a given candidate set (bitmask):
    # This is expensive to compute exactly. Use a simple greedy:
    # Pick vertices in some order, skip if neighbor already picked.
    
    def greedy_is_size(candidates):
        """Greedy maximum independent set size on candidate bitmask."""
        if candidates == 0:
            return 0
        count = 0
        remaining = candidates
        while remaining:
            # Pick the vertex with minimum degree in remaining subgraph
            # For simplicity, just pick the lowest set bit
            # Better: pick vertex with fewest neighbors in remaining
            best_v = -1
            best_deg = float('inf')
            r = remaining
            while r:
                lb = r & (-r)
                v = lb.bit_length() - 1
                deg = bin(conflicts0[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                r ^= lb
            # Include best_v
            count += 1
            # Remove best_v and its neighbors
            remaining &= ~(1 << best_v)
            remaining &= ~conflicts0[best_v]
        return count
    
    # This greedy is O(n^2) per call, which is too slow for B&B.
    # Let me use a simpler upper bound: just popcount(candidates)
    # And rely on the structure of the problem.
    
    # Alternative: since the current answer is 128 (all odds), and the graph is sparse,
    # let me try to improve upon 128 by checking if we can add some evens.
    
    # Actually, let me just try a local search / greedy improvement from the all-odd set.
    
    # Start with all odd numbers
    selected = set()
    for i in range(1, n + 1, 2):
        selected.add(i)
    
    # Try to add even numbers one by one
    # For each even number e, check if it conflicts with any selected odd number
    # If not, add it.
    
    # But we might need to remove some odds to add more evens.
    # Let's do a simple hill-climbing: try adding each unselected vertex, 
    # and if it causes conflicts, consider removing conflicting vertices.
    
    # Simpler: just try to add each even number if it doesn't conflict with current set.
    for e in range(2, n + 1, 2):
        can_add = True
        for s in selected:
            if (e * s) % (e + s) == 0:
                can_add = False
                break
        if can_add:
            selected.add(e)
    
    # Now try removing odds to add more evens
    # For each even not in selected, see which odds it conflicts with.
    # If removing those odds and adding the even increases size, do it.
    
    improved = True
    while improved:
        improved = False
        for e in range(2, n + 1, 2):
            if e in selected:
                continue
            # Find conflicts with selected
            conflicts_with = set()
            for s in selected:
                if (e * s) % (e + s) == 0:
                    conflicts_with.add(s)
            # If we remove conflicts_with and add e, net change = 1 - len(conflicts_with)
            if 1 - len(conflicts_with) > 0:
                selected -= conflicts_with
                selected.add(e)
                improved = True
    
    # Also try removing evens to add odds (though we started with all odds)
    # And general swaps: remove a set S, add a set T, if |T| > |S| and no conflicts
    
    # Convert to list
    result = sorted(selected)
    
    return result
# EVOLVE-BLOCK-END
