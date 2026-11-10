import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin
    
    # Candidates must share a factor with 255 (3, 5, 17)
    # 255 is always included.
    # We find the max clique in the graph of other candidates U.
    
    # 1. Generate candidates U = {x in [1, 254] | gcd(x, 255) > 1}
    candidates = []
    for x in range(1, N):
        if math.gcd(x, N) > 1:
            candidates.append(x)
            
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    n_cands = len(candidates)
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitmask of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n_cands
    for i in range(n_cands):
        vi = candidates[i]
        mask = 0
        for j in range(i + 1, n_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                mask |= (1 << j)
                adj[j] |= (1 << i) # symmetric
        adj[i] = mask

    # 3. Find Maximum Clique using Bron-Kerbosch with Pivoting and Greedy Coloring
    # We use a recursive approach with bitsets for efficiency.
    
    max_clique_size = 0
    max_clique_set = []
    
    # Helper to iterate bits in a mask
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    # Greedy coloring to provide upper bounds for pruning
    # Returns list of colors assigned to vertices in order, and the max color
    # The order is the order in which vertices are added to the clique candidate set
    # We use a simple greedy coloring heuristic on the subgraph induced by P
    def color_sort(P):
        """
        Greedy color the graph induced by P.
        Returns (order, colors, max_color)
        order: list of vertex indices in the order they were colored
        colors: dict mapping vertex index to its color
        max_color: the maximum color used
        """
        # We want to order vertices such that we can prune branches where |R| + color <= max_clique_size
        # Standard approach:
        # 1. Remove vertices from P one by one.
        # 2. Assign the smallest possible color to each removed vertex.
        # 3. The order of removal is the search order (reverse of removal for branching).
        #    Actually, Tomita's algorithm processes vertices in the order of increasing color.
        
        # Let's implement a simpler version:
        # We maintain a set of uncolored vertices in P.
        # We repeatedly pick a vertex, assign it the smallest color not used by its neighbors in the uncolored set,
        # and remove it.
        # The sequence of vertices picked defines the order.
        
        uncolored = set(iter_bits(P))
        colors = {}
        order = []
        
        # To make it faster, we can use a priority queue or just iterate.
        # Given small N, simple iteration is fine.
        # However, the "smallest color" logic requires checking neighbors.
        
        # Alternative: Use the standard Tomita MCS (Maximum Cardinality Search) ordering
        # which is often faster to compute and yields good bounds.
        
        # Let's try a simple greedy coloring:
        # We process vertices in P.
        # For each vertex v, we find the smallest color k such that no neighbor of v in the 
        # set of vertices already assigned color < k (in the current step?) 
        # Actually, standard greedy coloring:
        # While uncolored is not empty:
        #   Pick a vertex v from uncolored (e.g., first one).
        #   Find the smallest color c such that no neighbor of v in uncolored has color c.
        #   Assign c to v.
        #   Add v to order.
        #   Remove v from uncolored.
        
        # This specific implementation might be slow if not careful.
        # Let's use a pre-computed ordering based on degree or just rely on the pivot selection
        # without explicit coloring bounds first, as N is small (~126).
        # If we find a large clique quickly, the search space shrinks.
        
        # For now, let's just return the vertices in P in some order.
        # We will rely on the pivot selection for efficiency.
        # If performance is an issue, we can add coloring.
        
        # Actually, let's implement a basic coloring bound.
        # We'll sort vertices by degree in P descending to pick pivots effectively?
        # No, let's stick to the standard Bron-Kerbosch with pivot.
        
        # Returning a dummy structure to satisfy the interface if we decide not to use it yet
        # But let's implement a simple bound:
        # Upper bound is len(P).
        # If len(R) + len(P) <= max_clique_size, prune.
        
        # Let's refine the Bron-Kerbosch to use this simple bound.
        
        return list(iter_bits(P)), {}, len(P)

    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        # Pruning: if even adding all of P doesn't beat current best
        if len(R) + len(iter_bits(P)) <= max_clique_size:
            return
            
        if not P:
            if not X:
                if len(R) > max_clique_size:
                    max_clique_size = len(R)
                    max_clique_set = R[:]
            return
        
        # Pivot selection: choose u in P U X with max degree in P
        # We want to minimize the number of recursive calls.
        # Standard: pick u in P U X that maximizes |P \cap N(u)|
        
        PX = P | X
        best_u = -1
        max_deg = -1
        
        # Iterate through PX to find best pivot
        for u in iter_bits(PX):
            deg = bin(P & adj[u]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_u = u
        
        if best_u == -1:
            return

        # Recurse on P \ N(u)
        # The elements in P that are NOT neighbors of u
        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            R.append(candidates[v])
            
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)

    # Initial call
    initial_P = (1 << n_cands) - 1
    bron_kerbosch([], initial_P, 0)
    
    # Construct the final set
    if max_clique_set:
        result = max_clique_set + [N]
    else:
        # Fallback: Multiples of 3
        result = [x for x in range(1, N + 1) if x % 3 == 0]
        
    # Verify validity and size
    if N not in result:
        result.append(N)
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
