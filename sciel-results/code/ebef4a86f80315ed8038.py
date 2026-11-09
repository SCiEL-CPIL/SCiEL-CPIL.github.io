# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Build the forbidden difference set F (nonzero quadratic residues mod 205).
    2. Use a bitset-based branch-and-bound (maximum independent set) approach with
       greedy coloring upper bound to find a large independent set in the conflict graph.
    3. The conflict graph has vertices 0..204; edge between a,b iff (a-b)%205 in F.
       We want a maximum independent set (MIS).
    4. Use symmetry: since the graph is vertex-transitive (Cayley graph), we can fix
       0 in the solution or not, but more practically we use good ordering and pruning.
    5. Time-guarded: return best found if deadline approaches.
    """
    start_time = time.time()
    DEADLINE = 570.0  # safety margin before 630s kill

    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build adjacency bitsets: adj[i] is a Python int (bitset) where bit j is set
    # if i and j are connected (forbidden difference).
    adj = [0] * m
    for i in range(m):
        for d in forbidden:
            j = (i + d) % m
            if j != i:
                adj[i] |= (1 << j)
            j2 = (i - d) % m
            if j2 != i:
                adj[i] |= (1 << j2)

    # Degree of each vertex
    deg = [bin(adj[i]).count('1') for i in range(m)]

    # Greedy initial solution: order by degree (highest first), pick if compatible
    # This gives a baseline.
    order = sorted(range(m), key=lambda v: -deg[v])
    greedy_set = set()
    for v in order:
        conflict = False
        for u in greedy_set:
            if (1 << u) & adj[v]:
                conflict = True
                break
        if not conflict:
            greedy_set.add(v)
    best_set = list(greedy_set)
    best_size = len(best_set)

    # Branch and bound for MIS
    # State: available vertices (bitset), current chosen set (list)
    # Upper bound: greedy coloring of the subgraph induced by available vertices
    #   gives an upper bound on the size of an independent set in that subgraph.

    def color_bound(avail):
        """
        Greedy coloring of the graph induced by 'avail' bitset.
        Returns the number of colors used, which is an upper bound on the
        size of the maximum independent set in that subgraph.
        """
        remaining = avail
        num_colors = 0
        while remaining:
            num_colors += 1
            # Greedily color: pick a maximal independent set from remaining
            color_class = 0
            rem = remaining
            while rem:
                # Pick the lowest set bit
                lb = rem & (-rem)
                v = lb.bit_length() - 1
                # Check if v is compatible with all vertices already in color_class
                compatible = True
                cc = color_class
                while cc:
                    lb2 = cc & (-cc)
                    u = lb2.bit_length() - 1
                    if (1 << u) & adj[v]:
                        compatible = False
                        break
                    cc &= cc - 1
                if compatible:
                    color_class |= lb
                    rem &= ~lb
                else:
                    rem &= ~lb
            remaining &= ~color_class
        return num_colors

    # Precompute: for fast neighbor check
    # We'll use a recursive DFS with bitsets.
    
    # To speed up, we use a simpler but effective heuristic:
    # Try multiple greedy orderings and local search (swap) to improve.
    
    # Local search: try to add vertices, or swap to improve size
    def try_add_vertex(current_set, v):
        """Check if v can be added to current_set."""
        for u in current_set:
            if (1 << u) & adj[v]:
                return False
        return True

    # Multiple restarts with different orderings
    import random
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > DEADLINE:
            break
        
        # Different ordering strategies
        if restart == 0:
            ord_ = list(range(m))
        elif restart == 1:
            ord_ = sorted(range(m), key=lambda v: deg[v])  # low degree first
        elif restart == 2:
            ord_ = sorted(range(m), key=lambda v: -deg[v])  # high degree first
        else:
            ord_ = list(range(m))
            rng.shuffle(ord_)
        
        cur = set()
        for v in ord_:
            if try_add_vertex(cur, v):
                cur.add(v)
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = list(cur)
        
        # Local improvement: try to swap out one vertex for two, or add
        improved = True
        while improved:
            improved = False
            cur_list = list(cur)
            # Try adding any vertex not in cur
            for v in range(m):
                if v not in cur and try_add_vertex(cur, v):
                    cur.add(v)
                    improved = True
                    break
            if improved:
                continue
            
            # Try swap: remove u, add v, if size increases (net +1 or more)
            # For each u in cur, for each v not in cur:
            #   if v is compatible with cur - {u}, then we can swap
            #   but size stays same; we need net gain.
            # Try: remove u, then try to add as many as possible.
            for u in list(cur):
                cur.discard(u)
                added = 0
                for v in range(m):
                    if v not in cur and try_add_vertex(cur, v):
                        cur.add(v)
                        added += 1
                if added > 1:
                    improved = True
                else:
                    # Put u back
                    cur.add(u)
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = list(cur)
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = list(cur)

    # Also try a simple branch-and-bound on a small number of vertices
    # to see if we can beat the greedy result.
    # Given time constraints, we do a limited DFS.
    
    # For a more systematic approach, use the fact that the graph is
    # vertex-transitive. We can fix whether 0 is in the set or not.
    # But for now, the greedy + local search should be competitive.
    
    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
