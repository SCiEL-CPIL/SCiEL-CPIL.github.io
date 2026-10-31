import time
import numpy as np
import itertools
import sys

D = 11
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _check_deadline():
    return _elapsed() > DEADLINE - 5.0

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

D = 11

def _tomita_mis(adj_list, n):
    """
    Tomita's branch and bound algorithm for Maximum Independent Set.
    adj_list: list of sets of neighbors (incompatibilities)
    n: number of nodes
    """
    # Color sorting heuristic (Tomita)
    # We represent the graph as a list of adjacency lists (sets for fast lookup)
    
    # Initial coloring
    # colors[i] is the color of node i
    # We use a simple greedy coloring to get an upper bound
    # However, for Tomita, we need to maintain color classes during recursion.
    
    # Simplified Tomita implementation
    # R: current independent set
    # C: candidate set
    # We use bitmasks if n <= 64, but n=5280, so we use sets/lists.
    
    # To make it fast, we pre-compute a static coloring? No, dynamic is needed for bound.
    # Let's use a simpler recursive MIS with coloring bound.
    
    # colors[i] will be the color of node i in the subgraph induced by C.
    # We maintain a list of nodes in C.
    
    # This implementation is standard for Python but might be slow for 5000 nodes if not optimized.
    # We will use a list of lists for adjacency.
    
    # For efficiency, we'll convert adj_list to a list of lists.
    adj = [list(s) for s in adj_list]
    
    # Pre-compute initial colors? 
    # Tomita's algorithm:
    # 1. Color the graph.
    # 2. Sort vertices by color.
    # 3. Branch on the vertex with the highest color (or last in sorted order).
    
    # Let's use a recursive function with memoization? No, state is too large.
    
    # We will use a standard Tomita implementation.
    
    # To speed up, we can use a greedy coloring of the whole graph first to get an initial solution.
    # But Tomita does this internally.
    
    # Given the time limit (120s), and n=5280, a pure Python Tomita might be tight.
    # However, the graph is very sparse (incompatibilities are few).
    # Degree of a node:
    # Incompatibilities:
    # 1. Same vector (0 others)
    # 2. Intersection size 3, no sign flips.
    #    For a fixed vector, how many others have intersection size 3 with it and no sign flips?
    #    Support S (size 4).
    #    Pick 3 from S: $\binom{4}{3} = 4$.
    #    Pick 1 from complement (11-4=7): 7.
    #    Sign of the 3 intersection coords must match u. (1 way).
    #    Sign of the new coord can be + or - (2 ways).
    #    So $4 \times 7 \times 1 \times 2 = 56$ incompatibilities per node.
    #    Total edges $\approx 5280 \times 56 / 2 \approx 147k$.
    #    This is a very sparse graph.
    #    MIS on sparse graphs is often easier/faster.
    
    # Let's implement Tomita.
    
    # colors[i] is the color of node i
    # We maintain a list `cand` of nodes in the current candidate set.
    # We sort `cand` by color.
    
    # To avoid deep recursion overhead, we might need to be careful.
    # But depth is at most n (5280). Python recursion limit is 1000 by default.
    # We must increase recursion limit or use iterative deepening?
    # Tomita is recursive.
    # sys.setrecursionlimit(10000)
    
    # Let's use a simpler approach if Tomita is too risky:
    # Greedy + Local Search?
    # Or, since the graph is so sparse, maybe a greedy approach with random restarts is enough?
    # The previous greedy scored 1.69.
    # The inspirations scored 2.57 with Tomita.
    # Let's try Tomita.
    
    sys.setrecursionlimit(10000)
    
    best_set = []
    
    # Greedy coloring function for a set of nodes
    def color_graph(nodes):
        # Returns colors for nodes in `nodes`
        # Greedy: assign lowest possible color
        # We need to map node id to index in `nodes` for efficiency?
        # Or just use dict.
        # Since we call this often, we need it to be fast.
        
        # We will return a list of (node, color) sorted by color.
        colors = {}
        # Simple greedy: iterate through nodes, assign color
        # To make it deterministic and fast, we can use a fixed order.
        # However, Tomita requires the coloring to be valid for the induced subgraph.
        
        # Efficient coloring:
        # For each node in nodes:
        #   forbidden = set of colors of its neighbors in nodes
        #   color = smallest non-negative int not in forbidden
        #   colors[node] = color
        
        # This is O(N * D) per call.
        
        # Optimization: We can pre-compute neighbor sets.
        
        # Let's implement a fast greedy coloring.
        # We'll return a list of nodes sorted by color.
        
        # To speed up, we can use a bitset if n was small, but n=5280.
        
        # We'll use a dictionary for colors.
        node_colors = {}
        
        # Order of processing:
        # Tomita suggests sorting by degree or just arbitrary?
        # Arbitrary is fine for correctness, degree might help bound.
        
        # Let's just iterate through `nodes` (which is a list).
        # We need to know which neighbors are in `nodes`.
        # `nodes` is a list. We can make a set `nodes_set` for O(1) lookup.
        
        nodes_set = set(nodes)
        
        for u in nodes:
            # Find neighbors of u in nodes_set
            # adj[u] is a list of neighbors
            forbidden = set()
            for v in adj[u]:
                if v in nodes_set:
                    if v in node_colors:
                        forbidden.add(node_colors[v])
            
            # Find smallest color
            c = 0
            while c in forbidden:
                c += 1
            node_colors[u] = c
        
        # Sort nodes by color
        sorted_nodes = sorted(nodes, key=lambda x: node_colors[x])
        return sorted_nodes, node_colors

    def expand(R, C):
        nonlocal best_set
        
        if not C:
            if len(R) > len(best_set):
                best_set = list(R)
            return
        
        # Color C
        sorted_C, colors = color_graph(C)
        
        # Upper bound: max color + 1
        # Tomita's bound: max color assigned + 1
        # Since colors are 0..k, max color is k. Size of C is |C|.
        # The bound is max_color + 1.
        max_color = max(colors.values()) if colors else 0
        if len(R) + max_color + 1 <= len(best_set):
            return
        
        # Process nodes in reverse color order (highest color first)
        # Actually, Tomita processes from the end of the sorted list (highest color) backwards.
        # Wait, the standard Tomita implementation:
        # Sort C by color.
        # For i from len(C)-1 down to 0:
        #   u = C[i]
        #   if len(R) + i + 1 <= len(best_set): break
        #   R.add(u)
        #   C_new = C[0..i-1] intersect N(u)  (neighbors of u in C with lower color)
        #   expand(R, C_new)
        #   R.remove(u)
        
        # Note: C_new must be the neighbors of u in C that have color < color(u)?
        # Or just neighbors in C[0..i-1]?
        # In Tomita, C is sorted by color.
        # C_new = { v in C[0..i-1] | v is neighbor of u }
        
        # Let's implement this.
        
        # We need to be careful with the list slicing and set intersection.
        
        # To optimize, we can pre-compute the intersection.
        
        # C is a list.
        # We iterate i from len(C)-1 down to 0.
        
        # We need a list of neighbors of u that are in C.
        # Since C is small in deeper recursion, this is fast.
        
        # Optimization: Use a set for C for fast lookup?
        # C is a list. We can maintain a set `C_set` alongside `C`.
        
        # Let's refactor expand to take C as a list and maintain C_set?
        # Or just use set(C) inside?
        
        # Given the sparsity, set(C) is O(|C|).
        
        # Let's try.
        
        # We pass C as a list.
        
        # We need to update C in the loop.
        # Standard Tomita modifies C in place or creates new lists.
        
        # Let's use a list for C.
        
        # We need to know the neighbors of u in C.
        
        # We'll iterate.
        
        # To avoid copying large lists, we can use indices.
        
        # However, Python list slicing creates new lists.
        
        # For n=5280, this might be slow.
        
        # Alternative: Use bitmasks? No.
        
        # Let's rely on the sparsity.
        
        # We will implement a version that uses sets for C to speed up neighbor filtering.
        
        # Re-defining expand to use sets?
        # If we use sets, we lose the sorted order.
        # We need the sorted order for the bound.
        
        # So we keep the sorted list.
        
        # We can maintain a parallel set `C_set`.
        
        # Let's modify the function signature to pass C_set as well?
        # Or just compute it.
        
        # Let's compute C_set = set(C) at the start of expand.
        
        C_set = set(C)
        
        # Iterate from end to start
        for i in range(len(C) - 1, -1, -1):
            if len(R) + i + 1 <= len(best_set):
                break
            
            u = C[i]
            
            R.add(u)
            
            # C_new = neighbors of u in C[0..i-1]
            # Neighbors of u are in adj[u]
            # We need v in adj[u] such that v in C_set and index(v) < i?
            # Or just v in C_set and v != u?
            # Since C is sorted, C[0..i-1] are the nodes with lower or equal color?
            # Actually, colors are non-decreasing.
            # Nodes with same color are adjacent? No, independent set coloring means no adjacent nodes have same color.
            # So C[0..i-1] contains nodes with color <= color(u).
            # But we only want neighbors.
            
            # We can just check if v in C_set.
            # But we must exclude nodes with index > i?
            # In Tomita, we only consider nodes in C that come before u in the sorted order.
            # This ensures we don't double count and maintains the property.
            
            # So C_new = { v in adj[u] | v in C_set and C.index(v) < i }?
            # Finding index is O(N).
            
            # Better: Keep a list of nodes in C.
            # We can just filter adj[u] by C_set.
            # But we must respect the order?
            # The next call to expand will re-sort C_new.
            # So we just need the SET of valid neighbors.
            
            # So C_new = set(adj[u]) & C_set
            # But we must remove u? u is not in adj[u].
            # And we must only include nodes that were in C[0..i-1]?
            # If we include a node with index > i, it would have been processed later?
            # No, we are iterating backwards.
            # Nodes with index > i have already been "considered" in this level of recursion?
            # Actually, the logic of Tomita is:
            # We partition C into C[0..i-1] and {u} and C[i+1..].
            # We branch on u.
            # The new candidate set is C[0..i-1] intersect N(u).
            # Nodes in C[i+1..] are discarded for this branch because they have higher color and we are going to add u (which has lower color index? No, higher color value).
            # Wait, sorted by color ascending.
            # C[0] has color 0, C[-1] has color K.
            # We iterate i from K down to 0.
            # So u has high color.
            # C[0..i-1] has lower or equal color.
            # If we pick u, we can only add neighbors from C[0..i-1].
            # Why? Because any neighbor in C[i+1..] would have higher color.
            # But if they are neighbors, they must have different colors.
            # So they could have higher color.
            # But Tomita's proof relies on the fact that we can discard them.
            # Yes, the standard algorithm uses C[0..i-1].
            
            # So we need to filter adj[u] to only those in C[0..i-1].
            
            # How to check if v is in C[0..i-1] efficiently?
            # We can maintain a set of nodes in C[0..i-1]?
            # We can build it incrementally?
            # Or just check `C.index(v) < i`? Too slow.
            
            # Alternative:
            # Since we re-sort in the next call, we just need the set of neighbors in C[0..i-1].
            # We can create a set `C_prefix`?
            # We can maintain a set `C_set` and remove u? No, we need the prefix.
            
            # Let's use a different strategy:
            # Pass C as a list.
            # Create a set `C_set` = set(C).
            # For each u, `neighbors_in_C` = set(adj[u]) & C_set.
            # Then filter `neighbors_in_C` to keep only those with index < i.
            # We can pre-compute the index of each node in C?
            # `index_map = {node: idx for idx, node in enumerate(C)}`
            # Then `C_new = [v for v in neighbors_in_C if index_map[v] < i]`
            
            # This is O(deg(u)) per node.
            # Total O(N * deg).
            
            # Let's implement this.
            
            if i > 0:
                # Build index map for C?
                # This is O(N) per node u. Total O(N^2) per level.
                # Too slow.
                
                # Optimization:
                # We can just check `v in C_set` and then...
                # Actually, if we don't filter by index, we might include nodes with index > i.
                # Does it matter?
                # If we include a node w with index > i in C_new, and we add it later,
                # it means we are adding two nodes u and w that are neighbors?
                # No, w is a neighbor of u. So they can't both be in the independent set.
                # So if we add u, w MUST be removed from the candidate set.
                # So C_new should NOT contain w.
                # So we MUST filter out neighbors of u.
                # And we MUST filter out nodes with index > i?
                # If w has index > i, it is in C.
                # If w is a neighbor of u, it is removed.
                # If w is NOT a neighbor of u, it is kept?
                # But in Tomita, we discard C[i+1..].
                # Why?
                # Because we are processing in order of decreasing color.
                # If we don't discard them, we might double count or violate the bound.
                # The bound `len(R) + i + 1` relies on the fact that we have `i+1` nodes left in C[0..i].
                # If we keep nodes from C[i+1..], the count is wrong.
                
                # So we MUST restrict to C[0..i-1].
                
                # Efficient way:
                # We can maintain a set `active_nodes`?
                # Or, since we are iterating backwards, we can just use the list slice?
                # `C_prefix = C[:i]`
                # `C_prefix_set = set(C_prefix)`
                # `C_new = [v for v in adj[u] if v in C_prefix_set]`
                
                # This creates a new list and set each time.
                # `C[:i]` is O(i).
                # `set(C_prefix)` is O(i).
                # Total O(N^2) per level.
                
                # For N=5280, N^2 is 27 million.
                # If depth is 10, that's 270 million operations.
                # Python might be too slow.
                
                # However, the graph is sparse.
                # And the depth might be small?
                # Or the branching factor is small?
                
                # Let's try to optimize the inner loop.
                
                # We can pre-compute the sets for each i? No, C changes.
                
                # Let's use a simpler heuristic if Tomita is too slow:
                # Greedy with random restarts.
                # But inspiration says Tomita works.
                
                # Let's stick to Tomita and hope it's fast enough.
                # We can add a time check inside the loop.
                
                if _check_deadline():
                    return
                
                C_prefix = C[:i]
                C_prefix_set = set(C_prefix)
                C_new = [v for v in adj[u] if v in C_prefix_set]
            else:
                C_new = []
            
            expand(R, C_new)
            
            R.remove(u)
            
            if _check_deadline():
                return

    # Initial call
    # C = all nodes
    # R = empty set
    all_nodes = list(range(n))
    R = set()
    expand(R, all_nodes)
    
    return best_set


def construct():
    # Generate weight-4 vectors
    vectors = []
    indices = []
    
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            vectors.append(v)
            indices.append((coords, signs))
            
    n = len(vectors)
    
    # Build incompatibility graph
    # adj[i] is a set of indices j such that vectors[i] and vectors[j] are incompatible.
    
    adj = [set() for _ in range(n)]
    
    # For each pair, check compatibility.
    # This is O(N^2). N=5280. N^2 = 27 million.
    # Checking compatibility:
    # We need to check if dist^2 < 4.
    # We derived the condition:
    # Incompatible if:
    # 1. Same vector (dist 0).
    # 2. Intersection size 3 and no sign flips in intersection.
    
    # We can check this fast using the (coords, signs) representation.
    
    for i in range(n):
        if _check_deadline():
            break
        ci, si = indices[i]
        # ci is a tuple of 4 coords
        # si is a tuple of 4 signs
        
        # Create a dict for fast lookup?
        # Or just loop j?
        
        # To speed up, we can group by support?
        # Incompatibilities only happen if supports overlap.
        # If supports are disjoint, dist^2 = 8 >= 4. Compatible.
        # If supports overlap by 1 or 2, dist^2 >= 4. Compatible.
        # If supports overlap by 3 or 4, check signs.
        
        # So we only need to check j where |S_i \cap S_j| >= 3.
        
        # We can pre-group vectors by support?
        # There are 330 supports.
        # For each support, there are 16 vectors.
        
        # Let's group vectors by support.
        
    # Re-structure:
    # support_to_vectors = {}
    # for i, (coords, signs) in enumerate(indices):
    #     support_to_vectors.setdefault(coords, []).append(i)
    
    # Then for each i:
    #   ci = indices[i][0]
    #   # Check against all j in same support (intersection 4)
    #   for j in support_to_vectors[ci]:
    #       if i == j: continue
    #       # Check signs
    #       # Incompatible if signs are same?
    #       # Dist^2 = 4 * (num flips).
    #       # If 0 flips, dist^2 = 0. (Same vector)
    #       # If >= 1 flip, dist^2 >= 4.
    #       # So if i != j, they are always compatible?
    #       # Yes, distinct vectors with same support are compatible.
    #       # So NO edges within same support group?
    #       # Wait. If i != j, they are distinct.
    #       # So dist > 0.
    #       # Min dist for same support is 2 (1 flip).
    #       # So they are compatible.
    #       # So NO incompatibilities within same support.
        
    #   # Check against supports that overlap by 3.
    #   # How to find supports overlapping by 3?
    #   # For a given support S (size 4), the supports overlapping by 3 are those that share 3 elements and differ by 1.
    #   # There are 4 * 7 = 28 such supports.
    #   # For each such support S', check all vectors in S'.
    
    #   # This is much faster than N^2.
    
    support_to_vectors = {}
    for i, (coords, signs) in enumerate(indices):
        support_to_vectors.setdefault(coords, []).append(i)
        
    # Map from 3-subset to list of supports?
    # Or just generate the 28 neighbors for each support.
    
    for i in range(n):
        if _check_deadline():
            break
        ci, si = indices[i]
        
        # Generate supports overlapping by 3
        # Remove each of the 4 coords, add a new coord from the remaining 7.
        for k in range(4):
            # Remove ci[k]
            remaining = [c for c in ci if c != ci[k]]
            # Add new coord
            for new_c in range(D):
                if new_c in ci:
                    continue
                # New support
                new_coords = tuple(sorted(remaining + [new_c]))
                # Check if this support exists
                if new_coords in support_to_vectors:
                    for j in support_to_vectors[new_coords]:
                        if i == j:
                            continue
                        # Check signs
                        # Intersection is `remaining` (3 coords)
                        # We need to check if signs match on these 3 coords.
                        # If signs match, dist^2 = 2 < 4. Incompatible.
                        # If any sign differs, dist^2 >= 6. Compatible.
                        
                        # Get signs for j
                        cj, sj = indices[j]
                        
                        # Map coord to sign for i and j
                        # ci is sorted? No, itertools.combinations gives sorted.
                        # So ci[k] is the removed one.
                        # remaining is the other 3.
                        
                        # We need to compare si and sj on the 3 common coords.
                        
                        # Create dict for i
                        sign_i = {c: s for c, s in zip(ci, si)}
                        sign_j = {c: s for c, s in zip(cj, sj)}
                        
                        compatible = False
                        for c in remaining:
                            if sign_i[c] != sign_j[c]:
                                compatible = True
                                break
                            
                        if not compatible:
                            # Incompatible
                            adj[i].add(j)
                            adj[j].add(i)
                            
    # Run Tomita MIS
    best_set = _tomita_mis(adj, n)
    
    # Convert best_set to array
    if not best_set:
        # Fallback
        return np.array(vectors[:10], dtype=np.int64)
        
    result = np.array([vectors[i] for i in best_set], dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
