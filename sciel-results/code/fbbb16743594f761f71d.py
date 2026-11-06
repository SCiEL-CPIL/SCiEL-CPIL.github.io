import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # 1. Construct Paley tournament on 67 vertices
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # adj[i][j] is 1 if i -> j
    # We store the tournament as a list of lists or a bitmask for speed
    # For n=67, a list of bitmasks is efficient.
    # adj[i] is a bitmask where bit j is 1 if i -> j
    
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                # i -> j
                adj[i] |= (1 << j)
            else:
                # j -> i
                adj[j] |= (1 << i)
                
    # Helper to check if a tournament (list of bitmasks) is valid
    # Every set of 4 vertices must have a common dominator.
    # A vertex v dominates a set S if v has arcs to all vertices in S.
    # So for each subset S of size 4, we need some v (v not in S? No, "common dominator" usually means a vertex in the tournament, not necessarily in S? 
    # Wait, "common dominator" in Schütte's problem: "Every set of k vertices has a vertex with arcs to all k." 
    # Does the dominator have to be in the set? No. "A vertex with arcs to all four". 
    # In tournament theory, a "king" or "dominator" of a set is a vertex that beats everyone in the set. 
    # The vertex can be inside or outside the set. 
    # Let's re-read carefully: "Every set of 4 vertices must have a vertex with arcs to all four."
    # This implies the vertex is in the tournament. It doesn't say "in the set".
    # However, if the vertex is in the set, it can't have an arc to itself. 
    # So the dominator must be a vertex $v$ such that for all $u \in S$, $v \neq u$ and $v \to u$.
    # If $S$ has 4 vertices, and $v \in S$, then $v$ cannot dominate itself. So $v$ must be outside $S$.
    # Wait, if $v \in S$, the set of vertices *in S* that $v$ dominates is $S \setminus \{v\}$. 
    # The condition is "vertex with arcs to all four". This means arcs to all 4 vertices in the set.
    # This is only possible if the dominator is NOT in the set.
    # So for every subset $S$ of size 4, there exists a vertex $v \notin S$ such that $v \to u$ for all $u \in S$.
    
    def is_valid(adj, n):
        # Precompute dominators for each 4-set? Too slow.
        # Instead, for each vertex v, compute the set of 4-sets it dominates.
        # A vertex v dominates a 4-set S if S is a subset of out-neighbors of v.
        # Let out[v] be the set of vertices v beats.
        # If |out[v]| < 4, v dominates no 4-sets.
        # We need the union of all 4-subsets of out[v] for all v to cover all 4-subsets of {0..n-1}.
        
        # This is a set cover problem.
        # For n=67, number of 4-sets is C(67,4) ~ 736k.
        # For each v, number of 4-sets dominated is C(|out[v]|, 4).
        # Sum of C(|out[v]|, 4) should be at least C(n, 4).
        
        # Efficient check:
        # 1. For each v, get list of out-neighbors.
        # 2. Generate all 4-combinations of out-neighbors.
        # 3. Mark them as covered.
        # 4. Check if all 4-combinations of {0..n-1} are covered.
        
        # This might be slow in Python for n=67.
        # C(67,4) = 736,281.
        # Average out-degree is 33. C(33,4) = 40,920.
        # 67 * 40,920 ~ 2.7 million operations. This is feasible in 1-2 seconds.
        
        covered = set()
        for v in range(n):
            # Get out-neighbors
            out_neighbors = []
            mask = adj[v]
            for u in range(n):
                if u != v and (mask & (1 << u)):
                    out_neighbors.append(u)
            
            if len(out_neighbors) < 4:
                continue
                
            for quad in itertools.combinations(out_neighbors, 4):
                covered.add(quad)
                
        # Check if all 4-sets of {0..n-1} are covered
        total_quads = 0
        for quad in itertools.combinations(range(n), 4):
            total_quads += 1
            if quad not in covered:
                return False
        return True
        
    # Greedy Vertex Removal
    current_n = p
    current_adj = adj[:]
    
    # Verify Paley 67 is valid (sanity check)
    # is_valid(current_adj, current_n) takes ~1s. We know it is valid.
    
    removed = 0
    while time.time() < deadline - 10: # Leave 10s for final check
        improved = False
        for v in range(current_n):
            # Try removing v
            # Create new adj list
            new_n = current_n - 1
            new_adj = [0] * new_n
            
            # Map old indices to new indices
            # If we remove v, indices > v shift down by 1.
            for i in range(current_n):
                if i == v:
                    continue
                new_i = i if i < v else i - 1
                mask = current_adj[i]
                new_mask = 0
                for j in range(current_n):
                    if j == v:
                        continue
                    if mask & (1 << j):
                        new_j = j if j < v else j - 1
                        new_mask |= (1 << new_j)
                new_adj[new_i] = new_mask
                
            # Check validity
            if is_valid(new_adj, new_n):
                current_n = new_n
                current_adj = new_adj
                improved = True
                removed += 1
                break # Restart loop with new n
                
        if not improved:
            break
            
    # If we found a smaller valid tournament, return it
    if current_n < p:
        # Convert adj back to orientation bits
        orientation = []
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if current_adj[i] & (1 << j):
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return {'n': current_n, 'orientation': orientation}
    else:
        # Fallback to Paley 67
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i] & (1 << j):
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}
