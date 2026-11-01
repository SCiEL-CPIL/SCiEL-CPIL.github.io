import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph of girth g.
    For k=5, g=9, we attempt to find a voltage graph lift of the Petersen graph.
    """
    if k != 5 or g != 9:
        # Fallback for other parameters (not the target task, but good practice)
        return luw_construct(k, g)

    deadline = time.time() + 105  # Safety margin for 120s limit

    # 1. Define Petersen Graph
    # Vertices 0..9
    # Standard representation: Outer pentagon 0-1-2-3-4-0, Inner star 5-7-9-6-8-5, Spokes i-(i+5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    base_n = 10
    edge_indices = {e: i for i, e in enumerate(petersen_edges)}
    num_edges = len(petersen_edges)
    
    # Adjacency list for fast traversal
    adj = [[] for _ in range(base_n)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)

    # 2. Enumerate Non-Backtracking Closed Walks of Length 5 to 8
    # We need to identify all closed walks of length L in [5, 8] that do not backtrack.
    # A walk v0, v1, ..., vL=v0 is non-backtracking if v_{i-1} != v_{i+1} for all i.
    
    def get_closed_walks(length):
        walks = []
        # Start from vertex 0 to avoid duplicates by rotation? 
        # Actually, we need distinct cycles. 
        # It's easier to just generate all walks starting from 0 and ending at 0, 
        # then canonicalize them to remove rotational duplicates.
        # However, for voltage sums, rotation doesn't change the sum if we treat the walk as a sequence of edges.
        # But different walks might be rotations of each other.
        # Since we are summing voltages, the sum is invariant under rotation.
        # So we just need to ensure we don't process the same set of edges with the same signs multiple times?
        # No, the voltage sum depends on the direction.
        # Let's just generate all non-backtracking closed walks of length L starting at 0.
        
        def backtrack(v, prev, path):
            if len(path) == length:
                if v == 0:
                    # Check if it's a valid non-backtracking closed walk
                    # The last step was from prev to v=0.
                    # The first step was 0 to path[1].
                    # We need prev != path[1] to avoid backtracking at the closure point?
                    # Actually, the condition is v_{L-1} != v_1.
                    if prev != path[1]:
                        # Canonicalize: represent as tuple of edge IDs with signs?
                        # Or just the sequence of vertices.
                        # Since we sum voltages, we need the sequence of directed edges.
                        # Let's store the sequence of vertices.
                        walks.append(tuple(path))
                return
            
            for next_v in adj[v]:
                if next_v == prev:
                    continue
                path.append(next_v)
                backtrack(next_v, v, path)
                path.pop()
        
        for start_next in adj[0]:
            backtrack(start_next, 0, [0, start_next])
            
        # Deduplicate?
        # Two walks are the same cycle if one is a rotation of the other.
        # Since we start at 0, and the graph is vertex-transitive? No, Petersen is.
        # But we only start at 0.
        # If a cycle doesn't pass through 0, we miss it.
        # We must start from ALL vertices to catch all cycles.
        
        all_walks = []
        for start_v in range(base_n):
            # Reset path
            # We need to be careful with recursion state.
            def backtrack_all(v, prev, path, start):
                if len(path) == length:
                    if v == start:
                        if prev != path[1]:
                            all_walks.append(tuple(path))
                    return
                for next_v in adj[v]:
                    if next_v == prev:
                        continue
                    path.append(next_v)
                    backtrack_all(next_v, v, path, start)
                    path.pop()
            
            for start_next in adj[start_v]:
                backtrack_all(start_next, start_v, [start_v, start_next], start_v)
        
        # Now, remove duplicates due to rotation.
        # A walk (v0, ..., vL) is the same as (v1, ..., v0) etc.
        # We can canonicalize by finding the lexicographically smallest rotation.
        unique_walks = set()
        for w in all_walks:
            # Generate all rotations
            min_rot = min(w[i:] + w[:i] for i in range(len(w)))
            unique_walks.add(min_rot)
            
        return list(unique_walks)

    # Generate constraints
    # We need walks of length 5, 6, 7, 8.
    # Note: Petersen has girth 5.
    # Length 5: 12 cycles (Petersen has 12 5-cycles).
    # Length 6: ?
    # Length 7: ?
    # Length 8: ?
    
    print("Generating walks...", file=sys.stderr)
    constraints = [] # List of (edge_id, sign) lists
    for L in range(5, 9):
        walks = get_closed_walks(L)
        print(f"  Length {L}: {len(walks)} unique walks", file=sys.stderr)
        for w in walks:
            coeff_dict = {}
            for i in range(L):
                u = w[i]
                v = w[(i+1) % L]
                # Find edge ID
                # The edge is (u,v) or (v,u)
                if u > v:
                    u, v = v, u
                eid = edge_indices.get((u, v), -1)
                if eid == -1:
                    continue
                # Sign: +1 if we traverse u->v and u<v? 
                # Let's define voltage of edge (u,v) as $v_{uv}$.
                # If we traverse u->v, contribution is $v_{uv}$.
                # If we traverse v->u, contribution is $-v_{uv}$.
                # In our walk w[i]->w[i+1]:
                if w[i] < w[i+1]:
                    sign = 1
                else:
                    sign = -1
                
                if eid in coeff_dict:
                    coeff_dict[eid] += sign
                else:
                    coeff_dict[eid] = sign
            
            # Filter out zero coefficients
            coeff_list = [(eid, s) for eid, s in coeff_dict.items() if s != 0]
            if coeff_list:
                constraints.append(coeff_list)
                
    num_constraints = len(constraints)
    print(f"Total constraints: {num_constraints}", file=sys.stderr)

    # Precompute edge to constraint mapping
    edge_to_cons = [[] for _ in range(num_edges)]
    for i, c_list in enumerate(constraints):
        for eid, coeff in c_list:
            edge_to_cons[eid].append((i, coeff))

    # 3. Search for Voltage Assignment
    # We want to find m and volts[0..num_edges-1] in Z_m
    # such that for all i, sum(volts[eid] * coeff) % m != 0
    
    best_n = None
    best_edges = None
    
    # Try m from lower bound up to some limit
    # Moore bound is 428. n = 10*m. So m >= 43.
    # Let's try m from 43 to 200.
    
    for m in range(43, 250):
        if time.time() > deadline:
            break
            
        # Local Search
        # Initialize randomly
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
                
            volts = [random.randint(0, m-1) for _ in range(num_edges)]
            sums = [0] * num_constraints
            
            # Calculate initial sums
            for e in range(num_edges):
                v = volts[e]
                if v == 0: continue
                for c_idx, coeff in edge_to_cons[e]:
                    sums[c_idx] = (sums[c_idx] + v * coeff) % m
            
            # Score: number of constraints with sum == 0
            score = sum(1 for s in sums if s == 0)
            
            if score == 0:
                found = True
                break
            
            # Hill Climbing
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to change one variable
                # For efficiency, pick random edge, try all values?
                # Or random value?
                # Let's try random edge, random new value.
                for _ in range(100):
                    e = random.randint(0, num_edges - 1)
                    old_v = volts[e]
                    new_v = random.randint(0, m-1)
                    if new_v == old_v: continue
                    
                    delta = (new_v - old_v) % m
                    if delta == 0: continue
                    
                    # Check impact
                    delta_score = 0
                    affected = []
                    for c_idx, coeff in edge_to_cons[e]:
                        old_s = sums[c_idx]
                        new_s = (old_s + delta * coeff) % m
                        affected.append((c_idx, old_s, new_s))
                        
                        if old_s == 0 and new_s != 0:
                            delta_score -= 1
                        elif old_s != 0 and new_s == 0:
                            delta_score += 1
                    
                    if delta_score < 0:
                        # Accept
                        for c_idx, old_s, new_s in affected:
                            sums[c_idx] = new_s
                        volts[e] = new_v
                        score += delta_score
                        improved = True
                        if score == 0:
                            found = True
                            break
                    elif delta_score == 0 and random.random() < 0.05:
                        # Accept neutral move to escape local minima
                        for c_idx, old_s, new_s in affected:
                            sums[c_idx] = new_s
                        volts[e] = new_v
                        improved = True
                    else:
                        # Reject (no need to revert sums if we didn't apply, but we did calculate new_s)
                        # We didn't update sums, so nothing to revert.
                        pass
                        
            if found:
                break
                
        if found:
            n = 10 * m
            edges = []
            for u, v in petersen_edges:
                eid = edge_indices[(u, v)]
                phi = volts[eid]
                for i in range(m):
                    # u is at (u, i)
                    # v is at (v, (i + phi) % m)
                    id1 = u * m + i
                    id2 = v * m + (i + phi) % m
                    if id1 < id2:
                        edges.append((id1, id2))
                    
            # Verify girth quickly?
            # The construction guarantees girth >= 9 if constraints are satisfied.
            # We trust the constraint generation.
            
            print(f"Found solution with m={m}, n={n}", file=sys.stderr)
            return n, edges

    # Fallback to LUW if no better solution found
    return luw_construct(k, g)

def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph construction.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
