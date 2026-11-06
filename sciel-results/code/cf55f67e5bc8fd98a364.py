import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n = 20
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a permutation.
    # There are 3 distinct cycles for each set of 4 vertices.
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    # We represent a cycle as a tuple of 4 vertices in order.
    all_cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        all_cycles.append((a, b, c, d))
        all_cycles.append((a, b, d, c))
        all_cycles.append((a, c, b, d))
        
    # Precompute edges for each cycle
    # cycle_edges[i] = list of 4 edges (u,v) in cycle i
    # edge_cycles[j] = list of cycle indices that contain edge j
    # Edge index mapping: (u,v) -> idx, where idx = u*n + v (if u<v) or just use dict
    edge_to_idx = {}
    idx_to_edge = []
    edge_list = []
    for u in range(n):
        for v in range(u+1, n):
            idx = len(edge_list)
            edge_to_idx[(u,v)] = idx
            idx_to_edge.append((u,v))
            edge_list.append((u,v))
            
    num_edges = len(edge_list)
    
    # Map cycle to edge indices
    cycle_edge_indices = []
    # Initialize edge_cycles
    edge_cycles = [[] for _ in range(num_edges)]
    
    for cyc in all_cycles:
        u, v, x, y = cyc
        e1 = (min(u,v), max(u,v))
        e2 = (min(v,x), max(v,x))
        e3 = (min(x,y), max(x,y))
        e4 = (min(y,u), max(y,u))
        
        idx1 = edge_to_idx[e1]
        idx2 = edge_to_idx[e2]
        idx3 = edge_to_idx[e3]
        idx4 = edge_to_idx[e4]
        
        cycle_edge_indices.append([idx1, idx2, idx3, idx4])
        
        edge_cycles[idx1].append(len(cycle_edge_indices)-1)
        edge_cycles[idx2].append(len(cycle_edge_indices)-1)
        edge_cycles[idx3].append(len(cycle_edge_indices)-1)
        edge_cycles[idx4].append(len(cycle_edge_indices)-1)

    def is_bad_cycle(adj, cyc_idx):
        """Check if a specific cycle is bad (non-rainbow). Returns True if bad."""
        e_indices = cycle_edge_indices[cyc_idx]
        colors = []
        for idx in e_indices:
            u, v = idx_to_edge[idx]
            c = adj[u][v]
            if c == -1:
                return False # Not a cycle in the current graph
            colors.append(c)
        return len(set(colors)) < 4

    def get_initial_bad_counts(adj):
        """Calculate bad count for all cycles."""
        bad_counts = [0] * len(all_cycles)
        for i in range(len(all_cycles)):
            if is_bad_cycle(adj, i):
                bad_counts[i] = 1
        return bad_counts

    def total_bad(bad_counts):
        return sum(bad_counts)

    def make_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    edges.append([u, v, adj[u][v]])
        return edges

    def edge_count(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    count += 1
        return count

    best_edges = [[0, v, 0] for v in range(1, n)] # Star fallback
    best_count = len(best_edges)
    
    def solve_attempt(initializer):
        adj = [[-1]*n for _ in range(n)]
        
        # Initialize
        for u in range(n):
            for v in range(u+1, n):
                c = initializer(u, v)
                adj[u][v] = c
                adj[v][u] = c
        
        bad_counts = get_initial_bad_counts(adj)
        current_bad = total_bad(bad_counts)
        
        max_iterations = 5000
        no_improve_streak = 0
        max_no_improve = 100
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            if current_bad == 0:
                break
            
            # Try to fix a random bad cycle
            # Find a random bad cycle
            bad_cycle_indices = [i for i, c in enumerate(bad_counts) if c > 0]
            if not bad_cycle_indices:
                break
            target_cyc = random.choice(bad_cycle_indices)
            
            e_indices = cycle_edge_indices[target_cyc]
            
            # Try recoloring one of the edges in this cycle
            improved = False
            random.shuffle(e_indices)
            
            for e_idx in e_indices:
                u, v = idx_to_edge[e_idx]
                old_c = adj[u][v]
                
                if old_c == -1:
                    # Try to add this edge with a new color
                    for new_c in range(n):
                        adj[u][v] = new_c
                        adj[v][u] = new_c
                        
                        # Update bad counts for all cycles containing this edge
                        delta_bad = 0
                        affected_cycles = edge_cycles[e_idx]
                        # We need to check if this move increases or decreases bad count
                        # Save old states? No, just check before/after for affected cycles.
                        # But we already changed adj.
                        # Let's revert and check properly.
                        adj[u][v] = -1
                        adj[v][u] = -1
                        
                        # Check if adding edge (u,v) with color new_c reduces bad count
                        # 1. The target cycle might become good.
                        # 2. Other cycles containing (u,v) might become bad.
                        
                        # Since we are adding an edge, previously these cycles were not 4-cycles (missing edge).
                        # So they were not "bad" (they were "not cycles").
                        # Now they are cycles. If they are non-rainbow, bad count increases.
                        # If they are rainbow, bad count stays same (for those).
                        # For the target cycle, it was bad. If now it is rainbow, bad count decreases by 1.
                        
                        # Check target cycle
                        is_target_good = True
                        colors_target = []
                        for idx in e_indices:
                            uu, vv = idx_to_edge[idx]
                            cc = adj[uu][vv]
                            if idx == e_idx:
                                cc = new_c
                            if cc == -1:
                                is_target_good = False
                                break
                            colors_target.append(cc)
                        if is_target_good and len(set(colors_target)) < 4:
                            is_target_good = False # Still bad
                        # Wait, if it was bad and is still bad, delta is 0.
                        # If it was bad and is now good, delta is -1.
                        
                        # Check other cycles
                        new_bad_created = 0
                        for cyc_idx in affected_cycles:
                            if cyc_idx == target_cyc:
                                continue
                            # Check if this cycle becomes bad
                            # It wasn't a cycle before (missing e_idx).
                            # Now it is a cycle.
                            # If non-rainbow, it becomes bad.
                            if is_bad_cycle(adj, cyc_idx) is False: # adj has -1 for e_idx, so is_bad_cycle returns False for "not a cycle"
                                # We need to check if it *would* be bad with new_c
                                pass
                            
                        # This logic is getting complex. Let's use a simpler "try and revert" for small n.
                        # But we need speed.
                        
                        # Alternative: Just try recoloring existing edges first.
                        continue # Skip adding edges in this pass for simplicity, focus on recoloring
                
                # Try recoloring
                for new_c in range(n):
                    if new_c == old_c:
                        continue
                    
                    adj[u][v] = new_c
                    adj[v][u] = new_c
                    
                    # Calculate delta
                    delta = 0
                    affected_cycles = edge_cycles[e_idx]
                    
                    for cyc_idx in affected_cycles:
                        # Check old state (need to revert? No, we can calculate based on old_c and new_c)
                        # Actually, it's easier to just check if the cycle was bad before and is bad now.
                        # But we changed adj.
                        # Let's store old bad status for affected cycles.
                        pass
                    
                    # Revert and do proper check
                    adj[u][v] = old_c
                    adj[v][u] = old_c
                    
                    # Pre-calculate bad status of affected cycles with old_c
                    old_bad_statuses = {}
                    for cyc_idx in affected_cycles:
                        old_bad_statuses[cyc_idx] = 1 if is_bad_cycle(adj, cyc_idx) else 0
                        
                    # Apply new_c
                    adj[u][v] = new_c
                    adj[v][u] = new_c
                    
                    new_bad_statuses = {}
                    for cyc_idx in affected_cycles:
                        new_bad_statuses[cyc_idx] = 1 if is_bad_cycle(adj, cyc_idx) else 0
                    
                    delta = sum(new_bad_statuses.values()) - sum(old_bad_statuses.values())
                    
                    if delta < 0:
                        # Accept move
                        for cyc_idx in affected_cycles:
                            bad_counts[cyc_idx] = new_bad_statuses[cyc_idx]
                        current_bad += delta
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u][v] = old_c
                        adj[v][u] = old_c
                        
                if improved:
                    break
            
            if not improved:
                # Try to remove an edge from the bad cycle
                # Pick the edge in the cycle that is in the most bad cycles
                e_bad_counts = {}
                for e_idx in e_indices:
                    cnt = 0
                    for cyc_idx in edge_cycles[e_idx]:
                        if bad_counts[cyc_idx] > 0:
                            cnt += 1
                    e_bad_counts[e_idx] = cnt
                
                max_bad = -1
                worst_e_idx = None
                for e_idx in e_indices:
                    if e_bad_counts[e_idx] > max_bad:
                        max_bad = e_bad_counts[e_idx]
                        worst_e_idx = e_idx
                
                if worst_e_idx is not None:
                    u, v = idx_to_edge[worst_e_idx]
                    old_c = adj[u][v]
                    adj[u][v] = -1
                    adj[v][u] = -1
                    
                    # Update bad counts
                    for cyc_idx in edge_cycles[worst_e_idx]:
                        if bad_counts[cyc_idx] > 0:
                            # Cycle was bad. Now it's not a cycle, so not bad.
                            bad_counts[cyc_idx] = 0
                            current_bad -= 1
                    # Note: Removing an edge cannot create new bad cycles.
                    
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
                    if no_improve_streak > max_no_improve:
                        break
        
        return adj

    # Define initializers
    def init_random(u, v):
        return random.randint(0, n-1)
    
    def init_mod_sum(u, v):
        return (u + v) % n
        
    def init_structured_rand(u, v, offsets):
        return (offsets[u] + v) % n

    # Run multiple attempts
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        if attempt % 3 == 0:
            init_func = lambda u, v: random.randint(0, n-1)
        elif attempt % 3 == 1:
            init_func = lambda u, v: (u + v) % n
        else:
            offsets = list(range(n))
            random.shuffle(offsets)
            init_func = lambda u, v, offs=offsets: (offs[u] + v) % n
            
        adj = solve_attempt(init_func)
        
        if time.time() > deadline:
            break
            
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_edges = make_edges(adj)
            
    return best_edges
