import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 20
    # Generate all edges in K_20
    edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(edges)
            edges.append((u, v))
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx
    m = len(edges) # 190
    
    # Precompute all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a permutation of edges.
    # In a simple graph, a 4-cycle on vertices a,b,c,d can be:
    # (a,b), (b,c), (c,d), (d,a)
    # (a,c), (c,b), (b,d), (d,a) -> (a,c), (b,c), (b,d), (a,d)
    # (a,d), (d,c), (c,b), (b,a) -> (a,d), (c,d), (b,c), (a,b)
    # Basically, for any 4 vertices, there are 3 distinct 4-cycles (chords).
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    
    cycles = []
    # Map vertex to list of incident edge indices? No, just iterate vertices.
    vertices = list(range(n))
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        # Get edge indices
        # Check if edges exist in the complete graph (they do)
        # We need to represent the cycle as a set of 4 edge indices
        
        # Cycle 1: a-b, b-c, c-d, d-a
        e_ab = edge_to_idx.get((a,b))
        e_bc = edge_to_idx.get((b,c))
        e_cd = edge_to_idx.get((c,d))
        e_da = edge_to_idx.get((d,a))
        if None not in (e_ab, e_bc, e_cd, e_da):
            cycles.append((e_ab, e_bc, e_cd, e_da))
            
        # Cycle 2: a-b, b-d, d-c, c-a
        e_bd = edge_to_idx.get((b,d))
        e_dc = edge_to_idx.get((d,c))
        e_ca = edge_to_idx.get((c,a))
        if None not in (e_ab, e_bd, e_dc, e_ca):
            cycles.append((e_ab, e_bd, e_dc, e_ca))
            
        # Cycle 3: a-c, c-b, b-d, d-a
        e_ac = edge_to_idx.get((a,c))
        e_cb = edge_to_idx.get((c,b))
        e_db = edge_to_idx.get((d,b))
        e_ad = edge_to_idx.get((d,a))
        if None not in (e_ac, e_cb, e_db, e_ad):
            cycles.append((e_ac, e_cb, e_db, e_ad))
            
    num_cycles = len(cycles)
    
    # Helper to check if a cycle is rainbow given current colours and active status
    def is_rainbow(cycle):
        c1 = colours[cycle[0]]
        c2 = colours[cycle[1]]
        c3 = colours[cycle[2]]
        c4 = colours[cycle[3]]
        return len({c1, c2, c3, c4}) == 4

    # Helper to get all violating cycles
    # Optimization: We can maintain a count of violations per edge or just scan.
    # Scanning 13k cycles is fast in Python if vectorized or simple loops.
    # Let's just scan.
    
    def get_violations():
        v_list = []
        for c in cycles:
            if not active[c[0]] or not active[c[1]] or not active[c[2]] or not active[c[3]]:
                continue
            if not is_rainbow(c):
                v_list.append(c)
        return v_list

    def try_add_edge(edge_idx, colour):
        # Temporarily add
        active[edge_idx] = True
        colours[edge_idx] = colour
        
        # Check only cycles containing edge_idx
        # Precompute edge_to_cycles for speed? 
        # Let's do a quick check on all cycles for now, or build index.
        # Building index is better.
        
        # For speed, we will precompute edge_to_cycles
        pass

    # Precompute edge_to_cycles
    edge_to_cycles = {i: [] for i in range(m)}
    for c in cycles:
        for e in c:
            edge_to_cycles[e].append(c)

    def check_edge_validity(edge_idx, colour):
        # Check all cycles containing edge_idx
        for c in edge_to_cycles[edge_idx]:
            if not active[c[0]] or not active[c[1]] or not active[c[2]] or not active[c[3]]:
                continue
            # Temporarily set colour
            old_colour = colours[edge_idx]
            colours[edge_idx] = colour
            if not is_rainbow(c):
                colours[edge_idx] = old_colour
                return False
        return True

    def repair_and_optimize():
        # State
        active = [False] * m
        colours = [0] * m
        
        # Initialize with a random subset of edges
        # Let's start with ~40% density
        init_density = 0.4
        num_init = int(m * init_density)
        init_edges = random.sample(range(m), num_init)
        for e in init_edges:
            active[e] = True
            colours[e] = random.randint(0, 19)
            
        # Repair phase
        # Iterate until no violations or time limit
        while time.time() < deadline - 10:
            violations = get_violations()
            if not violations:
                break
            
            # Pick a random violation
            v_cycle = random.choice(violations)
            
            # Try to fix by changing a colour of one of the edges in the cycle
            fixed = False
            for e in v_cycle:
                if not active[e]:
                    continue
                # Try all colours for this edge
                for new_c in range(20):
                    if new_c == colours[e]:
                        continue
                    # Check if changing colour e to new_c fixes this cycle and breaks no others
                    # We need to check all cycles containing e
                    old_c = colours[e]
                    colours[e] = new_c
                    
                    valid = True
                    for c in edge_to_cycles[e]:
                        if not active[c[0]] or not active[c[1]] or not active[c[2]] or not active[c[3]]:
                            continue
                        if not is_rainbow(c):
                            valid = False
                            break
                    
                    if valid:
                        fixed = True
                        break
                    else:
                        colours[e] = old_c
                
                if fixed:
                    break
            
            if not fixed:
                # If no colour change works, remove a random edge from the cycle
                e_to_remove = random.choice(v_cycle)
                active[e_to_remove] = False

        # Optimization phase: Try to add more edges
        while time.time() < deadline - 5:
            # Find inactive edges
            inactive_edges = [i for i in range(m) if not active[i]]
            if not inactive_edges:
                break
            
            random.shuffle(inactive_edges)
            added_any = False
            for e in inactive_edges:
                if time.time() > deadline - 5:
                    break
                # Try to find a valid colour for e
                for c in range(20):
                    if check_edge_validity(e, c):
                        active[e] = True
                        colours[e] = c
                        added_any = True
                        break
            
            if not added_any:
                # Stuck. Try some random swaps or colour shuffles?
                # Simple shuffle of colours in active edges
                active_idx = [i for i in range(m) if active[i]]
                if active_idx:
                    # Randomly shuffle colours of a subset
                    subset = random.sample(active_idx, min(10, len(active_idx)))
                    cols = [colours[i] for i in subset]
                    random.shuffle(cols)
                    for i, c in zip(subset, cols):
                        colours[i] = c
                    # Repair again briefly
                    for _ in range(5):
                        violations = get_violations()
                        if not violations:
                            break
                        v_cycle = random.choice(violations)
                        fixed = False
                        for e in v_cycle:
                            if not active[e]: continue
                            for new_c in range(20):
                                if new_c == colours[e]: continue
                                old_c = colours[e]
                                colours[e] = new_c
                                valid = True
                                for c2 in edge_to_cycles[e]:
                                    if not active[c2[0]] or not active[c2[1]] or not active[c2[2]] or not active[c2[3]]: continue
                                    if not is_rainbow(c2):
                                        valid = False
                                        break
                                if valid:
                                    fixed = True
                                    break
                                else:
                                    colours[e] = old_c
                            if fixed: break
                        if not fixed:
                            e_to_remove = random.choice(v_cycle)
                            active[e_to_remove] = False
                # If still stuck, break to restart or just return
                # To avoid infinite loop, we break out of the add loop if no progress
                break
        
        # Return current best
        result = []
        for i in range(m):
            if active[i]:
                u, v = edges[i]
                result.append([u, v, colours[i]])
        return result

    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        res = repair_and_optimize()
        count = len(res)
        if count > best_count:
            best_count = count
            best_edges = res
            
        # If we reached a very high density (e.g., > 80% of 190 = 152), we might be close to optimal
        # But let's keep searching until deadline
        
    return best_edges
# EVOLVE-BLOCK-END
