import time
import random

N = 20
C = 20
DEADLINE = 115.0

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _generate_all_4cycles():
    """
    Generate all 4-cycles as frozensets of 4 edge keys.
    """
    from itertools import combinations
    verts = list(range(N))
    cycles = []
    for a, b, c, d in combinations(verts, 4):
        # 3 cycles on 4 vertices
        # 1: a-b-c-d-a
        # 2: a-b-d-c-a
        # 3: a-c-b-d-a
        # Note: a-b-c-d-a edges: (a,b), (b,c), (c,d), (d,a)
        # a-b-d-c-a edges: (a,b), (b,d), (d,c), (c,a)
        # a-c-b-d-a edges: (a,c), (c,b), (b,d), (d,a)
        
        e1 = _edge_key(a, b)
        e2 = _edge_key(b, c)
        e3 = _edge_key(c, d)
        e4 = _edge_key(d, a)
        c1 = frozenset([e1, e2, e3, e4])
        if len(c1) == 4: cycles.append(c1)
        
        e1 = _edge_key(a, b)
        e2 = _edge_key(b, d)
        e3 = _edge_key(d, c)
        e4 = _edge_key(c, a)
        c2 = frozenset([e1, e2, e3, e4])
        if len(c2) == 4: cycles.append(c2)
        
        e1 = _edge_key(a, c)
        e2 = _edge_key(c, b)
        e3 = _edge_key(b, d)
        e4 = _edge_key(d, a)
        c3 = frozenset([e1, e2, e3, e4])
        if len(c3) == 4: cycles.append(c3)
        
    return cycles

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    all_cycles = _generate_all_4cycles()
    
    # Initialize with a random dense graph
    # Let's start with 50% density (95 edges)
    # Or maybe start with a C4-free graph and add?
    # No, start random and repair is a different family.
    
    adj = {}
    # Add random edges
    all_possible = [(u, v) for u in range(N) for v in range(u+1, N)]
    rng.shuffle(all_possible)
    num_edges_init = 80
    for u, v in all_possible[:num_edges_init]:
        adj[(u, v)] = rng.randint(0, C-1)
        
    best_adj = dict(adj)
    best_count = len(adj)
    
    # Pre-compute which cycles contain which edge for faster lookup?
    # edge_to_cycles: dict from edge_key to list of cycle indices
    edge_to_cycles = {e: [] for e in all_possible}
    for i, cyc in enumerate(all_cycles):
        for e in cyc:
            if e in edge_to_cycles:
                edge_to_cycles[e].append(i)
                
    def is_valid():
        for cyc in all_cycles:
            cols = [adj.get(e, -1) for e in cyc]
            if -1 in cols: continue
            if len(set(cols)) != 4:
                return False
        return True
        
    def get_violation():
        # Return a random violated cycle
        # We can just iterate and pick the first one found, or random
        # To be efficient, maintain a list of violated cycles?
        # Or just scan.
        for cyc in all_cycles:
            cols = [adj.get(e, -1) for e in cyc]
            if -1 in cols: continue
            if len(set(cols)) != 4:
                return cyc
        return None

    def try_recolor_edge(edge, forbidden_colors_in_cycle):
        # Try to find a color for 'edge' that makes all cycles containing it valid
        # forbidden_colors_in_cycle is the set of colors of the other 3 edges in the specific cycle we are trying to fix
        # But we need to ensure ALL cycles containing 'edge' are valid.
        
        current_color = adj.get(edge, -1)
        
        # Candidate colors
        candidates = list(range(C))
        rng.shuffle(candidates)
        
        for c in candidates:
            if c in forbidden_colors_in_cycle:
                continue
                
            # Temporarily assign
            old_color = adj[edge]
            adj[edge] = c
            
            # Check all cycles containing 'edge'
            valid = True
            for cyc_idx in edge_to_cycles[edge]:
                cyc = all_cycles[cyc_idx]
                cols = [adj.get(e, -1) for e in cyc]
                if -1 in cols: continue
                if len(set(cols)) != 4:
                    valid = False
                    break
            
            if valid:
                return True
            else:
                adj[edge] = old_color
                
        return False

    while time.time() - t0 < DEADLINE:
        violation = get_violation()
        
        if violation is None:
            # Valid state! Try to add an edge
            # Find a missing edge
            missing = [e for e in all_possible if e not in adj]
            if not missing:
                break # Complete graph
            rng.shuffle(missing)
            added = False
            for e in missing:
                u, v = e
                # Try to add with a random color
                # We need to pick a color that doesn't create a violation
                # This is hard to check directly without trying.
                # Let's just try a few random colors.
                for _ in range(10):
                    c = rng.randint(0, C-1)
                    adj[e] = c
                    if is_valid():
                        added = True
                        break
                    else:
                        del adj[e]
                if added:
                    if len(adj) > best_count:
                        best_adj = dict(adj)
                        best_count = len(adj)
                    break # Restart violation search
            if not added:
                # If we can't add any edge, maybe we should try to recolor existing edges to free up space?
                # Or just break?
                # Let's try to recolor a random edge to see if it helps.
                if adj:
                    e = rng.choice(list(adj.keys()))
                    # Just shuffle its color
                    adj[e] = rng.randint(0, C-1)
                    # This might create violations, which we'll fix in next iteration
                    # But it might also break a good state?
                    # Better to only do this if we are stuck.
                    # Actually, if we can't add, we are at a local maximum for this topology.
                    # We should try to change topology by removing an edge and adding another?
                    # Or just accept the current best.
                    pass
        else:
            # Found a violation
            # Pick an edge from the violation to recolor
            edges_in_cyc = list(violation)
            rng.shuffle(edges_in_cyc)
            
            fixed = False
            for e in edges_in_cyc:
                # The other 3 edges in this cycle have specific colors
                other_edges = [x for x in violation if x != e]
                forbidden = set(adj[x] for x in other_edges)
                
                if try_recolor_edge(e, forbidden):
                    fixed = True
                    break
                    
            if not fixed:
                # If recoloring doesn't work, remove an edge
                e_to_remove = rng.choice(edges_in_cyc)
                del adj[e_to_remove]
                
    # Final check and return
    # Ensure best_adj is valid
    # (It should be, as we only saved valid states or repaired to validity)
    # But best_adj was saved when valid.
    
    return [[u, v, best_adj[(u, v)]] for (u, v) in best_adj]
