import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    max_colours = 15
    
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1
            
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4_edges.append([
                        edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
                        edge_idx[(b, c)], edge_idx[(b, d)],
                        edge_idx[(c, d)]
                    ])
                    
    num_k4s = len(k4_edges)
    edge_k4s = [[] for _ in range(num_edges)]
    for k, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_k4s[e].append(k)
            
    def get_distinct_count(e_list, colours):
        return len(set(colours[e] for e in e_list))
        
    # State
    colours = [random.randint(0, max_colours - 1) for _ in range(num_edges)]
    k4_distinct = [get_distinct_count(e_list, colours) for e_list in k4_edges]
    bad_count = sum(1 for d in k4_distinct if d < 5)
    
    best_score = float('inf')
    best_solution = None
    
    # Helper to apply a move and return delta bad_count
    # We will manage bad_count globally
    
    def apply_move(e_idx, new_colour):
        nonlocal bad_count
        old_colour = colours[e_idx]
        if old_colour == new_colour:
            return
        
        affected = edge_k4s[e_idx]
        old_states = {}
        for k in affected:
            old_states[k] = k4_distinct[k]
            
        colours[e_idx] = new_colour
        
        for k in affected:
            nd = get_distinct_count(k4_edges[k], colours)
            od = old_states[k]
            if od < 5 and nd >= 5:
                bad_count -= 1
            elif od >= 5 and nd < 5:
                bad_count += 1
            k4_distinct[k] = nd
            
    def revert_move(e_idx, old_colour, affected, old_states):
        nonlocal bad_count
        colours[e_idx] = old_colour
        for k in affected:
            od = k4_distinct[k]
            nd = old_states[k]
            if od < 5 and nd >= 5:
                bad_count += 1
            elif od >= 5 and nd < 5:
                bad_count -= 1
            k4_distinct[k] = nd

    # Main Loop
    while time.time() < deadline:
        if bad_count == 0:
            # Check best
            used = set(colours)
            if len(used) < best_score:
                best_score = len(used)
                best_solution = colours.copy()
                if best_score <= 13:
                    break
            
            # Try to reduce colours
            if len(used) > 13:
                # Pick a colour to remove
                c_target = random.choice(list(used))
                
                # Local search to minimize count(c_target) while keeping bad_count=0
                # We will allow temporary increases in bad_count if it helps reduce count?
                # No, keep bad_count=0 strictly.
                
                # Find edges with c_target
                candidates = [i for i, c in enumerate(colours) if c == c_target]
                
                # Try to recolor them
                for _ in range(200):
                    if time.time() > deadline: break
                    if not candidates:
                        break
                    e_idx = random.choice(candidates)
                    old_c = colours[e_idx]
                    # Pick a new colour from used \ {c_target}
                    other_colours = [c for c in used if c != c_target]
                    if not other_colours:
                        break
                    new_c = random.choice(other_colours)
                    
                    affected = edge_k4s[e_idx]
                    old_states = {k: k4_distinct[k] for k in affected}
                    
                    # Apply
                    colours[e_idx] = new_c
                    new_bad = bad_count
                    for k in affected:
                        nd = get_distinct_count(k4_edges[k], colours)
                        od = old_states[k]
                        if od < 5 and nd >= 5: new_bad -= 1
                        elif od >= 5 and nd < 5: new_bad += 1
                        k4_distinct[k] = nd
                        
                    if new_bad == 0:
                        # Accept
                        bad_count = 0
                        # Update candidates and used
                        # Remove e_idx from candidates
                        candidates.remove(e_idx)
                        # If new_c was not in used, add it? No, it is from used.
                        # If old_c count becomes 0, remove from used
                        # This bookkeeping is tedious. 
                        # Just recalculate used every time we accept?
                        # Or just check at the end of the inner loop.
                    else:
                        # Revert
                        colours[e_idx] = old_c
                        for k in affected:
                            k4_distinct[k] = old_states[k]
        else:
            # Reduce bad_count
            e_idx = random.randint(0, num_edges - 1)
            old_c = colours[e_idx]
            new_c = random.randint(0, max_colours - 1)
            
            affected = edge_k4s[e_idx]
            old_states = {k: k4_distinct[k] for k in affected}
            
            colours[e_idx] = new_c
            new_bad = bad_count
            for k in affected:
                nd = get_distinct_count(k4_edges[k], colours)
                od = old_states[k]
                if od < 5 and nd >= 5: new_bad -= 1
                elif od >= 5 and nd < 5: new_bad += 1
                k4_distinct[k] = nd
                
            if new_bad < bad_count:
                bad_count = new_bad
            else:
                # Revert
                colours[e_idx] = old_c
                for k in affected:
                    k4_distinct[k] = old_states[k]

    if best_solution:
        return best_solution
    else:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
