import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute 4-cycles and edge-to-cycle mapping
    cycles = []
    edge_to_cycles = {}
    
    for combo in itertools.combinations(range(n), 4):
        v1, v2, v3, v4 = combo
        
        # 3 cycles
        c1 = [(v1,v2), (v2,v3), (v3,v4), (v4,v1)]
        idx1 = len(cycles)
        cycles.append(c1)
        for e in c1:
            if e not in edge_to_cycles: edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx1)
            
        c2 = [(v1,v2), (v2,v4), (v4,v3), (v3,v1)]
        idx2 = len(cycles)
        cycles.append(c2)
        for e in c2:
            if e not in edge_to_cycles: edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx2)
            
        c3 = [(v1,v3), (v3,v2), (v2,v4), (v4,v1)]
        idx3 = len(cycles)
        cycles.append(c3)
        for e in c3:
            if e not in edge_to_cycles: edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx3)
            
    num_cycles = len(cycles)
    
    # Helper to check if a cycle is rainbow given a coloring dict
    def is_rainbow(c, coloring):
        c0 = coloring[c[0]]
        c1 = coloring[c[1]]
        c2 = coloring[c[2]]
        c3 = coloring[c[3]]
        return len({c0, c1, c2, c3}) == 4

    best_edges = [[0, v, 0] for v in range(1, n)]
    best_count = 19
    
    # Local Search on K_n
    # State: coloring dict, violated_cycles set (or list)
    
    for attempt in range(50): # Try a few random restarts
        if time.time() > deadline:
            break
            
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        coloring = {e: random.randint(0, 19) for e in edges}
        
        # Initial violation check
        violated = set()
        for i, c in enumerate(cycles):
            if not is_rainbow(c, coloring):
                violated.add(i)
                
        if not violated:
            res = [[u, v, coloring[(u,v)]] for u, v in edges]
            return res
            
        # Local search
        for step in range(5000):
            if time.time() > deadline:
                break
                
            if not violated:
                res = [[u, v, coloring[(u,v)]] for u, v in edges]
                if len(res) > best_count:
                    best_count = len(res)
                    best_edges = res
                return best_edges
                
            # Pick a random violated cycle
            v_idx = random.choice(list(violated))
            c = cycles[v_idx]
            
            # Try to recolor one of the edges in c to fix it
            fixed = False
            random.shuffle(c)
            for e in c:
                old_c = coloring[e]
                # Try each color
                for new_c in range(20):
                    if new_c == old_c:
                        continue
                    coloring[e] = new_c
                    
                    # Check if v_idx is fixed
                    if is_rainbow(c, coloring):
                        # Check if we broke any other cycles
                        # Only cycles containing e are affected
                        broken = False
                        for other_idx in edge_to_cycles[e]:
                            if other_idx == v_idx:
                                continue
                            if not is_rainbow(cycles[other_idx], coloring):
                                # This move breaks another cycle
                                # We can still accept it if it reduces total violations?
                                # For simplicity, let's reject if it breaks any new one.
                                # Or accept and update violated set.
                                broken = True
                                break
                            
                        if not broken:
                            # Update violated set
                            violated.discard(v_idx)
                            # Other cycles containing e were already valid and remain valid?
                            # We checked them. If they were in violated, they might be fixed?
                            # No, we only checked if they are rainbow.
                            # If they were violated, and now they are rainbow, we should remove them from violated.
                            # If they were valid, and now they are violated, we should add them.
                            # My "broken" check only looked at if they are NOT rainbow.
                            # So if broken is False, all other cycles containing e are rainbow.
                            # So we just remove v_idx from violated.
                            # Wait, what if other cycles containing e were in violated?
                            # If they are now rainbow, they should be removed.
                            # My loop `if not is_rainbow` sets broken=True.
                            # So if broken is False, all other cycles containing e ARE rainbow.
                            # So any of them that were in violated are now fixed.
                            for other_idx in edge_to_cycles[e]:
                                violated.discard(other_idx)
                                
                            fixed = True
                            break
                        else:
                            # Revert
                            coloring[e] = old_c
                            # Don't break, try next color
                            continue
                    else:
                        # Cycle v_idx not fixed, revert
                        coloring[e] = old_c
                        continue
                
                if fixed:
                    break
            
            if not fixed:
                # Random kick
                e = random.choice(edges)
                old_c = coloring[e]
                new_c = random.randint(0, 19)
                coloring[e] = new_c
                # Update violated set for cycles containing e
                for idx in edge_to_cycles[e]:
                    if is_rainbow(cycles[idx], coloring):
                        violated.discard(idx)
                    else:
                        violated.add(idx)

    return best_edges
