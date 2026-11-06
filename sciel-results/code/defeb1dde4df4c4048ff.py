# EVOLVE-BLOCK-START
def construct():
    import random
    from itertools import combinations

    n = 33
    
    triples = list(combinations(range(n), 3))
    triple_index = {}
    for i, (a, b, c) in enumerate(triples):
        triple_index[(a, b, c)] = i
    
    num_triples = len(triples)
    
    all_quads = list(combinations(range(n), 4))
    all_quints = list(combinations(range(n), 5))
    
    quad_index = {q: i for i, q in enumerate(all_quads)}
    quint_index = {q: i for i, q in enumerate(all_quints)}
    
    red_count = [0] * len(all_quads)
    blue_count = [0] * len(all_quints)
    
    triple_to_quads = [[] for _ in range(num_triples)]
    triple_to_quints = [[] for _ in range(num_triples)]
    
    for i, (a, b, c) in enumerate(triples):
        for x in range(n):
            if x in (a, b, c): continue
            quad = tuple(sorted((a, b, c, x)))
            triple_to_quads[i].append(quad_index[quad])
        for x in range(n):
            if x in (a, b, c): continue
            for y in range(x+1, n):
                if y in (a, b, c): continue
                quint = tuple(sorted((a, b, c, x, y)))
                triple_to_quints[i].append(quint_index[quint])
                
    # Initialization: (a+b+c) % 2
    colors = [0 if (a+b+c) % 2 == 0 else 1 for a, b, c in triples]
    
    # Initialize counts
    for i, (a, b, c) in enumerate(triples):
        if colors[i] == 0:
            for qi in triple_to_quads[i]:
                red_count[qi] += 1
        else:
            for qi in triple_to_quints[i]:
                blue_count[qi] += 1
                
    # Track violations
    # total_violations = count of quads with red_count==4 + count of quints with blue_count==10
    total_violations = 0
    for rc in red_count:
        if rc == 4: total_violations += 1
    for bc in blue_count:
        if bc == 10: total_violations += 1

    def flip(i):
        nonlocal total_violations
        old = colors[i]
        new = 1 - old
        colors[i] = new
        
        if old == 0: # Red -> Blue
            # Update Red Quads
            for qi in triple_to_quads[i]:
                old_rc = red_count[qi]
                red_count[qi] = old_rc - 1
                if old_rc == 4:
                    total_violations -= 1
                elif old_rc - 1 == 4: # Impossible, rc goes down
                    pass
            # Update Blue Quints
            for qi in triple_to_quints[i]:
                old_bc = blue_count[qi]
                blue_count[qi] = old_bc + 1
                if old_bc == 10:
                    total_violations -= 1 # Was violation, now 11? No, 10->11 is not violation. 
                    # Wait. Violation is EXACTLY 10? 
                    # "no 5 vertices may have all ten triples blue". 
                    # Yes, exactly 10 blue. If it becomes 11, it's impossible (max 10).
                    # So if old_bc == 10, it was a violation. Now it is 11? No, max is 10.
                    # Ah, blue_count counts blue edges. Max is 10.
                    # So if old_bc == 10, it was a violation. New is 11? No, can't be 11.
                    # Wait, if I flip Red->Blue, blue_count increases.
                    # If it was 9, it becomes 10. New violation!
                    # If it was 10, it becomes 11? No, only 10 edges in a quint.
                    # So if old_bc == 10, it was a violation. Flipping one of its edges to Blue... 
                    # Wait, if it's all blue (10), and I flip one to Red, blue_count becomes 9. Violation removed.
                    # If I flip a Red edge to Blue in a quint that has 9 Blue edges, it becomes 10. Violation created.
                    
                    # Let's re-evaluate logic.
                    # Violation: blue_count == 10.
                    # If old_bc == 10, it is a violation.
                    # If I flip an edge IN this quint from Red to Blue:
                    # This edge was Red, so it contributed 0 to blue_count.
                    # Now it is Blue, so it contributes 1.
                    # So blue_count becomes 11? No, max 10.
                    # This implies the edge was NOT blue. So the quint had 9 Blue edges and 1 Red edge?
                    # No, if blue_count == 10, ALL edges are Blue.
                    # So I cannot flip a Red edge to Blue in a quint with blue_count==10, because there are no Red edges.
                    # So this case (old_bc==10 and flipping R->B in this quint) is impossible.
                    
                    # So:
                    # If old_bc == 9, new_bc = 10. New violation. total_violations += 1.
                    # If old_bc == 10, new_bc = 11 (Impossible).
                    # If old_bc < 9, no change in violation status.
                    
                    if old_bc == 9:
                        total_violations += 1
                        # if new_bc == 10, violation created
                        # if old_bc == 10, violation removed (but impossible as discussed)
                        
        else: # Blue -> Red
            # Update Blue Quints
            for qi in triple_to_quints[i]:
                old_bc = blue_count[qi]
                blue_count[qi] = old_bc - 1
                if old_bc == 10:
                    total_violations -= 1 # Violation removed
            # Update Red Quads
            for qi in triple_to_quads[i]:
                old_rc = red_count[qi]
                red_count[qi] = old_rc + 1
                if old_rc == 3:
                    total_violations += 1 # New violation (4 red)
                # If old_rc == 4, new_rc = 5 (Impossible)
                
    # Local Search
    max_iter = 10000
    for _ in range(max_iter):
        if total_violations == 0:
            break
            
        # Find a violating quad
        violating_quad = -1
        for qi, rc in enumerate(red_count):
            if rc == 4:
                violating_quad = qi
                break
                
        if violating_quad != -1:
            quad = all_quads[violating_quad]
            a, b, c, d = quad
            quad_triples = [
                triple_index[(a,b,c)],
                triple_index[(a,b,d)],
                triple_index[(a,c,d)],
                triple_index[(b,c,d)]
            ]
            
            # Try flipping each
            best_i = -1
            best_vc = total_violations
            
            for ti in quad_triples:
                flip(ti)
                if total_violations < best_vc:
                    best_vc = total_violations
                    best_i = ti
                else:
                    flip(ti) # Undo
                    continue
                flip(ti) # Undo for next iteration
                
            if best_i != -1:
                flip(best_i)
            else:
                # No improvement, pick random to escape
                flip(quad_triples[random.randint(0, 3)])
                
        else:
            # Find violating quint
            violating_quint = -1
            for qi, bc in enumerate(blue_count):
                if bc == 10:
                    violating_quint = qi
                    break
                    
            if violating_quint != -1:
                quint = all_quints[violating_quint]
                quint_triples = [triple_index[t] for t in combinations(quint, 3)]
                
                best_i = -1
                best_vc = total_violations
                
                for ti in quint_triples:
                    flip(ti)
                    if total_violations < best_vc:
                        best_vc = total_violations
                        best_i = ti
                    else:
                        flip(ti)
                        continue
                    flip(ti)
                    
                if best_i != -1:
                    flip(best_i)
                else:
                    flip(quint_triples[random.randint(0, 9)])
            else:
                break

    def verify():
        return total_violations == 0

    if verify():
        return {'n': n, 'colors': colors}
    else:
        n_small = 9
        return {'n': n_small, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_small), 3)]}
# EVOLVE-BLOCK-END
