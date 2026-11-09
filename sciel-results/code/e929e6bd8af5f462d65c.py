import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflicts and compatible lists
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Precompute compatible list (indices not conflicting with i)
    compatible = [set(range(n)) - {i} - conflict[i] for i in range(n)]
    
    # Precompute reciprocals
    recips = [1.0 / numbers[i] for i in range(n)]
    
    def reciprocal_sum(indices):
        return sum(recips[i] for i in indices)
    
    def elapsed():
        return time.time() - start_time
    
    # Greedy: try multiple orderings
    best_selected = set()
    best_sum = 0.0
    
    # Ordering 1: ascending a
    order1 = sorted(range(n), key=lambda i: numbers[i])
    selected = set()
    for i in order1:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    # Ordering 2: descending a (start with large numbers, fewer conflicts)
    order2 = sorted(range(n), key=lambda i: -numbers[i])
    selected = set()
    for i in order2:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    # Ordering 3: by 1/a / degree (weight per conflict)
    order3 = sorted(range(n), key=lambda i: (1.0 / numbers[i]) / (len(conflict[i]) + 1), reverse=True)
    selected = set()
    for i in order3:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    # Ordering 4: by 1/a * (n - degree) - prefer high weight, low conflict
    order4 = sorted(range(n), key=lambda i: recips[i] * (n - len(conflict[i])), reverse=True)
    selected = set()
    for i in order4:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    # Ordering 5: only numbers > 128 first (they have fewer conflicts), then fill
    order5 = sorted(range(n), key=lambda i: (numbers[i] <= 128, -numbers[i]))
    selected = set()
    for i in order5:
        if not (conflict[i] & selected):
            selected.add(i)
    s = reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = set(selected)
    
    selected = set(best_selected)
    
    # Local search with multiple move types
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        
        # Phase 1: Try single swap: remove i, add j
        for i in list(selected):
            if elapsed() >= DEADLINE:
                break
            new_sel = selected - {i}
            # Find best j to add: compatible with all of new_sel
            for j in range(n):
                if j in new_sel:
                    continue
                if not (conflict[j] & new_sel):
                    delta_s = recips[j] - recips[i]
                    if delta_s > 1e-12:
                        cand = new_sel | {j}
                        s = reciprocal_sum(cand)
                        if s > best_sum + 1e-12:
                            selected = cand
                            best_sum = s
                            best_selected = set(selected)
                            improved = True
                            break
        
        # Phase 2: Try double swap: remove i, add j, add k (where j,k are compatible with each other and new_sel)
        if not improved and elapsed() < DEADLINE:
            for i in list(selected):
                if elapsed() >= DEADLINE:
                    break
                new_sel = selected - {i}
                # Find pairs (j, k) that can both be added
                candidates = [j for j in range(n) if j not in new_sel and not (conflict[j] & new_sel)]
                # Sort by reciprocal descending
                candidates.sort(key=lambda j: -recips[j])
                found = False
                for idx_j in range(len(candidates)):
                    j = candidates[idx_j]
                    for idx_k in range(idx_j + 1, len(candidates)):
                        k = candidates[idx_k]
                        # Check j and k are compatible with each other
                        if j not in conflict[k]:
                            cand = new_sel | {j, k}
                            s = reciprocal_sum(cand)
                            if s > best_sum + 1e-12:
                                selected = cand
                                best_sum = s
                                best_selected = set(selected)
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
        
        # Phase 3: Try triple swap: remove i, add j, add k, add l
        if not improved and elapsed() < DEADLINE:
            for i in list(selected):
                if elapsed() >= DEADLINE:
                    break
                new_sel = selected - {i}
                candidates = [j for j in range(n) if j not in new_sel and not (conflict[j] & new_sel)]
                candidates.sort(key=lambda j: -recips[j])
                # Try adding up to 3 candidates greedily
                found = False
                for count in [3, 2]:
                    if found:
                        break
                    # Try all combinations of 'count' from top candidates
                    top_cands = candidates[:min(10, len(candidates))]
                    for combo_start in range(len(top_cands)):
                        if elapsed() >= DEADLINE:
                            break
                        # Greedily try to add 'count' elements starting from combo_start
                        trial_sel = set(new_sel)
                        added = 0
                        for j in top_cands[combo_start:]:
                            if added >= count:
                                break
                            if not (conflict[j] & trial_sel):
                                trial_sel.add(j)
                                added += 1
                        if added >= 2:  # Need at least 2 to offset removing 1
                            s = reciprocal_sum(trial_sel)
                            if s > best_sum + 1e-12:
                                selected = trial_sel
                                best_sum = s
                                best_selected = set(selected)
                                improved = True
                                found = True
                                break
        
        # Phase 4: Try removing i and j, adding k and l
        if not improved and elapsed() < DEADLINE:
            sel_list = list(selected)
            for ii in range(len(sel_list)):
                if elapsed() >= DEADLINE:
                    break
                i = sel_list[ii]
                for jj in range(ii + 1, len(sel_list)):
                    j = sel_list[jj]
                    new_sel = selected - {i, j}
                    candidates = [x for x in range(n) if x not in new_sel and not (conflict[x] & new_sel)]
                    candidates.sort(key=lambda x: -recips[x])
                    # Try adding 2 or 3 elements
                    for count in [3, 2]:
                        trial_sel = set(new_sel)
                        added = 0
                        for x in candidates:
                            if added >= count:
                                break
                            if not (conflict[x] & trial_sel):
                                trial_sel.add(x)
                                added += 1
                        if added >= 2:
                            s = reciprocal_sum(trial_sel)
                            if s > best_sum + 1e-12:
                                selected = trial_sel
                                best_sum = s
                                best_selected = set(selected)
                                improved = True
                                break
                    if improved:
                        break
        
        # Phase 5: Multi-add without removal (add as many as possible)
        if not improved and elapsed() < DEADLINE:
            candidates = [j for j in range(n) if j not in selected and not (conflict[j] & selected)]
            candidates.sort(key=lambda j: -recips[j])
            for j in candidates:
                if not (conflict[j] & selected):
                    cand = selected | {j}
                    s = reciprocal_sum(cand)
                    if s > best_sum + 1e-12:
                        selected = cand
                        best_sum = s
                        best_selected = set(selected)
                        improved = True
                        break
    
    # Final polish: try adding any single element that fits
    if elapsed() < DEADLINE - 5.0:
        for j in range(n):
            if j not in selected and not (conflict[j] & selected):
                cand = selected | {j}
                s = reciprocal_sum(cand)
                if s > best_sum + 1e-12:
                    selected = cand
                    best_sum = s
                    best_selected = set(selected)
    
    return [numbers[i] for i in best_selected]
