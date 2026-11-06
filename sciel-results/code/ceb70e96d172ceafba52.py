"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    N = 256
    weights = [0.0] + [1.0 / x for x in range(1, N + 1)]
    
    # Suffix sums
    suffix = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
    
    best_set = None
    best_weight = -1.0
    
    def greedy_complete(selected, forbidden, start_idx):
        selected = set(selected)
        forbidden = set(forbidden)
        for x in range(start_idx, N + 1):
            if x in forbidden or x in selected:
                continue
            valid = True
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    a, b = sel_list[i], sel_list[j]
                    triple = sorted([x, a, b])
                    if triple[0] + triple[2] == 2 * triple[1]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.add(x)
                for u in sel_list:
                    r1 = 2 * x - u
                    r2 = 2 * u - x
                    if 1 <= r1 <= N and r1 not in selected:
                        forbidden.add(r1)
                    if 1 <= r2 <= N and r2 not in selected:
                        forbidden.add(r2)
                    if (x + u) % 2 == 0:
                        r3 = (x + u) // 2
                        if 1 <= r3 <= N and r3 not in selected:
                            forbidden.add(r3)
        return selected
    
    K = 12
    
    def branch(idx, selected, forbidden):
        nonlocal best_set, best_weight
        
        if time.time() - start_time > DEADLINE:
            return
        
        if idx > K:
            completed = greedy_complete(selected, forbidden, K + 1)
            w = sum(weights[x] for x in completed)
            if w > best_weight:
                best_weight = w
                best_set = list(completed)
            return
        
        # Upper bound
        current_w = sum(weights[x] for x in selected)
        ub = current_w
        for x in range(idx, N + 1):
            if x not in forbidden:
                ub += weights[x]
        if ub <= best_weight:
            return
        
        x = idx
        
        # Option 1: Include x
        if x not in forbidden:
            valid = True
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    a, b = sel_list[i], sel_list[j]
                    triple = sorted([x, a, b])
                    if triple[0] + triple[2] == 2 * triple[1]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                new_selected = selected | {x}
                new_forbidden = set(forbidden)
                for u in sel_list:
                    r1 = 2 * x - u
                    r2 = 2 * u - x
                    if 1 <= r1 <= N and r1 not in new_selected:
                        new_forbidden.add(r1)
                    if 1 <= r2 <= N and r2 not in new_selected:
                        new_forbidden.add(r2)
                    if (x + u) % 2 == 0:
                        r3 = (x + u) // 2
                        if 1 <= r3 <= N and r3 not in new_selected:
                            new_forbidden.add(r3)
                branch(idx + 1, new_selected, new_forbidden)
        
        # Option 2: Exclude x
        branch(idx + 1, selected, forbidden)
    
    # Initialize with greedy
    greedy_sel = set()
    greedy_forb = set()
    for x in range(1, N + 1):
        if x in greedy_forb:
            continue
        valid = True
        for u in greedy_sel:
            for v in greedy_sel:
                if u == v:
                    continue
                triple = sorted([x, u, v])
                if triple[0] + triple[2] == 2 * triple[1]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            greedy_sel.add(x)
            for u in list(greedy_sel):
                if u == x:
                    continue
                r1 = 2 * x - u
                r2 = 2 * u - x
                if 1 <= r1 <= N and r1 not in greedy_sel:
                    greedy_forb.add(r1)
                if 1 <= r2 <= N and r2 not in greedy_sel:
                    greedy_forb.add(r2)
                if (x + u) % 2 == 0:
                    r3 = (x + u) // 2
                    if 1 <= r3 <= N and r3 not in greedy_sel:
                        greedy_forb.add(r3)
    
    if greedy_sel:
        best_set = list(greedy_sel)
        best_weight = sum(weights[x] for x in greedy_sel)
    
    branch(1, set(), set())
    
    if best_set is None or not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
