def construct():
    import time
    import random
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 85
    
    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [1.0 / d for d in denominators]
    
    best_total = 0.0
    best_selected = set()
    
    # Multiple greedy initializations
    for _ in range(500):
        order = list(range(n))
        random.shuffle(order)
        selected = set()
        total = 0.0
        for i in order:
            if total + vals[i] < 1.0:
                selected.add(i)
                total += vals[i]
        if total > best_total:
            best_total = total
            best_selected = set(selected)
    
    # Local search
    selected = set(best_selected)
    total = best_total
    
    while time.time() < deadline - 2:
        for _ in range(100000):
            if time.time() > deadline - 2:
                break
            move_type = random.choice(['add', 'remove', 'swap', 'add2', 'remove2'])
            
            if move_type == 'add' and len(selected) < n:
                candidates = [i for i in range(n) if i not in selected and total + vals[i] < 1.0]
                if candidates:
                    i = random.choice(candidates)
                    new_total = total + vals[i]
                    if new_total > best_total:
                        best_total = new_total
                        best_selected = set(selected | {i})
                        selected.add(i)
                        total = new_total
            elif move_type == 'remove' and len(selected) > 0:
                i = random.choice(list(selected))
                new_total = total - vals[i]
                if new_total < 1.0 and new_total > best_total:
                    best_total = new_total
                    best_selected = set(selected - {i})
                    selected.remove(i)
                    total = new_total
            elif move_type == 'swap' and len(selected) > 0 and len(selected) < n:
                sel_list = list(selected)
                unselected = [i for i in range(n) if i not in selected]
                if unselected:
                    i_out = random.choice(sel_list)
                    i_in = random.choice(unselected)
                    new_total = total - vals[i_out] + vals[i_in]
                    if new_total < 1.0 and new_total > best_total:
                        best_total = new_total
                        best_selected = (selected - {i_out}) | {i_in}
                        selected = set(best_selected)
                        total = new_total
            elif move_type == 'add2' and len(selected) < n - 1:
                unselected = [i for i in range(n) if i not in selected]
                if len(unselected) >= 2:
                    i1, i2 = random.sample(unselected, 2)
                    new_total = total + vals[i1] + vals[i2]
                    if new_total < 1.0 and new_total > best_total:
                        best_total = new_total
                        best_selected = set(selected | {i1, i2})
                        selected = set(best_selected)
                        total = new_total
            elif move_type == 'remove2' and len(selected) >= 2:
                sel_list = list(selected)
                i1, i2 = random.sample(sel_list, 2)
                new_total = total - vals[i1] - vals[i2]
                if new_total < 1.0 and new_total > best_total:
                    best_total = new_total
                    best_selected = set(selected - {i1, i2})
                    selected = set(best_selected)
                    total = new_total
    
    # Final greedy completion
    selected = set(best_selected)
    total = best_total
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i not in selected and total + vals[i] < 1.0:
                if total + vals[i] > best_total:
                    best_total = total + vals[i]
                    best_selected = set(selected | {i})
                    selected = set(best_selected)
                    total = best_total
                    changed = True
    
    # Verify with exact fractions
    exact_total = Fraction(0)
    for i in best_selected:
        exact_total += Fraction(1, denominators[i])
    
    if exact_total >= 1:
        # Fallback to simple greedy
        selected, total = [], Fraction()
        for i, d in enumerate(range(2, 49)):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    return sorted(best_selected)
