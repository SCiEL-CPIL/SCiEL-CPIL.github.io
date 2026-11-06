def construct():
    from fractions import Fraction
    import time
    import random
    
    start_time = time.time()
    deadline = 80.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    best_selected = None
    best_deficit = Fraction(1)
    
    def update(sel, tot):
        nonlocal best_selected, best_deficit
        if tot < 1:
            deficit = 1 - tot
            if deficit < best_deficit:
                best_deficit = deficit
                best_selected = tuple(sorted(sel))
    
    def local_search(sel, tot, max_iters=500):
        nonlocal best_selected, best_deficit
        sel = set(sel)
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline:
                return
            improved = False
            
            # Try additions
            for i in range(n):
                if i not in sel:
                    new_tot = tot + recip[i]
                    if new_tot < 1:
                        new_deficit = 1 - new_tot
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_selected = tuple(sorted(sel | {i}))
                            sel.add(i)
                            tot = new_tot
                            improved = True
            
            # Try 1-for-1 swaps
            for i in list(sel):
                if improved:
                    break
                for j in range(n):
                    if j not in sel:
                        new_tot = tot - recip[i] + recip[j]
                        if 0 < new_tot < 1:
                            new_deficit = 1 - new_tot
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                new_sel = set(sel)
                                new_sel.discard(i)
                                new_sel.add(j)
                                best_selected = tuple(sorted(new_sel))
                                sel = new_sel
                                tot = new_tot
                                improved = True
                                break
            
            # Try 1-for-2 swaps (remove one, add two)
            if not improved:
                for i in list(sel):
                    if improved:
                        break
                    base = tot - recip[i]
                    for j in range(n):
                        if j not in sel:
                            t1 = base + recip[j]
                            if t1 >= 1:
                                continue
                            for k in range(j+1, n):
                                if k not in sel:
                                    new_tot = t1 + recip[k]
                                    if 0 < new_tot < 1:
                                        new_deficit = 1 - new_tot
                                        if new_deficit < best_deficit:
                                            best_deficit = new_deficit
                                            new_sel = set(sel)
                                            new_sel.discard(i)
                                            new_sel.add(j)
                                            new_sel.add(k)
                                            best_selected = tuple(sorted(new_sel))
                                            sel = new_sel
                                            tot = new_tot
                                            improved = True
                                            break
                    if improved:
                        break
            
            # Try 2-for-1 swaps (remove two, add one)
            if not improved:
                sel_list = list(sel)
                for ai in range(len(sel_list)):
                    if improved:
                        break
                    for bi in range(ai+1, len(sel_list)):
                        i, j = sel_list[ai], sel_list[bi]
                        base = tot - recip[i] - recip[j]
                        for k in range(n):
                            if k not in sel:
                                new_tot = base + recip[k]
                                if 0 < new_tot < 1:
                                    new_deficit = 1 - new_tot
                                    if new_deficit < best_deficit:
                                        best_deficit = new_deficit
                                        new_sel = set(sel)
                                        new_sel.discard(i)
                                        new_sel.discard(j)
                                        new_sel.add(k)
                                        best_selected = tuple(sorted(new_sel))
                                        sel = new_sel
                                        tot = new_tot
                                        improved = True
                                        break
                    if improved:
                        break
            
            if not improved:
                break
    
    # Multiple greedy initializations with different orderings
    orderings = [
        list(range(n)),                          # natural order
        list(range(n-1, -1, -1)),                # reverse
        sorted(range(n), key=lambda i: recip[i]),  # ascending value
        sorted(range(n), key=lambda i: -recip[i]), # descending value
    ]
    
    random.seed(42)
    for _ in range(10):
        orderings.append(list(random.sample(range(n), n)))
    
    for order in orderings:
        if time.time() > deadline:
            break
        sel = set()
        tot = Fraction(0)
        for i in order:
            if tot + recip[i] < 1:
                sel.add(i)
                tot += recip[i]
        update(sel, tot)
        local_search(sel, tot)
    
    # Random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        sel = set()
        tot = Fraction(0)
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if tot + recip[i] < 1:
                sel.add(i)
                tot += recip[i]
        update(sel, tot)
        local_search(sel, tot)
    
    if best_selected is None:
        return [0]
    
    return list(best_selected)
