import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute gcd table and compatibility
    gcd_table = {}
    compat = [[False]*201 for _ in range(201)]
    for m1 in range(20, 201):
        for m2 in range(m1+1, 201):
            g = math.gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            gcd_table[(m2, m1)] = g
            if g > 1:
                compat[m1][m2] = True
                compat[m2][m1] = True
    
    def score(sel):
        return sum(1.0 / m for m, r in sel)
    
    def find_valid_residues(m, sel):
        valid = []
        for r in range(m):
            ok = True
            for em, er in sel:
                g = gcd_table[(m, em)]
                if r % g == er % g:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    def greedy_with_order(order, rng):
        sel = []
        for m in order:
            if time.time() > deadline:
                break
            # Check compatibility with all selected
            is_compat = True
            for em, _ in sel:
                if not compat[m][em]:
                    is_compat = False
                    break
            if not is_compat:
                continue
            
            valid = find_valid_residues(m, sel)
            if valid:
                r = rng.choice(valid)
                sel.append((m, r))
        return sel
    
    def local_search(sel, time_limit):
        best = list(sel)
        best_s = score(best)
        rng = random.Random(42)
        
        while time.time() < time_limit:
            improved = False
            selected_m = set(m for m, r in best)
            
            # 1. Try adding an unselected modulus
            # Prioritize smaller moduli
            for m_new in range(20, 201):
                if time.time() > time_limit:
                    break
                if m_new in selected_m:
                    continue
                
                is_compat = True
                for em, _ in best:
                    if not compat[m_new][em]:
                        is_compat = False
                        break
                if not is_compat:
                    continue
                
                valid = find_valid_residues(m_new, best)
                if valid:
                    r_new = rng.choice(valid)
                    new_sel = best + [(m_new, r_new)]
                    s = score(new_sel)
                    if s > best_s + 1e-12:
                        best = new_sel
                        best_s = s
                        improved = True
                        selected_m.add(m_new)
                        break # Re-loop to allow more additions

            # 2. Try 1-swap: remove one, add one smaller
            if len(best) >= 1:
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                    m_remove, _ = best[i]
                    remaining = [x for j, x in enumerate(best) if j != i]
                    
                    for m_try in range(20, m_remove):
                        if time.time() > time_limit:
                            break
                        if m_try in selected_m:
                            continue
                            
                        is_compat = True
                        for em, _ in remaining:
                            if not compat[m_try][em]:
                                is_compat = False
                                break
                        if not is_compat:
                            continue

                        valid = find_valid_residues(m_try, remaining)
                        if valid:
                            r_try = rng.choice(valid)
                            new_sel = remaining + [(m_try, r_try)]
                            s = score(new_sel)
                            if s > best_s + 1e-12:
                                best = new_sel
                                best_s = s
                                improved = True
                                selected_m.remove(m_remove)
                                selected_m.add(m_try)
                                break
                    if improved:
                        break

            # 3. Try 2-swap: remove two, add two
            if len(best) >= 2:
                n = len(best)
                for i in range(n):
                    if time.time() > time_limit:
                        break
                    for j in range(i+1, n):
                        if time.time() > time_limit:
                            break
                        m1, _ = best[i]
                        m2, _ = best[j]
                        remaining = [x for k, x in enumerate(best) if k != i and k != j]
                        removed_score = 1.0/m1 + 1.0/m2
                        
                        # Try adding two moduli
                        # This is O(N^2) inside the loop, which is slow.
                        # We'll just try to add one better one if possible?
                        # Or just skip 2-swap for now if too slow.
                        pass

            if not improved:
                break
                
        return best, best_s

    best_solution = []
    best_score = 0.0
    
    # Phase 1: Multiple greedy strategies
    order_asc = list(range(20, 201))
    order_desc = list(range(200, 19, -1))
    rng_shuf = random.Random(12345)
    order_shuf = list(range(20, 201))
    rng_shuf.shuffle(order_shuf)
    
    orders = [order_asc, order_desc, order_shuf]
    
    for restart in range(800):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(restart * 7919 + 13)
        order = orders[restart % len(orders)]
        
        if restart % 3 == 0:
            temp_order = list(order)
            rng.shuffle(temp_order)
            order = temp_order
        
        sel = greedy_with_order(order, rng)
        s = score(sel)
        
        if s > best_score:
            best_score = s
            best_solution = sel
    
    # Phase 2: Local search
    if best_solution and time.time() < deadline - 10:
        refined, refined_score = local_search(best_solution, deadline - 10)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined
    
    # Phase 3: More restarts
    if time.time() < deadline - 20:
        for restart in range(800, 1200):
            if time.time() > deadline - 20:
                break
            rng = random.Random(restart * 104729 + 37)
            order = orders[restart % len(orders)]
            if restart % 2 == 0:
                temp_order = list(order)
                rng.shuffle(temp_order)
                order = temp_order
            sel = greedy_with_order(order, rng)
            s = score(sel)
            if s > best_score:
                best_score = s
                best_solution = sel
        
        if best_solution and time.time() < deadline - 10:
            refined, refined_score = local_search(best_solution, deadline - 10)
            if refined_score > best_score:
                best_score = refined_score
                best_solution = refined
    
    # Final local search
    if best_solution and time.time() < deadline - 3:
        refined, refined_score = local_search(best_solution, deadline - 3)
        if refined_score > best_score:
            best_score = refined_score
            best_solution = refined
    
    return [[m, r] for m, r in best_solution]
