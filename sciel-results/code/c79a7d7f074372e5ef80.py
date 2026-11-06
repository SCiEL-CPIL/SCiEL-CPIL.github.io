import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 16
    vertices = range(N)
    all_triples = list(itertools.combinations(vertices, 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_triples)]
    
    for six_vertices in itertools.combinations(vertices, 6):
        a, b, c, d, e, f = six_vertices
        pairs_list = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        
        for p1, p2, p3 in pairs_list:
            config_triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        config_triples.append(idx)
            
            config_id = len(k222_configs)
            k222_configs.append(config_triples)
            
            for t_idx in config_triples:
                triple_to_k222[t_idx].append(config_id)
                
    num_k222 = len(k222_configs)
    
    def update_counts(ti, sign, k222_counts, num_violations):
        old_viol = num_violations
        for cid in triple_to_k222[ti]:
            old_count = k222_counts[cid]
            new_count = old_count + sign
            k222_counts[cid] = new_count
            
            if sign == 1:
                if old_count == 7:
                    old_viol += 1
            else:
                if old_count == 8:
                    old_viol -= 1
        return old_viol

    def greedy_init(rng):
        present = [False] * num_triples
        k222_counts = [0] * num_k222
        num_viol = 0
        count = 0
        
        # Sort triples by degree (number of configs they touch) descending
        # This is a heuristic: place "high impact" triples first?
        # Or ascending? Let's try descending first.
        degrees = [len(triple_to_k222[i]) for i in range(num_triples)]
        order = list(range(num_triples))
        order.sort(key=lambda x: degrees[x], reverse=True)
        
        # Shuffle slightly to break symmetry
        rng.shuffle(order)
        
        for ti in order:
            ok = True
            for cid in triple_to_k222[ti]:
                if k222_counts[cid] >= 7:
                    ok = False
                    break
            if ok:
                present[ti] = True
                count += 1
                num_viol = update_counts(ti, 1, k222_counts, num_viol)
                
        return present, k222_counts, count, num_viol

    def local_search(present, k222_counts, count, num_viol, rng, time_limit):
        """Hybrid Local Search: Greedy Repair + Simulated Annealing."""
        current_score = count * 1.0 - num_viol * 1000.0  # High penalty for violations
        best_score = current_score
        best_present = present[:]
        best_count = count
        best_viol = num_viol
        
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.05
        stuck_counter = 0
        max_stuck = 50
        
        while time.time() < time_limit:
            if time.time() > deadline:
                break
            
            # Phase 1: Greedy Repair if invalid
            if num_viol > 0:
                # Find a violated config and remove a random triple from it
                violated = [i for i in range(num_k222) if k222_counts[i] >= 8]
                if violated:
                    cid = rng.choice(violated)
                    config_triples = k222_configs[cid]
                    candidates = [ti for ti in config_triples if present[ti]]
                    if candidates:
                        ti = rng.choice(candidates)
                        present[ti] = False
                        count -= 1
                        num_viol = update_counts(ti, -1, k222_counts, num_viol)
                        current_score = count * 1.0 - num_viol * 1000.0
                        if current_score > best_score:
                            best_score = current_score
                            best_present = present[:]
                            best_count = count
                            best_viol = num_viol
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        continue
                else:
                    # Should not happen if num_viol > 0, but safety break
                    pass

            # Phase 2: Try to add a valid triple (Greedy Fill)
            added = False
            for _ in range(10): # Try a few random additions
                ti = rng.randint(0, num_triples - 1)
                if present[ti]:
                    continue
                ok = True
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] >= 7:
                        ok = False
                        break
                if ok:
                    present[ti] = True
                    count += 1
                    num_viol = update_counts(ti, 1, k222_counts, num_viol)
                    current_score = count * 1.0 - num_viol * 1000.0
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                    added = True
                    break
            
            if added:
                continue

            # Phase 3: Simulated Annealing Moves (when stuck)
            move_type = rng.random()
            
            if move_type < 0.5:
                # Swap: Remove one, Add one
                # Try to find a swap that improves or maintains score
                improved = False
                for _ in range(5):
                    ti1 = rng.randint(0, num_triples - 1)
                    ti2 = rng.randint(0, num_triples - 1)
                    if ti1 == ti2:
                        continue
                    if present[ti1] == present[ti2]:
                        continue
                    
                    if present[ti1]:
                        remove_ti = ti1
                        add_ti = ti2
                    else:
                        remove_ti = ti2
                        add_ti = ti1
                    
                    # Calculate delta
                    delta_viol = 0
                    # Remove remove_ti
                    for cid in triple_to_k222[remove_ti]:
                        if k222_counts[cid] == 8:
                            delta_viol -= 1
                    # Add add_ti (after removal)
                    for cid in triple_to_k222[add_ti]:
                        adj_count = k222_counts[cid]
                        if cid in triple_to_k222[remove_ti]:
                            adj_count -= 1
                        if adj_count == 7:
                            delta_viol += 1
                    
                    delta_score = 0.0 - delta_viol * 1000.0 # Count doesn't change
                    
                    # Accept if better or with probability exp(delta/T)
                    if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                        present[remove_ti] = False
                        present[add_ti] = True
                        num_viol = update_counts(remove_ti, -1, k222_counts, num_viol)
                        num_viol = update_counts(add_ti, 1, k222_counts, num_viol)
                        current_score = count * 1.0 - num_viol * 1000.0
                        if current_score > best_score:
                            best_score = current_score
                            best_present = present[:]
                            best_count = count
                            best_viol = num_viol
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        improved = True
                        break
                
                if improved:
                    continue

            elif move_type < 0.8:
                # Add random (might create violation, SA handles it)
                ti = rng.randint(0, num_triples - 1)
                if present[ti]:
                    continue
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 7:
                        delta_viol += 1
                delta_score = 1.0 - delta_viol * 1000.0
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti] = True
                    count += 1
                    num_viol = update_counts(ti, 1, k222_counts, num_viol)
                    current_score = count * 1.0 - num_viol * 1000.0
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
            else:
                # Remove random
                ti = rng.randint(0, num_triples - 1)
                if not present[ti]:
                    continue
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 8:
                        delta_viol -= 1
                delta_score = -1.0 - delta_viol * 1000.0
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti] = False
                    count -= 1
                    num_viol = update_counts(ti, -1, k222_counts, num_viol)
                    current_score = count * 1.0 - num_viol * 1000.0
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
                if stuck_counter > max_stuck:
                    T = 3.0
                    stuck_counter = 0
                    
        return best_present, best_count, best_viol

    def polish(present, k222_counts, count, num_viol, rng, time_limit):
        """Final polish: remove violations and add more triples."""
        while time.time() < time_limit:
            if time.time() > deadline:
                break
            
            # Remove violations
            violated = [i for i in range(num_k222) if k222_counts[i] >= 8]
            if violated:
                cid = rng.choice(violated)
                config_triples = k222_configs[cid]
                candidates = [ti for ti in config_triples if present[ti]]
                if candidates:
                    ti = rng.choice(candidates)
                    present[ti] = False
                    count -= 1
                    num_viol = update_counts(ti, -1, k222_counts, num_viol)
                    continue
            
            # Add triples
            added = False
            for _ in range(20):
                ti = rng.randint(0, num_triples - 1)
                if present[ti]:
                    continue
                ok = True
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] >= 7:
                        ok = False
                        break
                if ok:
                    present[ti] = True
                    count += 1
                    num_viol = update_counts(ti, 1, k222_counts, num_viol)
                    added = True
                    break
            if not added:
                break
        
        return present, k222_counts, count, num_viol

    best_global_count = 0
    best_global_result = []
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(42 + r)
        
        present, k222_counts, count, num_viol = greedy_init(rng)
        
        if count > best_global_count:
            best_global_count = count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
        
        # Local search
        ls_deadline = min(time.time() + time_per_restart * 0.8, deadline - 5.0)
        best_present, best_count, best_viol = local_search(present, k222_counts, count, num_viol, rng, ls_deadline)
        
        # Polish
        if best_count > best_global_count or best_viol == 0:
            present2 = best_present[:]
            k222_counts2 = [0] * num_k222
            count2 = 0
            num_viol2 = 0
            for i in range(num_triples):
                if present2[i]:
                    count2 += 1
                    num_viol2 = update_counts(i, 1, k222_counts2, num_viol2)
            
            polish_deadline = min(time.time() + time_per_restart * 0.2, deadline - 2.0)
            polished_present, polished_counts, polished_count, polished_viol = polish(
                present2, k222_counts2, count2, num_viol2, rng, polish_deadline
            )
            
            if polished_count > best_global_count:
                best_global_count = polished_count
                best_global_result = [list(all_triples[i]) for i in range(num_triples) if polished_present[i]]
        
        if best_count > best_global_count:
            best_global_count = best_count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
            
    if not best_global_result:
        best_global_result = [[0, 1, 2]]
        
    return best_global_result
