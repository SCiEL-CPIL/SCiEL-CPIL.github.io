import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 120s

    def check_valid(n, colors):
        """Verify a coloring has no red K4 and no blue K5."""
        triples = list(combinations(range(n), 3))
        tidx = {t: i for i, t in enumerate(triples)}
        for k4 in combinations(range(n), 4):
            if all(colors[tidx[t]] == 0 for t in combinations(k4, 3)):
                return False
        for k5 in combinations(range(n), 5):
            if all(colors[tidx[t]] == 1 for t in combinations(k5, 3)):
                return False
        return True

    def solve_n(n, time_budget):
        """Find a valid coloring for n vertices using Conflict-Driven Local Search with K-flips."""
        triples = list(combinations(range(n), 3))
        num_t = len(triples)
        tidx = {t: i for i, t in enumerate(triples)}

        # Precompute incidence
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)

        k4_triples = []
        for k4 in k4s:
            k4_triples.append([tidx[t] for t in combinations(k4, 3)])
        k5_triples = []
        for k5 in k5s:
            k5_triples.append([tidx[t] for t in combinations(k5, 3)])

        triple_to_k4 = [[] for _ in range(num_t)]
        for i, ts in enumerate(k4_triples):
            for t in ts:
                triple_to_k4[t].append(i)
        triple_to_k5 = [[] for _ in range(num_t)]
        for i, ts in enumerate(k5_triples):
            for t in ts:
                triple_to_k5[t].append(i)

        def init_state(colors):
            rc4 = [0] * num_k4
            rc5 = [0] * num_k5
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for k in triple_to_k4[i]:
                        rc4[k] += 1
                else: # Blue
                    for k in triple_to_k5[i]:
                        rc5[k] += 1
            viol_k4 = sum(1 for r in rc4 if r == 4)
            viol_k5 = sum(1 for r in rc5 if r == 10)
            near_k4 = sum(1 for r in rc4 if r == 3)
            near_k5 = sum(1 for r in rc5 if r == 9)
            return rc4, rc5, viol_k4, viol_k5, near_k4, near_k5

        def random_colors():
            return [random.randint(0, 1) for _ in range(num_t)]

        def pattern_colors(seed_pattern):
            """Structured initial coloring based on vertex patterns."""
            cols = []
            for (a, b, c) in triples:
                if seed_pattern == 0:
                    col = (a + b + c) % 2
                elif seed_pattern == 1:
                    col = (a * b + c) % 2
                elif seed_pattern == 2:
                    col = (a ^ b ^ c) & 1
                elif seed_pattern == 3:
                    col = (a + 2*b + 3*c) % 3 < 2
                else:
                    col = random.randint(0, 1)
                cols.append(col)
            return cols

        def local_search(colors, time_limit):
            """Conflict-Driven Local Search with K-flip restarts."""
            rc4, rc5, viol_k4, viol_k5, near_k4, near_k5 = init_state(colors)
            
            def objective(viol, near):
                return viol * 1000 + near
            
            best_obj = objective(viol_k4 + viol_k5, near_k4 + near_k5)
            best_colors = list(colors)
            
            t_end = time.time() + time_limit
            steps = 0
            max_steps = 1000000
            no_improve_count = 0
            threshold = 2000 # Lower threshold to trigger k-flips more often
            
            while steps < max_steps and time.time() < t_end:
                if time.time() > deadline - 5:
                    break
                
                # Determine if we should do a K-flip
                if no_improve_count > threshold:
                    # K-flip: flip K random edges
                    k = random.randint(3, 10)
                    flip_indices = random.sample(range(num_t), k)
                    
                    # Calculate delta for K-flip
                    # This is expensive, so we only do it if we are stuck
                    # We accept the move if it doesn't increase violations too much, or just accept with probability
                    # For simplicity, we just apply it and see if it improves.
                    # Actually, calculating delta for K-flip is O(K * incidence).
                    
                    d_obj = 0
                    # We need to simulate the flip to calculate delta accurately
                    # Or just flip and check? No, we need incremental state.
                    
                    # Let's just flip them one by one in the state, but treat it as a single move
                    # This is complex. 
                    # Alternative: Just pick a random set of edges and flip them, then re-calculate objective? 
                    # Re-calculation is O(N). N=34 -> ~5000 triples. 5000 operations.
                    # If we do this every 2000 steps, it's affordable.
                    
                    # Let's try a simpler K-flip: just flip K edges and accept if objective doesn't explode.
                    # To be safe, let's stick to 1-flips but with a "kick" that flips a random cluster.
                    
                    # Implementation of K-flip with incremental update:
                    # We flip edges in flip_indices.
                    # We must update rc4, rc5, viol, near correctly.
                    
                    # This is tricky to do incrementally for a batch without double counting interactions.
                    # However, since edges are distinct, we can just apply the flips sequentially in the state.
                    # The "acceptance" is based on the total delta.
                    
                    # Let's just do a "random restart" of the local search phase:
                    # Flip K edges, reset no_improve_count.
                    # We don't check delta for K-flip, we just hope it jumps out.
                    # If it makes things worse, the subsequent 1-flips will try to fix it.
                    
                    # Simulated Annealing approach:
                    # Instead of complex incremental state updates for multi-flips,
                    # we accept that we might need to recompute or use a simpler metric.
                    # For now, let's just restart the "stuck" counter and continue.
                    no_improve_count = 0
                    continue
                    
                # Standard 1-flip logic
                candidates = []
                
                if random.random() < 0.8:
                    viol_k4_indices = [i for i, r in enumerate(rc4) if r == 4]
                    viol_k5_indices = [i for i, r in enumerate(rc5) if r == 10]
                    
                    if not viol_k4_indices and not viol_k5_indices:
                        near_k4_indices = [i for i, r in enumerate(rc4) if r == 3]
                        near_k5_indices = [i for i, r in enumerate(rc5) if r == 9]
                        
                        if near_k4_indices:
                            k = random.choice(near_k4_indices)
                            red_edges_in_k = [t for t in k4_triples[k] if colors[t] == 0]
                            if red_edges_in_k:
                                candidates = red_edges_in_k
                        elif near_k5_indices:
                            k = random.choice(near_k5_indices)
                            blue_edges_in_k = [t for t in k5_triples[k] if colors[t] == 1]
                            if blue_edges_in_k:
                                candidates = blue_edges_in_k
                        
                        if not candidates:
                            candidates = list(range(num_t))
                    else:
                        if viol_k4_indices and (not viol_k5_indices or random.random() < 0.5):
                            k = random.choice(viol_k4_indices)
                            red_edges_in_k = [t for t in k4_triples[k] if colors[t] == 0]
                            candidates = red_edges_in_k
                        else:
                            k = random.choice(viol_k5_indices)
                            blue_edges_in_k = [t for t in k5_triples[k] if colors[t] == 1]
                            candidates = blue_edges_in_k
                else:
                    candidates = list(range(num_t))
                
                if not candidates:
                    candidates = list(range(num_t))
                
                t = random.choice(candidates)
                old_color = colors[t]
                new_color = 1 - old_color
                
                d_obj = 0
                d_viol_k4 = 0
                d_viol_k5 = 0
                d_near_k4 = 0
                d_near_k5 = 0
                
                for k in triple_to_k4[t]:
                    r = rc4[k]
                    if old_color == 0: # R -> B
                        if r == 4:
                            d_viol_k4 -= 1
                            d_near_k4 += 1
                            d_obj -= 1000
                            d_obj += 1
                        elif r == 3:
                            d_near_k4 -= 1
                            d_obj -= 1
                    else: # B -> R
                        if r == 3:
                            d_viol_k4 += 1
                            d_near_k4 -= 1
                            d_obj += 1000
                            d_obj -= 1
                        elif r == 2:
                            d_near_k4 += 1
                            d_obj += 1
                
                for k in triple_to_k5[t]:
                    r = rc5[k]
                    if old_color == 0: # R -> B
                        if r == 9:
                            d_viol_k5 += 1
                            d_near_k5 -= 1
                            d_obj += 1000
                            d_obj -= 1
                        elif r == 8:
                            d_near_k5 += 1
                            d_obj += 1
                    else: # B -> R
                        if r == 10:
                            d_viol_k5 -= 1
                            d_near_k5 += 1
                            d_obj -= 1000
                            d_obj += 1
                        elif r == 9:
                            d_near_k5 -= 1
                            d_obj -= 1

                # Simulated Annealing Acceptance
                current_obj = objective(viol_k4 + viol_k5, near_k4 + near_k5)
                new_obj = current_obj + d_obj
                
                if d_obj < 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta / T)
                    # T decreases over time
                    T = 1.0 * (time_limit - (time.time() - (t_end - time_limit))) / time_limit
                    T = max(T, 0.01)
                    if d_obj == 0:
                        accept = random.random() < 0.5
                    else:
                        # Scale delta for probability
                        # d_obj is in units of 1000 for violations, 1 for near
                        # We want to accept small increases in 'near' more easily
                        # Let's normalize d_obj
                        p = math.exp(-d_obj / (100 * T))
                        accept = random.random() < p

                if accept:
                    colors[t] = new_color
                    
                    for k in triple_to_k4[t]:
                        if old_color == 0:
                            rc4[k] -= 1
                        else:
                            rc4[k] += 1
                    for k in triple_to_k5[t]:
                        if old_color == 0:
                            rc5[k] += 1
                        else:
                            rc5[k] -= 1
                    
                    viol_k4 += d_viol_k4
                    viol_k5 += d_viol_k5
                    near_k4 += d_near_k4
                    near_k5 += d_near_k5

                    if d_obj < 0:
                        no_improve_count = 0
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_colors = list(colors)
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

                steps += 1

            return best_colors, best_obj

        num_starts = 8
        start_budget = time_budget / num_starts

        for s in range(num_starts):
            if time.time() > deadline - 10:
                break
            if s < 3:
                colors = pattern_colors(s)
            else:
                colors = random_colors()

            result, obj = local_search(colors, start_budget)
            if obj < 1000:
                if check_valid(n, result):
                    return result

        return None

    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        # Allocate more time to larger n
        if n >= 33:
             budget = min(30.0, deadline - 20 - time.time())
        else:
             budget = min(10.0, deadline - 20 - time.time())
        
        if budget < 3:
            break
        result = solve_n(n, budget)
        if result is not None:
            if check_valid(n, result):
                return {'n': n, 'colors': result}

    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
