import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

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
        """Find a valid coloring for n vertices using Simulated Annealing."""
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

        def calculate_energy(viol_k4, viol_k5, near_k4, near_k5):
            # High penalty for violations, lower for near-misses to guide search
            return 10000 * (viol_k4 + viol_k5) + (near_k4 + near_k5)

        def random_colors():
            return [random.randint(0, 1) for _ in range(num_t)]

        def structured_colors(mode):
            """Generate structured initial colorings."""
            cols = []
            for (a, b, c) in triples:
                if mode == 0:
                    col = (a + b + c) % 2
                elif mode == 1:
                    col = (a * b + c) % 2
                elif mode == 2:
                    col = (a ^ b ^ c) & 1
                elif mode == 3:
                    # Modular arithmetic with larger modulus
                    col = (a + 2*b + 3*c) % 3 < 2 
                elif mode == 4:
                    # Random but with some correlation
                    col = random.randint(0, 1)
                else:
                    col = (a * 3 + b * 5 + c * 7) % 4 < 2
                cols.append(col)
            return cols

        def simulated_annealing(colors, time_limit):
            """Run SA on the given initial coloring."""
            rc4, rc5, viol_k4, viol_k5, near_k4, near_k5 = init_state(colors)
            current_energy = calculate_energy(viol_k4, viol_k5, near_k4, near_k5)
            best_energy = current_energy
            best_colors = list(colors)
            
            T = 100.0
            T_min = 0.1
            cooling_rate = 0.9995
            t_start = time.time()
            steps = 0
            max_steps = 500000
            no_improve = 0
            stuck_threshold = 1000
            kick_threshold = 5000
            
            while time.time() - t_start < time_limit and steps < max_steps:
                if time.time() > deadline - 5:
                    break
                
                # Choose move: 1-flip or K-flip
                if random.random() < 0.1 and no_improve > 100:
                    # K-flip: flip K random triples
                    K = random.randint(2, 5)
                    flip_indices = random.sample(range(num_t), K)
                    
                    # Calculate delta for K-flip is complex, so we just apply and re-calc energy? 
                    # No, re-calc is O(N). N=5000. 5000 ops is fine if done rarely.
                    # But incremental is better. Let's stick to 1-flips for speed and simplicity, 
                    # but use "random walk" behavior by accepting bad moves more often when stuck.
                    pass 
                
                # 1-flip
                t = random.randint(0, num_t - 1)
                old_color = colors[t]
                new_color = 1 - old_color
                
                d_energy = 0
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
                            d_energy -= 10000
                            d_energy += 1
                        elif r == 3:
                            d_near_k4 -= 1
                            d_energy -= 1
                    else: # B -> R
                        if r == 3:
                            d_viol_k4 += 1
                            d_near_k4 -= 1
                            d_energy += 10000
                            d_energy -= 1
                        elif r == 2:
                            d_near_k4 += 1
                            d_energy += 1
                
                for k in triple_to_k5[t]:
                    r = rc5[k]
                    if old_color == 0: # R -> B
                        if r == 9:
                            d_viol_k5 += 1
                            d_near_k5 -= 1
                            d_energy += 10000
                            d_energy -= 1
                        elif r == 8:
                            d_near_k5 += 1
                            d_energy += 1
                    else: # B -> R
                        if r == 10:
                            d_viol_k5 -= 1
                            d_near_k5 += 1
                            d_energy -= 10000
                            d_energy += 1
                        elif r == 9:
                            d_near_k5 -= 1
                            d_energy -= 1

                new_energy = current_energy + d_energy
                
                # Acceptance
                if d_energy < 0:
                    accept = True
                else:
                    if d_energy == 0:
                        accept = random.random() < 0.5
                    else:
                        # Simulated Annealing
                        # Scale delta to temperature
                        # d_energy is large for violations (10000), small for near (1)
                        # We want to accept small increases in 'near' easily, but violations are hard.
                        # Let's use a scaled delta.
                        # If d_energy is from near-misses, it's small. If from violations, it's huge.
                        # This might make it too hard to accept moves that fix a violation but create many near-misses?
                        # Actually, fixing a violation reduces energy by 10000. Creating near-misses adds 1.
                        # So moves that fix violations are always accepted.
                        # Moves that create violations are rarely accepted unless T is high.
                        # This is good.
                        
                        # However, we might get stuck in a state with 0 violations but many near-misses.
                        # We want to explore the landscape of near-misses.
                        # So, let's treat 'near' energy and 'violation' energy separately for temperature scaling?
                        # Or just use a high T initially.
                        
                        p = math.exp(-d_energy / T)
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
                    current_energy = new_energy

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = list(colors)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

                # Cool down
                T = max(T * cooling_rate, T_min)
                
                # If stuck for a long time, "reheat" or restart
                if no_improve > kick_threshold:
                    # Reheat
                    T = min(T * 1.5, 100.0)
                    no_improve = 0
                    # Optional: Random kick
                    # for _ in range(10):
                    #    idx = random.randint(0, num_t-1)
                    #    colors[idx] = 1 - colors[idx]
                    #    # Update state... (too complex for inline, just re-init if we want to be safe? No, that's slow)
                    #    # Let's just rely on SA acceptance of bad moves at higher T.

                steps += 1

            return best_colors, best_energy

        # Try multiple starts
        # Allocate time based on n
        # Larger n is harder, but we want to find the largest n.
        # We will try n=34 first. If it fails, try 33, etc.
        
        # For n=34, we have ~5000 triples.
        # SA step is fast.
        
        # We'll do a few long runs and a few short runs.
        runs = []
        if time_budget > 50:
            runs.append((10, 0.5)) # 10 runs, 50% of time each? No, total time budget.
            # Let's just do a loop until time runs out.
        
        best_result = None
        best_obj = float('inf')
        
        t_start_solve = time.time()
        run_id = 0
        
        while time.time() - t_start_solve < time_budget:
            if time.time() > deadline - 10:
                break
                
            # Choose initial coloring
            if run_id < 5:
                colors = structured_colors(run_id)
            else:
                colors = random_colors()
            
            # Time per run: start with 2s, increase if successful?
            # Just use a fixed chunk, e.g., 2s per run, or remaining time if last.
            time_left = time_budget - (time.time() - t_start_solve)
            if time_left < 1.0:
                break
            
            # Use at most 2 seconds per run to allow many restarts
            run_time = min(2.0, time_left)
            
            result, obj = simulated_annealing(colors, run_time)
            
            if obj < best_obj:
                best_obj = obj
                best_result = result
                
            if obj == 0:
                if check_valid(n, result):
                    return result
            
            run_id += 1
            
        return None

    # Try to find solution for largest n first
    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        
        # Allocate time: give more time to larger n?
        # Actually, if we find a valid coloring for n, we are done.
        # So we should spend most time on the largest n we hope to solve.
        # Let's try n=34 with a decent budget.
        if n >= 30:
            budget = min(60.0, deadline - 25 - time.time())
        else:
            budget = min(10.0, deadline - 25 - time.time())
            
        if budget < 2:
            break
            
        result = solve_n(n, budget)
        if result is not None:
            if check_valid(n, result):
                return {'n': n, 'colors': result}

    # Fallback: return a small valid coloring
    # n=9 is small. Let's just return a valid one.
    # A simple valid coloring for small n:
    # For n=9, we can use the seed's fallback or just a random one and check.
    # Let's try to quickly find a valid one for n=10 or so.
    n_fallback = 10
    if time.time() < deadline - 5:
        res = solve_n(n_fallback, 2.0)
        if res and check_valid(n_fallback, res):
            return {'n': n_fallback, 'colors': res}
            
    # Absolute fallback: n=9
    n = 9
    colors = []
    for a, b, c in combinations(range(n), 3):
        # Just use a pattern that is likely valid for small n
        # Or just random and check?
        # For n=9, it's very likely a random coloring is valid?
        # No, P(no red K4) is not 1.
        # Let's use the seed's fallback logic which was valid for n=9.
        colors.append(0 if (a + b + c) % 3 == 0 else 1)
        
    return {'n': n, 'colors': colors}
