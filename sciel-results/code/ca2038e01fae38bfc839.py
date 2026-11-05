import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def try_solve(n, time_limit):
        if time.time() > time_limit:
            return None, None
            
        # Precompute
        triples = list(combinations(range(n), 3))
        M = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # 4-sets
        sets_4 = list(combinations(range(n), 4))
        triple_to_4sets = [[] for _ in range(M)]
        set4_to_triples = []
        for s in sets_4:
            tris = []
            for i in range(4):
                t = tuple(sorted(s[:i] + s[i+1:]))
                idx = triple_idx[t]
                tris.append(idx)
                triple_to_4sets[idx].append(len(set4_to_triples))
            set4_to_triples.append(tris)
            
        # 5-sets
        sets_5 = list(combinations(range(n), 5))
        triple_to_5sets = [[] for _ in range(M)]
        set5_to_triples = []
        for s in sets_5:
            tris = []
            for combo in combinations(s, 3):
                idx = triple_idx[tuple(combo)]
                tris.append(idx)
                triple_to_5sets[idx].append(len(set5_to_triples))
            set5_to_triples.append(tris)
            
        # Initialize random colors
        colors = [random.randint(0, 1) for _ in range(M)]
        
        # Initialize counts
        red_count_4 = [0] * len(sets_4)
        for i, tris in enumerate(set4_to_triples):
            for t in tris:
                if colors[t] == 0:
                    red_count_4[i] += 1
                    
        blue_count_5 = [0] * len(sets_5)
        for i, tris in enumerate(set5_to_triples):
            for t in tris:
                if colors[t] == 1:
                    blue_count_5[i] += 1
                    
        # Initialize violation counts
        num_red_k4 = sum(1 for c in red_count_4 if c == 4)
        num_blue_k5 = sum(1 for c in blue_count_5 if c == 10)
        
        # Helper to check if solution
        def is_solution():
            return num_red_k4 == 0 and num_blue_k5 == 0
            
        # Local Search
        end_time = time.time() + time_limit
        while time.time() < end_time:
            if is_solution():
                return n, colors
                
            # Pick a violation
            fixed = False
            # Try to fix a red K4
            if num_red_k4 > 0 and (num_blue_k5 == 0 or random.random() < num_red_k4 / (num_red_k4 + num_blue_k5)):
                # Find a random 4-set that is a violation
                # Instead of scanning, we can just pick random 4-sets until we find one
                # But if violations are rare, this is slow.
                # Given we are starting random, violations are dense.
                # As we approach solution, violations are sparse.
                # Better: Keep a list of violating sets?
                # For now, let's just pick random 4-set. If not violation, try 5-set.
                
                # Optimization: If violations are few, this is bad.
                # Let's maintain lists of violating sets.
                pass # Logic below
                
            # Simplified: Just pick random set, check, flip.
            # If we are stuck, this might be slow.
            # Let's implement the list maintenance.
            
            # Actually, maintaining lists is complex.
            # Let's stick to random picking but add a "random flip" if no violation found quickly?
            # No, just pick random 4-set. If violation, flip. Else pick random 5-set. If violation, flip. Else random flip.
            
            # This is O(1) expected if violations are frequent.
            # If violations are rare, we do many "wasted" picks.
            # But we check is_solution() at start of loop, so if we are close, we exit.
            
            # To improve "rare violation" case:
            # We can pick a random triple and flip it?
            # No, that doesn't target violations.
            
            # Let's try this:
            # 1. Pick random 4-set. If violation, flip a random triple in it.
            # 2. Else, pick random 5-set. If violation, flip a random triple in it.
            # 3. Else, flip a random triple.
            
            # This ensures progress.
            
            # Try 4-set
            if num_red_k4 > 0:
                # Pick random 4-set
                # To avoid infinite loop if num_red_k4 is small, we can just pick one.
                # If we pick a non-violation, we fall through.
                s_idx = random.randint(0, len(sets_4)-1)
                if red_count_4[s_idx] == 4:
                    tris = set4_to_triples[s_idx]
                    t_idx = random.choice(tris)
                    old_c = colors[t_idx]
                    new_c = 1 - old_c
                    colors[t_idx] = new_c
                    
                    # Update counts
                    for s4_idx in triple_to_4sets[t_idx]:
                        old_c4 = red_count_4[s4_idx]
                        if old_c == 0: # red -> blue
                            red_count_4[s4_idx] -= 1
                            new_c4 = red_count_4[s4_idx]
                            if old_c4 == 4: num_red_k4 -= 1
                            if new_c4 == 4: num_red_k4 += 1
                        else: # blue -> red
                            red_count_4[s4_idx] += 1
                            new_c4 = red_count_4[s4_idx]
                            if old_c4 == 4: num_red_k4 -= 1
                            if new_c4 == 4: num_red_k4 += 1
                            
                    for s5_idx in triple_to_5sets[t_idx]:
                        old_c5 = blue_count_5[s5_idx]
                        if old_c == 1: # blue -> red
                            blue_count_5[s5_idx] -= 1
                            new_c5 = blue_count_5[s5_idx]
                            if old_c5 == 10: num_blue_k5 -= 1
                            if new_c5 == 10: num_blue_k5 += 1
                        else: # red -> blue
                            blue_count_5[s5_idx] += 1
                            new_c5 = blue_count_5[s5_idx]
                            if old_c5 == 10: num_blue_k5 -= 1
                            if new_c5 == 10: num_blue_k5 += 1
                            
                    fixed = True
            else:
                # No red K4, try blue K5
                if num_blue_k5 > 0:
                    s_idx = random.randint(0, len(sets_5)-1)
                    if blue_count_5[s_idx] == 10:
                        tris = set5_to_triples[s_idx]
                        t_idx = random.choice(tris)
                        old_c = colors[t_idx]
                        new_c = 1 - old_c
                        colors[t_idx] = new_c
                        
                        for s4_idx in triple_to_4sets[t_idx]:
                            old_c4 = red_count_4[s4_idx]
                            if old_c == 0:
                                red_count_4[s4_idx] -= 1
                                new_c4 = red_count_4[s4_idx]
                                if old_c4 == 4: num_red_k4 -= 1
                                if new_c4 == 4: num_red_k4 += 1
                            else:
                                red_count_4[s4_idx] += 1
                                new_c4 = red_count_4[s4_idx]
                                if old_c4 == 4: num_red_k4 -= 1
                                if new_c4 == 4: num_red_k4 += 1
                                
                        for s5_idx in triple_to_5sets[t_idx]:
                            old_c5 = blue_count_5[s5_idx]
                            if old_c == 1:
                                blue_count_5[s5_idx] -= 1
                                new_c5 = blue_count_5[s5_idx]
                                if old_c5 == 10: num_blue_k5 -= 1
                                if new_c5 == 10: num_blue_k5 += 1
                            else:
                                blue_count_5[s5_idx] += 1
                                new_c5 = blue_count_5[s5_idx]
                                if old_c5 == 10: num_blue_k5 -= 1
                                if new_c5 == 10: num_blue_k5 += 1
                                
                        fixed = True
            if not fixed:
                # Random flip
                t_idx = random.randint(0, M-1)
                old_c = colors[t_idx]
                new_c = 1 - old_c
                colors[t_idx] = new_c
                
                for s4_idx in triple_to_4sets[t_idx]:
                    old_c4 = red_count_4[s4_idx]
                    if old_c == 0:
                        red_count_4[s4_idx] -= 1
                        new_c4 = red_count_4[s4_idx]
                        if old_c4 == 4: num_red_k4 -= 1
                        if new_c4 == 4: num_red_k4 += 1
                    else:
                        red_count_4[s4_idx] += 1
                        new_c4 = red_count_4[s4_idx]
                        if old_c4 == 4: num_red_k4 -= 1
                        if new_c4 == 4: num_red_k4 += 1
                        
                for s5_idx in triple_to_5sets[t_idx]:
                    old_c5 = blue_count_5[s5_idx]
                    if old_c == 1:
                        blue_count_5[s5_idx] -= 1
                        new_c5 = blue_count_5[s5_idx]
                        if old_c5 == 10: num_blue_k5 -= 1
                        if new_c5 == 10: num_blue_k5 += 1
                    else:
                        blue_count_5[s5_idx] += 1
                        new_c5 = blue_count_5[s5_idx]
                        if old_c5 == 10: num_blue_k5 -= 1
                        if new_c5 == 10: num_blue_k5 += 1

        return None, None

    # Try n=34
    res_n, res_colors = try_solve(34, 100.0)
    if res_colors:
        return {'n': res_n, 'colors': res_colors}
        
    # If failed, try smaller n if time permits
    if time.time() < deadline - 5.0:
        for n_try in [33, 32, 31, 30]:
            if time.time() > deadline - 5.0:
                break
            res_n, res_colors = try_solve(n_try, 10.0)
            if res_colors:
                return {'n': res_n, 'colors': res_colors}
                
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
