import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute all 4-term APs
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            ap = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                index_to_aps[idx].append(ap_idx)
    
    # Fast check for bad AP (uses <= 2 distinct colors)
    def is_bad_ap(ap, colors):
        c1 = colors[ap[0]]
        c2 = colors[ap[1]]
        c3 = colors[ap[2]]
        c4 = colors[ap[3]]
        # Count distinct colors
        # If all same -> bad
        if c1 == c2 == c3 == c4:
            return True
        # If exactly 2 distinct -> bad
        # This happens if the multiset is {a,a,a,b}, {a,a,b,b}, or {a,a,b,c} where b!=c but we only have 2 values? No.
        # Let's just use a small helper or manual logic.
        # Manual logic for distinct count < 3:
        # If any two are equal, we have at most 3 distinct.
        # If three are equal, we have at most 2 distinct (the 4th is either same or different).
        # If four are equal, 1 distinct.
        # If exactly two pairs, 2 distinct.
        # If one pair and two singles, 3 distinct (if singles are different from pair and each other) or 2 (if a single matches the pair).
        
        # Efficient check:
        # If c1==c2:
        #   if c3==c4: bad (2 colors max)
        #   else: if c1==c3 or c1==c4: bad (2 colors) else good (3 colors)
        # Elif c1==c3:
        #   if c2==c4: bad else good
        # Elif c1==c4:
        #   if c2==c3: bad else good
        # Elif c2==c3:
        #   if c2==c4: bad else good
        # Elif c2==c4:
        #   if c1==c3: bad else good
        # Elif c3==c4:
        #   if c2==c3: bad else good
        # Else: good (4 distinct)
        
        if c1 == c2:
            if c3 == c4:
                return True
            else:
                return (c1 == c3) or (c1 == c4)
        elif c1 == c3:
            return (c2 == c4)
        elif c1 == c4:
            return (c2 == c3)
        elif c2 == c3:
            return (c2 == c4)
        elif c2 == c4:
            return (c1 == c3)
        elif c3 == c4:
            return (c2 == c3)
        else:
            return False

    def calculate_initial_penalty(colors):
        total = 0
        ap_bad = [False] * len(aps)
        for i, ap in enumerate(aps):
            if is_bad_ap(ap, colors):
                total += 1
                ap_bad[i] = True
        return total, ap_bad

    def try_find_solution(K, time_limit):
        # Use local search to minimize violations, then minimize colors
        best_solution = None
        best_penalty = float('inf')
        best_color_count = float('inf')
        
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Random initialization
            colors = [random.randint(0, K - 1) for _ in range(N)]
            penalty, ap_bad = calculate_initial_penalty(colors)
            
            if penalty == 0:
                nc = len(set(colors))
                if nc < best_color_count:
                    best_color_count = nc
                    best_solution = list(colors)
                continue

            # Local Search: Simulated Annealing or Simple Hill Climbing
            # We will use a hybrid: first fix violations, then reduce colors.
            current_penalty = penalty
            current_ap_bad = ap_bad
            current_colors = colors
            
            # Phase 1: Eliminate violations
            for step in range(2000):
                if time.time() > time_limit:
                    break
                if current_penalty == 0:
                    break
                    
                # Pick a random index involved in a violation
                # To speed up, we can pick a random index and see if it helps, 
                # or pick from the list of bad APs.
                # Let's pick a random index.
                i = random.randint(0, N - 1)
                old_color = current_colors[i]
                
                best_new_color = old_color
                best_new_penalty = current_penalty
                
                # Try all K colors
                for c in range(K):
                    if c == old_color:
                        continue
                    # Calculate delta
                    delta = 0
                    for ap_idx in index_to_aps[i]:
                        was_bad = current_ap_bad[ap_idx]
                        # Apply temp
                        current_colors[i] = c
                        is_now_bad = is_bad_ap(aps[ap_idx], current_colors)
                        # Revert
                        current_colors[i] = old_color
                        
                        if was_bad and not is_now_bad:
                            delta -= 1
                        elif not was_bad and is_now_bad:
                            delta += 1
                    
                    new_penalty = current_penalty + delta
                    if new_penalty < best_new_penalty:
                        best_new_penalty = new_penalty
                        best_new_color = c
                
                if best_new_color != old_color:
                    # Apply best
                    for ap_idx in index_to_aps[i]:
                        was_bad = current_ap_bad[ap_idx]
                        current_colors[i] = best_new_color
                        is_now_bad = is_bad_ap(aps[ap_idx], current_colors)
                        if was_bad and not is_now_bad:
                            current_penalty -= 1
                            current_ap_bad[ap_idx] = False
                        elif not was_bad and is_now_bad:
                            current_penalty += 1
                            current_ap_bad[ap_idx] = True
                    current_colors[i] = best_new_color
                    current_penalty = best_new_penalty
                else:
                    # No improvement, maybe random move?
                    # For simplicity, stick to hill climbing for violations.
                    pass

            # Phase 2: Reduce number of colors (if penalty is 0)
            if current_penalty == 0:
                nc = len(set(current_colors))
                # Try to merge colors
                # Pick a color to eliminate
                while nc > 2: # Minimum 3 colors required? No, min 3 distinct per AP. Total colors can be 3.
                    if time.time() > time_limit:
                        break
                    used_colors = list(set(current_colors))
                    if len(used_colors) <= 1:
                        break
                    target_color = random.choice(used_colors)
                    positions = [i for i in range(N) if current_colors[i] == target_color]
                    
                    improved = False
                    for pos in random.sample(positions, len(positions)):
                        old_c = current_colors[pos]
                        # Try changing to a different used color
                        other_colors = [c for c in used_colors if c != target_color]
                        if not other_colors:
                            break
                        new_c = random.choice(other_colors)
                        
                        # Check if this move keeps penalty 0
                        # Calculate delta penalty
                        delta = 0
                        for ap_idx in index_to_aps[pos]:
                            was_bad = current_ap_bad[ap_idx]
                            current_colors[pos] = new_c
                            is_now_bad = is_bad_ap(aps[ap_idx], current_colors)
                            current_colors[pos] = old_c
                            if was_bad and not is_now_bad:
                                delta -= 1
                            elif not was_bad and is_now_bad:
                                delta += 1
                        
                        if delta == 0: # Penalty stays 0
                            # Check if it reduces color count
                            # If this was the last instance of target_color, nc decreases
                            count_target = 0
                            for p in positions:
                                if current_colors[p] == target_color:
                                    count_target += 1
                            
                            if count_target == 1:
                                # Apply move
                                for ap_idx in index_to_aps[pos]:
                                    was_bad = current_ap_bad[ap_idx]
                                    current_colors[pos] = new_c
                                    is_now_bad = is_bad_ap(aps[ap_idx], current_colors)
                                    if was_bad and not is_now_bad:
                                        current_penalty -= 1
                                        current_ap_bad[ap_idx] = False
                                    elif not was_bad and is_now_bad:
                                        current_penalty += 1
                                        current_ap_bad[ap_idx] = True
                                current_colors[pos] = new_c
                                nc -= 1
                                improved = True
                                break
                            else:
                                # Even if not last, we can move to reduce entropy? 
                                # But we only care about count. If count > 1, moving one doesn't reduce count.
                                # So we must find the last one.
                                pass
                    if not improved:
                        break
                
                final_nc = len(set(current_colors))
                if final_nc < best_color_count:
                    best_color_count = final_nc
                    best_solution = list(current_colors)

        return best_solution

    # Try increasing K
    # K=3 is impossible for N=128? 
    # Ruzsa-Szemeredi-Trotter says max size of 3-color AP-free set in [1,N] is o(N). 
    # But here we allow 2 colors in an AP as long as there's a 3rd? 
    # No, "AT LEAST three distinct colors". 
    # So 3 colors total is the absolute minimum.
    
    # Start with K=3
    sol = try_find_solution(3, time_limit=start_time + 20.0)
    if sol:
        return sol
        
    # K=4
    sol = try_find_solution(4, time_limit=start_time + 40.0)
    if sol:
        return sol
        
    # K=5
    sol = try_find_solution(5, time_limit=start_time + 60.0)
    if sol:
        return sol
        
    # K=6
    sol = try_find_solution(6, time_limit=start_time + 75.0)
    if sol:
        return sol
        
    # Fallback: Greedy with many colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
