import random
import math
import time

def construct():
    N = 256
    # 1. Precompute all 4-term arithmetic progressions
    ap_list = []
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                ap_idx = len(ap_list)
                ap_list.append((a, b, c, e))
                index_to_aps[a].append(ap_idx)
                index_to_aps[b].append(ap_idx)
                index_to_aps[c].append(ap_idx)
                index_to_aps[e].append(ap_idx)
                
    total_aps = len(ap_list)
    
    # 2. Initialization: Thue-Morse
    colors = [0] * N
    for i in range(N):
        colors[i] = bin(i).count('1') % 2
        
    # 3. Calculate initial mono status
    is_mono = [0] * total_aps
    current_score = 0
    for p in range(total_aps):
        ap = ap_list[p]
        if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
            is_mono[p] = 1
            current_score += 1

    best_colors = colors[:]
    best_score = current_score
    best_is_mono = is_mono[:]

    # 4. Simulated Annealing
    start_time = time.time()
    time_limit = 20.0
    
    T = 1.0
    cooling = 0.999
    
    def polish(cols, mono_arr, score):
        improved = True
        while improved:
            improved = False
            for i in range(N):
                old_c = cols[i]
                new_c = 1 - old_c
                delta = 0
                # Calculate delta
                for p in index_to_aps[i]:
                    if mono_arr[p]:
                        delta -= 1
                    else:
                        ap = ap_list[p]
                        # Check if other 3 are new_c
                        # We need to identify the other 3 indices
                        # Simple way: create a list of the 4 indices, remove i, check the rest
                        # To avoid list creation overhead, manual check
                        c0, c1, c2, c3 = ap
                        if i == c0:
                            if cols[c1] == new_c and cols[c2] == new_c and cols[c3] == new_c:
                                delta += 1
                        elif i == c1:
                            if cols[c0] == new_c and cols[c2] == new_c and cols[c3] == new_c:
                                delta += 1
                        elif i == c2:
                            if cols[c0] == new_c and cols[c1] == new_c and cols[c3] == new_c:
                                delta += 1
                        else: # i == c3
                            if cols[c0] == new_c and cols[c1] == new_c and cols[c2] == new_c:
                                delta += 1
                
                new_score = score + delta
                if new_score < score:
                    # Accept flip
                    cols[i] = new_c
                    score = new_score
                    improved = True
                    # Update mono_arr
                    for p in index_to_aps[i]:
                        if mono_arr[p]:
                            mono_arr[p] = 0
                        else:
                            ap = ap_list[p]
                            c0, c1, c2, c3 = ap
                            if i == c0:
                                if cols[c1] == new_c and cols[c2] == new_c and cols[c3] == new_c:
                                    mono_arr[p] = 1
                            elif i == c1:
                                if cols[c0] == new_c and cols[c2] == new_c and cols[c3] == new_c:
                                    mono_arr[p] = 1
                            elif i == c2:
                                if cols[c0] == new_c and cols[c1] == new_c and cols[c3] == new_c:
                                    mono_arr[p] = 1
                            else:
                                if cols[c0] == new_c and cols[c1] == new_c and cols[c2] == new_c:
                                    mono_arr[p] = 1
        return cols, mono_arr, score

    steps = 0
    while time.time() - start_time < time_limit:
        i = random.randint(0, N - 1)
        old_c = colors[i]
        new_c = 1 - old_c
        
        delta = 0
        for p in index_to_aps[i]:
            if is_mono[p]:
                delta -= 1
            else:
                ap = ap_list[p]
                c0, c1, c2, c3 = ap
                if i == c0:
                    if colors[c1] == new_c and colors[c2] == new_c and colors[c3] == new_c:
                        delta += 1
                elif i == c1:
                    if colors[c0] == new_c and colors[c2] == new_c and colors[c3] == new_c:
                        delta += 1
                elif i == c2:
                    if colors[c0] == new_c and colors[c1] == new_c and colors[c3] == new_c:
                        delta += 1
                else:
                    if colors[c0] == new_c and colors[c1] == new_c and colors[c2] == new_c:
                        delta += 1
        
        new_score = current_score + delta
        
        if new_score < current_score:
            colors[i] = new_c
            current_score = new_score
            # Update is_mono
            for p in index_to_aps[i]:
                if is_mono[p]:
                    is_mono[p] = 0
                else:
                    ap = ap_list[p]
                    c0, c1, c2, c3 = ap
                    if i == c0:
                        if colors[c1] == new_c and colors[c2] == new_c and colors[c3] == new_c:
                            is_mono[p] = 1
                    elif i == c1:
                        if colors[c0] == new_c and colors[c2] == new_c and colors[c3] == new_c:
                            is_mono[p] = 1
                    elif i == c2:
                        if colors[c0] == new_c and colors[c1] == new_c and colors[c3] == new_c:
                            is_mono[p] = 1
                    else:
                        if colors[c0] == new_c and colors[c1] == new_c and colors[c2] == new_c:
                            is_mono[p] = 1
                            
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                best_is_mono = is_mono[:]
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    colors[i] = new_c
                    current_score = new_score
                    for p in index_to_aps[i]:
                        if is_mono[p]:
                            is_mono[p] = 0
                        else:
                            ap = ap_list[p]
                            c0, c1, c2, c3 = ap
                            if i == c0:
                                if colors[c1] == new_c and colors[c2] == new_c and colors[c3] == new_c:
                                    is_mono[p] = 1
                            elif i == c1:
                                if colors[c0] == new_c and colors[c2] == new_c and colors[c3] == new_c:
                                    is_mono[p] = 1
                            elif i == c2:
                                if colors[c0] == new_c and colors[c1] == new_c and colors[c3] == new_c:
                                    is_mono[p] = 1
                            else:
                                if colors[c0] == new_c and colors[c1] == new_c and colors[c2] == new_c:
                                    is_mono[p] = 1
        
        T *= cooling
        
        if steps % 1000 == 0:
            colors, is_mono, current_score = polish(colors, is_mono, current_score)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                best_is_mono = is_mono[:]
        
        steps += 1

    # Final polish
    best_colors, best_is_mono, best_score = polish(best_colors, best_is_mono, best_score)
    
    return best_colors

# EVOLVE-BLOCK-END
