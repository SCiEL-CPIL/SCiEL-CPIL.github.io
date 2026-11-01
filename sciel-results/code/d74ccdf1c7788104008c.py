import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Specific case for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n  # 729
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        # Precompute spheres: list of indices for each word within distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)

        # For fast lookup, convert spheres to sets
        sphere_sets = [set(s) for s in spheres]

        best_code = None
        best_size = total_words

        # Multiple restarts
        max_restarts = 300 # Increase restarts since each is fast
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # 1. Greedy Construction
            cover_count = [0] * total_words
            code = []
            code_set = set()
            
            while True:
                if time.time() > deadline:
                    break
                
                max_gain = -1
                best_idx = -1
                for i in range(total_words):
                    if i in code_set:
                        continue
                    gain = 0
                    for w in spheres[i]:
                        if cover_count[w] == 0:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_idx = i
                        if gain == 0:
                            break
                
                if best_idx == -1 or max_gain == 0:
                    break
                
                code.append(best_idx)
                code_set.add(best_idx)
                for w in spheres[best_idx]:
                    cover_count[w] += 1
                
                if min(cover_count) > 0:
                    break

            # 2. Local Search: Combined Removal and Swapping
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                
                for i in range(len(code)):
                    if time.time() > deadline:
                        break
                    
                    cw_idx = code[i]
                    # Find words covered ONLY by cw_idx
                    unique_words = []
                    for w in spheres[cw_idx]:
                        if cover_count[w] == 1:
                            unique_words.append(w)
                    
                    if not unique_words:
                        # Redundant, remove it
                        for w in spheres[cw_idx]:
                            cover_count[w] -= 1
                        code.pop(i)
                        code_set.remove(cw_idx)
                        changed = True
                        break # Restart loop to re-index
                    else:
                        # Try swapping with any other word w_new
                        # We need w_new to cover all unique_words
                        for w_new in range(total_words):
                            if w_new == cw_idx or w_new in code_set:
                                continue
                            
                            valid = True
                            for uw in unique_words:
                                if uw not in sphere_sets[w_new]:
                                    valid = False
                                    break
                            
                            if valid:
                                # Perform swap
                                for w in spheres[cw_idx]:
                                    cover_count[w] -= 1
                                for w in spheres[w_new]:
                                    cover_count[w] += 1
                                
                                code[i] = w_new
                                code_set.remove(cw_idx)
                                code_set.add(w_new)
                                changed = True
                                break # Restart loop
                                
                        if changed:
                            break

            # Final check and update best
            if len(code) < best_size:
                best_size = len(code)
                best_code = [list(all_words[idx]) for idx in code]
                
                if best_size <= 73:
                    break

        if best_code is not None:
            return best_code
        else:
            # Fallback: return the last code generated
            return [list(all_words[idx]) for idx in code] if 'code' in locals() else [list(t) for t in itertools.product(range(q), repeat=n)]

    else:
        # Fallback: Simple greedy for other params
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)

        cover_count = [0] * len(all_words)
        code = []
        code_set = set()

        while True:
            if time.time() > deadline:
                break
            max_gain = -1
            best_idx = -1
            for i in range(len(all_words)):
                if i in code_set:
                    continue
                gain = 0
                for w in spheres[i]:
                    if cover_count[w] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            if best_idx == -1 or max_gain == 0:
                break
            code.append(best_idx)
            code_set.add(best_idx)
            for w in spheres[best_idx]:
                cover_count[w] += 1
            if min(cover_count) > 0:
                break

        return [list(all_words[idx]) for idx in code]
