import math
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def is_valid(arr):
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None

        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)

        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def find_min_diameter_set(max_candidate=70):
        best_set = None
        
        for D in range(35, max_candidate + 1):
            if time.time() > deadline - 5:
                break
            
            found = None
            
            def backtrack(current_elements, diffs, covered, start_idx, candidates_list):
                nonlocal found
                if found is not None:
                    return
                
                if time.time() > deadline - 5:
                    return
                
                if len(covered) == 30:
                    found = current_elements[:]
                    return
                
                m = len(current_elements)
                k = len(candidates_list) - start_idx
                missing = 30 - len(covered)
                
                # Stronger pruning: each new element can add at most m+1 new diffs
                # (diffs to existing elements + diffs to other new elements)
                # But due to Sidon constraint, actual count is limited.
                # Simple bound: max new diffs = k * (m + k - 1) // 2
                max_new_diffs = k * (m + k - 1) // 2
                if max_new_diffs < missing:
                    return
                
                # Sort candidates by how many missing diffs they would cover
                # Compute for each candidate how many missing diffs it covers
                scored_candidates = []
                for i in range(start_idx, len(candidates_list)):
                    c = candidates_list[i]
                    # Check if valid
                    ok = True
                    new_diffs = []
                    for x in current_elements:
                        d = c - x
                        if d in diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    if not ok:
                        continue
                    # Count how many missing diffs this covers
                    covers = 0
                    for d in new_diffs:
                        if 1 <= d <= 30 and d not in covered:
                            covers += 1
                    scored_candidates.append((covers, i, c, new_diffs))
                
                # Try candidates with most coverage first
                scored_candidates.sort(key=lambda x: -x[0])
                
                for covers, i, c, new_diffs in scored_candidates:
                    if found is not None:
                        return
                    
                    new_covered = set(covered)
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_covered.add(d)
                    
                    new_diffs_set = diffs.union(new_diffs)
                    new_elements = current_elements + [c]
                    
                    backtrack(new_elements, new_diffs_set, new_covered, i + 1, candidates_list)
                    
                    if found is not None:
                        return
            
            candidates = list(range(1, D + 1))
            backtrack([0], set(), set(), 0, candidates)
            
            if found is not None:
                return found
                
        return None

    def local_search_refine(initial_set, time_limit):
        if time_limit <= 0:
            return initial_set
            
        current = list(initial_set)
        current.sort()
        valid, _, cur_diam = is_valid(current)
        if not valid:
            return initial_set
            
        best_set = current[:]
        best_diam = cur_diam
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() < time_limit:
            if T < T_min:
                T = T_min
                
            neighbor = current[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Replace an element with a smaller value to reduce diameter
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    # Try to move element to a smaller value
                    old_val = neighbor[idx]
                    new_val = random.randint(1, old_val - 1) if old_val > 1 else 1
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:]
                        
            elif move_type < 0.7:
                # Replace with random value
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    max_val = max(neighbor)
                    new_val = random.randint(1, max_val)
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:]
                        
            elif move_type < 0.85:
                # Remove an element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    neighbor.pop(idx)
                    neighbor.sort()
                    
            else:
                # Add an element
                if len(neighbor) < 32:
                    max_val = max(neighbor)
                    new_val = random.randint(1, max_val)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        neighbor = current[:]
                        
            if len(neighbor) > 32 or 0 not in neighbor:
                continue
            if max(neighbor) > 10**6:
                continue
                
            diffs = set()
            n = len(neighbor)
            valid_sidon = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = neighbor[j] - neighbor[i]
                    if d in diffs:
                        valid_sidon = False
                        break
                    diffs.add(d)
                if not valid_sidon:
                    break
                    
            if not valid_sidon:
                continue
                
            if not all(d in diffs for d in range(1, 31)):
                continue
                
            new_diam = max(neighbor) - min(neighbor)
            delta = new_diam - best_diam
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current = neighbor
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_set = neighbor[:]
                    
            T *= cooling
            
        return best_set

    # Main Execution
    best_set = find_min_diameter_set(max_candidate=65)
    
    if best_set is None:
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            if valid:
                for y in a:
                    if x > y:
                        diffs.add(x - y)
                a.append(x)
            x += 1
        best_set = a
    else:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5:
            refined = local_search_refine(best_set, deadline - 5)
            if refined:
                v, _, diam_ref = is_valid(refined)
                v_orig, _, diam_orig = is_valid(best_set)
                if v and diam_ref < diam_orig:
                    best_set = refined
                    print(f"Refined diameter from {diam_orig} to {diam_ref}", file=sys.stderr)
    
    if best_set is None:
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            if valid:
                for y in a:
                    if x > y:
                        diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    return best_set

# EVOLVE-BLOCK-END
