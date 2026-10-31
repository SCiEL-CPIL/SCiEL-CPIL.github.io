import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Generate Greedy Golomb Rulers
    def greedy_golomb(size, max_span_limit=1000):
        # Start with 0
        U = [0]
        diffs = set()
        for _ in range(size - 1):
            best_pos = None
            best_diffs = None
            current_max = max(U)
            # Search for next position
            for pos in range(current_max + 1, max_span_limit):
                new_diffs = set()
                valid = True
                for u in U:
                    d = abs(pos - u)
                    if d in diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    # We want the smallest pos that works?
                    # Or the one that minimizes future span?
                    # Greedy: smallest pos.
                    best_pos = pos
                    best_diffs = new_diffs
                    break
            
            if best_pos is None:
                return None
            
            U.append(best_pos)
            diffs.update(best_diffs)
        
        return U

    # Generate candidates for sizes 3 to 8
    for size in range(3, 9):
        if time.time() > deadline - 30:
            break
        # Try multiple random restarts or different starting offsets?
        # Greedy from 0 is deterministic.
        # Let's try greedy with some randomization in the "break" condition?
        # No, standard greedy is fine for small sizes.
        # Also try known optimal Golomb rulers if we can hardcode small ones?
        # Size 3: [0,1,3]
        # Size 4: [0,1,4,6]
        # Size 5: [0,1,4,9,11]
        # Size 6: [0,1,4,10,12,17]
        # Let's just use the greedy function and verify.
        
        cand = greedy_golomb(size)
        if cand:
            try_set(cand)
        
        # Also try a variant where we don't pick the absolute smallest, 
        # but one that might allow tighter packing later? 
        # For now, stick to simple greedy and local search.

    # Phase 2: Local Search on the best candidate
    if time.time() < deadline - 10:
        current = list(best_set)
        no_improve_count = 0
        max_iterations = 5000
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            improved = False
            max_cur = max(current)
            
            # Try adding an element
            if len(current) < 8:
                # Search limit: try to keep span small
                search_limit = max(max_cur + 10, 50)
                for new_el in range(1, search_limit + 1):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
            
            # Try replacing an element
            if len(current) >= 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    # Search in a window around the current element
                    center = current[i]
                    search_limit = max(center + 5, 30)
                    for new_el in range(1, search_limit + 1):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Random restart from a new greedy Golomb
                    size = random.randint(3, 7)
                    cand = greedy_golomb(size)
                    if cand:
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            no_improve_count = 0
                    else:
                        # If greedy fails, try a random set
                        size = random.randint(3, 5)
                        inner = random.sample(range(1, 30), size - 1)
                        cand = sorted([0] + inner)
                        try_set(cand)
                        if cand == best_set:
                            current = list(cand)
                            no_improve_count = 0

    return list(best_set)
