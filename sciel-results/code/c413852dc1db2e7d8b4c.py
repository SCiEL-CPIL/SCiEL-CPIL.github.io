import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    def greedy_greedy(order):
        chosen = []
        counts = {}
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        return chosen

    # 1. Deterministic Greedy
    best_set = greedy_greedy(range(1, 1001))
    best_size = len(best_set)
    
    # 2. Randomized Greedy
    # Run until 10s before deadline
    while time.time() < deadline - 10:
        order = list(range(1, 1001))
        random.shuffle(order)
        s = greedy_greedy(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    # 3. Local Search on best_set
    S = set(best_set)
    # Rebuild counts
    counts = {}
    lst = list(S)
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            b = lst[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            
    # Local search loop
    while time.time() < deadline - 1:
        # Try to add
        candidates = list(set(range(1, 1001)) - S)
        if not candidates:
            break
        random.shuffle(candidates)
        added = False
        for x in candidates:
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    valid = False
            if valid:
                # Add x
                S.add(x)
                for y in S:
                    if y == x: continue
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                added = True
                break
        
        if added:
            continue
            
        # Try swap
        if S and candidates:
            x_out = random.choice(list(S))
            x_in = random.choice(candidates)
            
            # Check validity of swap
            # Temporarily remove x_out
            # We don't want to modify S/counts if invalid
            # Check:
            # 1. Removing x_out reduces counts of sums involving x_out.
            # 2. Adding x_in increases counts of sums involving x_in.
            # We need to ensure that for all sums s, new_count(s) <= 2.
            
            # Sums affected by removal: x_out + y for y in S\{x_out}, and 2*x_out
            # Sums affected by addition: x_in + y for y in S\{x_out}, and 2*x_in
            # Note: S\{x_out} is the base set for both.
            
            # Let's just simulate on a copy? Too slow.
            # Let's check manually.
            
            # Sums that might become > 2 are those that are increased by addition.
            # Sums involving x_in:
            # 2*x_in
            # x_in + y for y in S\{x_out}
            
            # For each such sum s:
            # current_count = counts[s]
            # delta = 0
            # If s is also a sum involving x_out (i.e., s = x_out + y' or 2*x_out), 
            # then removal decreases it by 1.
            
            # So new_count = counts[s] + 1 - (1 if s involves x_out else 0)
            # We need new_count <= 2.
            
            valid_swap = True
            # Check sums involving x_in
            s = 2 * x_in
            delta = 0
            # Does s involve x_out?
            # s = 2*x_in. Involves x_out if 2*x_in = 2*x_out (x_in=x_out, impossible) 
            # or 2*x_in = x_out + y.
            # This is getting complicated.
            
            # Simpler: Just try the swap on a copy of the state?
            # No, state is small. 
            # But doing it every iteration is slow.
            
            # Let's skip complex swap logic and just do random removal + random addition attempts.
            # Remove x_out
            S.remove(x_out)
            for y in S:
                s = x_out + y
                counts[s] -= 1
                if counts[s] == 0: del counts[s]
            s = 2 * x_out
            counts[s] -= 1
            if counts[s] == 0: del counts[s]
            
            # Try to add x_in
            valid = True
            for y in S:
                s = x_in + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x_in
                if counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                S.add(x_in)
                for y in S:
                    if y == x_in: continue
                    s = x_in + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x_in
                counts[s] = counts.get(s, 0) + 1
            else:
                # Revert
                S.add(x_out)
                for y in S:
                    if y == x_out: continue
                    s = x_out + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x_out
                counts[s] = counts.get(s, 0) + 1

    return list(S)
