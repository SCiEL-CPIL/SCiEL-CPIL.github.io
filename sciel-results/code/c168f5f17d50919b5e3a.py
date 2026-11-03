import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit
    total = q ** n

    # Precompute neighbors (Hamming distance <= radius)
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]

    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        if radius == 1:
            temp = idx
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        neigh_set[idx].add(new_idx)
        else:
            # Fallback for general radius (slower)
            for j in range(total):
                dist = 0
                a = idx
                b = j
                for k in range(n):
                    da = (a // (q ** k)) % q
                    db = (b // (q ** k)) % q
                    if da != db:
                        dist += 1
                    if dist > radius:
                        break
                if dist <= radius:
                    neigh[idx].append(j)
                    neigh_set[idx].add(j)

    # Build Distance-2 Graph for Independent Set Initialization
    # Two words are connected if their Hamming distance is exactly 2.
    dist2_adj = [set() for _ in range(total)]
    
    for idx in range(total):
        for i in range(n):
            pow_q_i = q ** i
            digit_i = (idx // pow_q_i) % q
            for j in range(i + 1, n):
                pow_q_j = q ** j
                digit_j = (idx // pow_q_j) % q
                # Change digit i and j
                for new_i in range(q):
                    if new_i == digit_i:
                        continue
                    for new_j in range(q):
                        if new_j == digit_j:
                            continue
                        new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                        if new_idx != idx:
                            dist2_adj[idx].add(new_idx)
                            dist2_adj[new_idx].add(idx)

    rng = random.Random(42)

    # Helper: Check if a codeword is removable (all its neighbors are covered by others)
    def is_removable(c, counts):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    # Helper: Check if a swap is valid (removing old_c doesn't uncover any node that new_c doesn't cover)
    def is_valid_swap(old_c, new_c, counts, neigh_set_new):
        # If a neighbor u of old_c has count 1, it must be covered by new_c
        for u in neigh[old_c]:
            if counts[u] == 1 and u not in neigh_set_new:
                return False
        return True

    # Helper: Minimize the code by removing redundant codewords
    def minimize_code(code, counts):
        # code is a list of indices
        # counts is the coverage array
        # We want to remove any codeword that is redundant.
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                if is_removable(c, counts):
                    # Remove it
                    for u in neigh[c]:
                        counts[u] -= 1
                    code.pop(i)
                    changed = True
                    break
        return code

    # Helper: Repair the code to be a valid cover
    def repair_cover(code, code_set, counts):
        # Find uncovered words
        uncovered = [i for i in range(total) if counts[i] == 0]
        
        # Greedily add words to cover uncovered space
        while uncovered and time.time() < deadline - 5.0:
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if counts[u] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1:
                break
                
            code_set.add(best_c)
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
            
            # Recalculate uncovered
            uncovered = [i for i in range(total) if counts[i] == 0]
        
        # If still uncovered, force add (shouldn't happen usually)
        if uncovered:
            for u in uncovered:
                if u not in code_set:
                    code_set.add(u)
                    code.append(u)
                    for v in neigh[u]:
                        counts[v] += 1
                    # Update counts for all neighbors
                    # Note: this loop is O(total * neigh_size) which is fine for 729
        
        return code

    # --- Strategy 1: Greedy Independent Set Initialization ---
    
    def run_independent_set_strategy():
        candidates = list(range(total))
        rng.shuffle(candidates)
        
        code_set = set()
        in_set = [False] * total
        
        # Greedy Independent Set
        for v in candidates:
            if in_set[v]:
                continue
            # Check if v conflicts with any node in code_set
            # Conflict if u in code_set and u is in dist2_adj[v]
            is_conflict = False
            for u in dist2_adj[v]:
                if in_set[u]:
                    is_conflict = True
                    break
            if not is_conflict:
                code_set.add(v)
                in_set[v] = True
        
        code = list(code_set)
        
        # Compute counts
        counts = [0] * total
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        # Repair to Cover
        code = repair_cover(code, code_set, counts)
        
        # Minimize
        code = minimize_code(code, counts)
        
        return code, counts

    # --- Strategy 2: Simulated Annealing ---
    
    def simulated_annealing(initial_code, initial_counts, time_limit):
        code = initial_code[:]
        counts = initial_counts[:]
        code_set = set(code)
        
        best_code = code[:]
        best_size = len(code)
        
        temp = 1.0
        cool_rate = 0.9995
        min_temp = 0.01
        
        while time.time() < time_limit:
            if temp < min_temp:
                temp = min_temp
                
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c, counts):
                        # Perform removal
                        code.pop(idx)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            
            elif move_type < 0.9:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    if is_valid_swap(old_c, new_c, counts, neigh_set[new_c]):
                        # Perform swap
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        
                        # Update counts
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
            else:
                # Try Add (if we want to grow, but we want to minimize, so maybe skip or use for repair?)
                # Actually, adding increases size, so we only do it if it allows a future removal.
                # For now, let's stick to Remove and Swap.
                pass
                
            temp *= cool_rate
            
        # Final minimize
        code = minimize_code(code, counts)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        return best_code

    # --- Main Strategy ---
    
    # 1. Initialize with Independent Set
    code, counts = run_independent_set_strategy()
    best_code = code[:]
    best_size = len(code)
    
    # 2. Run Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sa_result = simulated_annealing(code, counts, deadline - 2.0)
        if len(sa_result) < best_size:
            best_code = sa_result
            best_size = len(best_code)
    
    # 3. Try a few random restarts of the Independent Set if time permits
    # This helps escape local minima in the SA process.
    attempts = 0
    max_attempts = 5
    while time.time() < deadline - 10.0 and attempts < max_attempts:
        attempts += 1
        new_code, new_counts = run_independent_set_strategy()
        if len(new_code) < best_size:
            best_code = new_code
            best_size = len(new_code)
        
        # Run SA on this new start
        remaining = deadline - time.time()
        if remaining > 10.0:
            sa_res = simulated_annealing(new_code, new_counts, deadline - 5.0)
            if len(sa_res) < best_size:
                best_code = sa_res
                best_size = len(best_code)
    
    # Final minimize
    # Recount for best_code to ensure consistency
    final_counts = [0] * total
    for c in best_code:
        for u in neigh[c]:
            final_counts[u] += 1
    best_code = minimize_code(best_code, final_counts)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result
