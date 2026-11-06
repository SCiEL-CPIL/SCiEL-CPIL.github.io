import time
import random
import bisect
from collections import defaultdict

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Precompute: for each a, which residues mod a are "blocked" by pairs (b,c) in set
    # But this is dynamic, so we compute on the fly with optimization
    
    def is_valid_addition(s_sorted, candidate):
        """
        Check if adding 'candidate' to 's_sorted' (sorted list) keeps validity.
        Optimized: use set-based residue checks where possible.
        """
        n = len(s_sorted)
        pos = bisect.bisect_left(s_sorted, candidate)
        
        # 1. candidate as 'a' (smallest)
        # b, c are from s_sorted[pos:]
        # For each b in s_sorted[pos:], check if any c in s_sorted[i:] has c ≡ -b (mod candidate)
        if candidate > 0:
            # Build residue set for s_sorted[pos:]
            residues = {}
            for k in range(pos, n):
                r = s_sorted[k] % candidate
                residues[r] = residues.get(r, 0) + 1
            
            for i in range(pos, n):
                b = s_sorted[i]
                target = (-b) % candidate
                # Check if any c in s_sorted[i:] has residue target
                # We need c >= b, so we can't just use the full residue count
                # But if target residue exists in s_sorted[i:], we need to check
                # For now, do the inner loop but break early
                for k in range(i, n):
                    if s_sorted[k] % candidate == target:
                        return False
        
        # 2. candidate as 'b' (middle)
        # a from s_sorted[:pos], c from s_sorted[pos:] U {candidate}
        for i in range(pos):
            a = s_sorted[i]
            # c can be candidate
            if (2 * candidate) % a == 0:
                return False
            # c can be s_sorted[k] for k >= pos
            target = (-candidate) % a
            for k in range(pos, n):
                if s_sorted[k] % a == target:
                    return False
        
        # 3. candidate as 'c' (largest)
        # a, b from s_sorted[:pos] with a < b
        for i in range(pos):
            a = s_sorted[i]
            for j in range(i + 1, pos):
                b = s_sorted[j]
                if (b + candidate) % a == 0:
                    return False
        
        return True

    def try_add(s_sorted, candidate):
        if is_valid_addition(s_sorted, candidate):
            pos = bisect.bisect_left(s_sorted, candidate)
            new_s = s_sorted[:]
            new_s.insert(pos, candidate)
            return True, new_s
        return False, s_sorted

    def greedy_with_order(order):
        """Greedy construction with given order of candidates."""
        s = []
        for candidate in order:
            ok, new_s = try_add(s, candidate)
            if ok:
                s = new_s
        return s

    # Multiple greedy strategies
    best_set = []
    best_len = 0
    
    # Strategy 1: High to low
    s1 = greedy_with_order(range(N, 0, -1))
    if len(s1) > best_len:
        best_set = s1
        best_len = len(s1)
    
    # Strategy 2: Randomized high to low (keep top half, randomize rest)
    for restart in range(3):
        if time.time() - start_time > deadline:
            break
        order = list(range(N, 0, -1))
        random.shuffle(order)
        # Bias toward larger numbers: sort by (value * random_factor)
        biased_order = sorted(order, key=lambda x: x * random.uniform(0.8, 1.2), reverse=True)
        s = greedy_with_order(biased_order)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
    
    # Strategy 3: Skip some small numbers
    for skip_frac in [0.1, 0.2, 0.3]:
        if time.time() - start_time > deadline:
            break
        order = list(range(N, 0, -1))
        # Remove bottom skip_frac fraction
        keep_from = int(N * skip_frac)
        order = [x for x in order if x >= keep_from] + [x for x in order if x < keep_from]
        s = greedy_with_order(order)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
    
    current_set = best_set[:]
    current_set_set = set(current_set)
    
    # Helper: try to add all possible missing elements
    def add_all_possible(s, s_set):
        changed = True
        while changed:
            changed = False
            for candidate in range(1, N + 1):
                if candidate not in s_set:
                    ok, new_s = try_add(s, candidate)
                    if ok:
                        s = new_s
                        s_set.add(candidate)
                        changed = True
        return s, s_set
    
    # Initial: add all possible
    current_set, current_set_set = add_all_possible(current_set, current_set_set)
    if len(current_set) > best_len:
        best_set = current_set[:]
        best_len = len(current_set)
    
    # Local search with swap heuristic
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < deadline:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # 1. Try single add
        missing = [x for x in range(1, N + 1) if x not in current_set_set]
        random.shuffle(missing)
        for candidate in missing:
            ok, new_s = try_add(current_set, candidate)
            if ok:
                current_set = new_s
                current_set_set.add(candidate)
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # 2. Try swap: remove one, try to add two
        if len(current_set) > 2:
            # Pick a random element to remove
            remove_val = random.choice(current_set)
            temp_set = [x for x in current_set if x != remove_val]
            temp_set_set = set(temp_set)
            
            # Try to add two missing elements
            missing = [x for x in range(1, N + 1) if x not in temp_set_set]
            random.shuffle(missing)
            
            added = 0
            new_set = temp_set[:]
            new_set_set = set(new_set)
            for candidate in missing:
                if added >= 2:
                    break
                ok, new_s = try_add(new_set, candidate)
                if ok:
                    new_set = new_s
                    new_set_set.add(candidate)
                    added += 1
            
            if added >= 2:
                # Net gain of 1
                current_set = new_set
                current_set_set = new_set_set
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)
        
        if not improved:
            # 3. Perturbation: remove k random, rebuild
            k = random.randint(3, 7)
            if len(current_set) <= k:
                k = max(1, len(current_set) // 3)
            
            remove_vals = random.sample(current_set, k)
            remove_set = set(remove_vals)
            perturbed = [x for x in current_set if x not in remove_set]
            perturbed_set = set(perturbed)
            
            # Rebuild with randomized greedy
            all_missing = [x for x in range(1, N + 1) if x not in perturbed_set]
            random.shuffle(all_missing)
            
            for candidate in all_missing:
                if time.time() - start_time > deadline:
                    break
                ok, new_s = try_add(perturbed, candidate)
                if ok:
                    perturbed = new_s
                    perturbed_set.add(candidate)
            
            if len(perturbed) > len(current_set):
                current_set = perturbed
                current_set_set = perturbed_set
                improved = True
                if len(current_set) > best_len:
                    best_set = current_set[:]
                    best_len = len(current_set)
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return best_set
