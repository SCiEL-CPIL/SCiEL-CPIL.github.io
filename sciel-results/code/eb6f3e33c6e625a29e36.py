"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    WALL_LIMIT = 90.0
    SAFETY_MARGIN = 8.0
    deadline = start_time + WALL_LIMIT - SAFETY_MARGIN

    def get_valid_candidates(values, sums, last_val):
        """Returns all x in (last_val, 256] that don't create duplicate consecutive sums."""
        valid = []
        n = len(values)
        # Precompute suffix sums of values for faster checking
        # suffix[i] = values[i] + values[i+1] + ... + values[n-1]
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + values[i]
        
        for x in range(last_val + 1, 257):
            # New sums are: x, x+values[-1], x+values[-1]+values[-2], ..., x+sum(values)
            # i.e., x + suffix[i] for i = n, n-1, ..., 0 (where suffix[n]=0)
            # suffix[n] = 0, so x + 0 = x
            # suffix[n-1] = values[n-1], so x + values[n-1]
            # etc.
            ok = True
            for i in range(n, -1, -1):
                s = x + suffix[i]
                if s in sums:
                    ok = False
                    break
            if ok:
                valid.append(x)
        return valid

    def add_element(values, sums, x):
        """Add x to end of values, update sums. Returns new sums set or None if invalid."""
        n = len(values)
        # New sums: x, x+values[-1], x+values[-1]+values[-2], ...
        new_sums = set()
        total = x
        new_sums.add(total)
        for i in range(n - 1, -1, -1):
            total += values[i]
            new_sums.add(total)
        if sums.intersection(new_sums):
            return None
        sums.update(new_sums)
        values.append(x)
        return sums

    def remove_last(values, sums):
        """Remove last element from values, update sums."""
        if not values:
            return
        x = values.pop()
        n = len(values)
        # Remove sums that involved x: x, x+values[-1], x+values[-1]+values[-2], ...
        total = x
        sums.discard(total)
        for i in range(n - 1, -1, -1):
            total += values[i]
            sums.discard(total)

    def rebuild_sums(values):
        """Rebuild the sums set from scratch."""
        sums = set()
        n = len(values)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += values[j]
                sums.add(s)
        return sums

    def is_valid_sequence(values):
        """Check if all consecutive block sums are distinct."""
        sums = set()
        n = len(values)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += values[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_increasing():
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            x = candidates[0]
            add_element(values, sums, x)
            last_val = x
        return values

    def greedy_largest_first():
        """Try picking larger candidates first to see if it yields longer sequences."""
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            x = candidates[-1]  # Pick largest
            add_element(values, sums, x)
            last_val = x
        return values

    def greedy_median():
        """Pick median candidate."""
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            x = candidates[len(candidates) // 2]
            add_element(values, sums, x)
            last_val = x
        return values

    def randomized_trial(strategy, seed):
        rand = random.Random(seed)
        values = []
        sums = set()
        last_val = 0
        
        while last_val < 256:
            if time.time() > deadline:
                break
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            
            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                k = min(max(3, len(candidates) // 4), len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                k = min(max(3, len(candidates) // 4), len(candidates))
                x = rand.choice(candidates[:k])
            elif strategy == 'median':
                k = max(3, len(candidates) // 3)
                mid = len(candidates) // 2
                lo = max(0, mid - k // 2)
                hi = min(len(candidates), mid + k // 2)
                x = rand.choice(candidates[lo:hi])
            elif strategy == 'spread':
                # Pick from evenly spaced positions
                if len(candidates) <= 5:
                    x = rand.choice(candidates)
                else:
                    n_cand = len(candidates)
                    positions = [int(i * (n_cand - 1) / 4) for i in range(5)]
                    x = candidates[rand.choice(positions)]
            elif strategy == 'safety':
                # Score candidates by how far their new sums are from existing sums
                best_x = candidates[0]
                best_score = -1
                sample = candidates if len(candidates) <= 20 else rand.sample(candidates, 20)
                for x in sample:
                    # Compute min distance of new sums to existing sums
                    total = x
                    min_dist = abs(total) if total not in sums else 0
                    for i in range(len(values) - 1, -1, -1):
                        total += values[i]
                        if total in sums:
                            min_dist = 0
                            break
                        d = min(abs(total - s) for s in sums) if sums else total
                        if d < min_dist:
                            min_dist = d
                    if min_dist > best_score:
                        best_score = min_dist
                        best_x = x
                x = best_x
            else:
                x = rand.choice(candidates)
                
            add_element(values, sums, x)
            last_val = x
            
        return values

    def local_search(values, sums, time_budget=5.0):
        """Apply local search mutations to try to extend the sequence."""
        if time.time() > deadline:
            return values, sums
        
        best_values = values[:]
        best_sums = sums
        best_len = len(values)
        end_time = time.time() + time_budget
        
        # Work on a copy
        vals = values[:]
        sm = rebuild_sums(vals)
        
        while time.time() < end_time and time.time() < deadline:
            improved = False
            
            # Try 1: Remove an element and try to extend
            if len(vals) > 1:
                for remove_idx in range(len(vals) - 1, -1, -1):
                    if time.time() > end_time or time.time() > deadline:
                        break
                    # Remove element at remove_idx
                    x = vals.pop(remove_idx)
                    sm = rebuild_sums(vals)
                    
                    # Now try to extend
                    last_val = vals[-1] if vals else 0
                    candidates = get_valid_candidates(vals, sm, last_val)
                    if candidates:
                        # Try each candidate
                        for new_x in candidates:
                            if new_x <= last_val:
                                continue
                            # Check if it creates a valid sequence
                            test_vals = vals[:] + [new_x]
                            if is_valid_sequence(test_vals):
                                if len(test_vals) > best_len:
                                    best_len = len(test_vals)
                                    best_values = test_vals[:]
                                    best_sums = rebuild_sums(test_vals)
                                    vals = test_vals[:]
                                    sm = best_sums
                                    improved = True
                                break
                    
                    # Restore
                    vals.insert(remove_idx, x)
                    sm = rebuild_sums(vals)
                    if improved:
                        break
            
            if not improved:
                # Try 2: Swap adjacent elements and see if we can extend
                if len(vals) >= 2:
                    swapped = False
                    for i in range(len(vals) - 1):
                        if time.time() > end_time or time.time() > deadline:
                            break
                        # Swap vals[i] and vals[i+1]
                        vals[i], vals[i+1] = vals[i+1], vals[i]
                        if is_valid_sequence(vals):
                            # Try to extend
                            last_val = vals[-1]
                            candidates = get_valid_candidates(vals, rebuild_sums(vals), last_val)
                            if candidates:
                                for new_x in candidates:
                                    test_vals = vals[:] + [new_x]
                                    if is_valid_sequence(test_vals):
                                        if len(test_vals) > best_len:
                                            best_len = len(test_vals)
                                            best_values = test_vals[:]
                                            best_sums = rebuild_sums(test_vals)
                                            vals = test_vals[:]
                                            improved = True
                                        break
                        # Undo swap
                        vals[i], vals[i+1] = vals[i+1], vals[i]
                        if improved:
                            swapped = True
                            break
                    if not swapped:
                        break
            
            if not improved:
                break
        
        return best_values, best_sums

    def dfs_construct(target_len, time_budget=10.0):
        """DFS to find a sequence of at least target_len."""
        best_seq = []
        end_time = time.time() + time_budget
        
        def dfs(values, sums, last_val, depth):
            nonlocal best_seq
            if time.time() > deadline or time.time() > end_time:
                return
            if len(values) > len(best_seq):
                best_seq = values[:]
            if len(values) >= target_len:
                return
            if last_val >= 256:
                return
            
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                return
            
            # Pruning: can we possibly reach target_len?
            # Max additional elements is at most 256 - last_val
            if len(values) + (256 - last_val) < target_len:
                return
            
            # Try candidates in order (smallest first for greedy-like behavior)
            for x in candidates:
                if time.time() > deadline or time.time() > end_time:
                    return
                
                new_sums = set(sums)
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                
                if not sums.intersection(new_sums):
                    dfs(values + [x], new_sums, x, depth + 1)
                
                if len(best_seq) >= target_len:
                    return
        
        dfs([], set(), 0, 0)
        return best_seq

    # Phase 1: Greedy baselines
    best_seq = greedy_increasing()
    best_len = len(best_seq)
    
    seq2 = greedy_largest_first()
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)
    
    seq3 = greedy_median()
    if len(seq3) > best_len:
        best_seq = seq3
        best_len = len(seq3)
    
    # Phase 2: Randomized trials with diverse strategies
    strategies = ['random', 'large', 'small', 'median', 'spread', 'safety']
    trial_count = 0
    while time.time() < deadline - 15.0:
        strategy = strategies[trial_count % len(strategies)]
        seq = randomized_trial(strategy, trial_count * 7919 + 13)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        trial_count += 1
    
    # Phase 3: Local search on best sequence
    if time.time() < deadline - 10.0:
        best_sums = rebuild_sums(best_seq)
        ls_seq, ls_sums = local_search(best_seq, best_sums, time_budget=min(5.0, deadline - time.time() - 5.0))
        if len(ls_seq) > best_len:
            best_len = len(ls_seq)
            best_seq = ls_seq
    
    # Phase 4: DFS to try to beat current best
    if time.time() < deadline - 5.0:
        target = best_len + 1
        dfs_budget = min(10.0, deadline - time.time() - 3.0)
        if dfs_budget > 2.0:
            dfs_seq = dfs_construct(target, time_budget=dfs_budget)
            if len(dfs_seq) > best_len:
                best_seq = dfs_seq
                best_len = len(dfs_seq)
                # If we found one, try for even longer
                if time.time() < deadline - 3.0:
                    target2 = best_len + 1
                    dfs_budget2 = min(5.0, deadline - time.time() - 2.0)
                    if dfs_budget2 > 1.0:
                        dfs_seq2 = dfs_construct(target2, time_budget=dfs_budget2)
                        if len(dfs_seq2) > best_len:
                            best_seq = dfs_seq2
                            best_len = len(dfs_seq2)
    
    # Final validation
    if not is_valid_sequence(best_seq):
        # Fallback to greedy
        best_seq = greedy_increasing()
    
    return best_seq

# EVOLVE-BLOCK-END
