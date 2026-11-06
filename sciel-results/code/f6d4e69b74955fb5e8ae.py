import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s kill
    
    best_seq = []
    best_len = 0
    
    def update_state(seq):
        """
        Rebuilds prefix_sums and forbidden_sums from a sequence.
        """
        prefix_sums = [0]
        forbidden_sums = set()
        for x in seq:
            new_prefix = prefix_sums[-1] + x
            for p in prefix_sums:
                forbidden_sums.add(new_prefix - p)
            prefix_sums.append(new_prefix)
        return prefix_sums, forbidden_sums

    def is_valid_candidate(x, prefix_sums, forbidden_sums):
        """
        Checks if adding x to the current sequence (ending at prefix_sums[-1])
        results in a valid sequence.
        """
        new_prefix = prefix_sums[-1] + x
        for p in prefix_sums:
            d = new_prefix - p
            if d in forbidden_sums:
                return False
        return True

    def greedy_construct(seed=None, prob_smallest=0.8):
        if seed is not None:
            random.seed(seed)
        
        seq = []
        prefix_sums = [0]
        forbidden_sums = set()
        last = 0
        
        for _ in range(256):
            valid_candidates = []
            for x in range(last + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if random.random() < prob_smallest:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            seq.append(x)
            last = x
            new_prefix = prefix_sums[-1] + x
            prefix_sums.append(new_prefix)
            
            for p in prefix_sums[:-1]:
                forbidden_sums.add(new_prefix - p)
            
        return seq

    def is_valid_sequence(seq):
        if not seq:
            return True
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        if any(x < 1 or x > 256 for x in seq):
            return False
        p = [0]
        for x in seq:
            p.append(p[-1] + x)
        sums = set()
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                s = p[j] - p[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Phase 1: Randomized Greedy to get a strong baseline
    attempt_count = 0
    greedy_deadline = start_time + 10.0  # Reserve 10s for greedy
    while time.time() < greedy_deadline:
        # Vary the probability to explore different structures
        prob = 0.5 + 0.5 * random.random()
        seq = greedy_construct(seed=attempt_count, prob_smallest=prob)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        attempt_count += 1
        if attempt_count > 100: # Cap number of greedy attempts to avoid infinite loop if time is tight
            break

    # Phase 2: Target-Length DFS
    # We try to find a sequence of length best_len + 1, then +2, etc.
    
    def dfs_target(target_len, deadline):
        """
        Attempts to find a sequence of exactly target_len.
        Returns the sequence if found, else None.
        """
        # State: current sequence, prefix_sums, forbidden_sums, last_val
        # We use recursion with backtracking.
        
        # To make DFS faster, we can use a heuristic to order candidates.
        # Smaller values first? Or random?
        # Let's use a randomized order with a bias towards smaller values,
        # but also try larger values to escape local minima.
        
        # We will use a stack-based approach or recursion.
        # Recursion is cleaner.
        
        sys.setrecursionlimit(3000)
        
        def _dfs(depth, last_val, prefix_sums, forbidden_sums, current_seq):
            nonlocal best_seq, best_len
            
            if time.time() > deadline:
                return None
            
            # Pruning: Can we reach target_len?
            # Max additional elements: 256 - last_val
            if depth + (256 - last_val) < target_len:
                return None
            
            if depth == target_len:
                return current_seq[:]
            
            # Generate candidates
            candidates = []
            for x in range(last_val + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_sums):
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Heuristic: Sort candidates?
            # Trying smaller ones first usually leads to longer sequences in these problems?
            # Or does it? Smaller sums might collide more easily?
            # Let's try a mixed strategy: Randomize the order, but bias towards
            # values that are "spread out".
            
            # Simple heuristic: Try smaller values first, but with some randomness.
            # Actually, for DFS to find *a* solution, the order matters less than
            # the pruning. But to find it *fast*, we need a good order.
            
            # Let's try: Sort by (x - last_val) to keep increments small?
            # Or just random?
            # Random is safer against deterministic traps.
            random.shuffle(candidates)
            
            for x in candidates:
                new_prefix = prefix_sums[-1] + x
                new_forbidden = set(forbidden_sums) # Copy is expensive!
                
                # Optimization: We don't need to copy the whole set if we backtrack carefully.
                # But sets are hard to backtrack incrementally without storing diffs.
                # Let's store the new diffs.
                
                new_diffs = []
                valid = True
                for p in prefix_sums:
                    d = new_prefix - p
                    if d in forbidden_sums:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Commit
                prefix_sums.append(new_prefix)
                forbidden_sums.update(new_diffs)
                current_seq.append(x)
                
                result = _dfs(depth + 1, x, prefix_sums, forbidden_sums, current_seq)
                
                if result is not None:
                    return result
                
                # Backtrack
                current_seq.pop()
                for d in new_diffs:
                    forbidden_sums.discard(d)
                prefix_sums.pop()
                
                if time.time() > deadline:
                    return None
            
            return None

        # Start DFS from empty sequence?
        # Or from the best greedy sequence?
        # Starting from empty is more thorough but slower.
        # Starting from a prefix of the best greedy sequence might be faster.
        
        # Let's try starting from empty with a strong heuristic.
        # To speed up, we can use the best greedy sequence as a "hint" for the order?
        # No, let's just run DFS from empty.
        
        # However, a full DFS from empty might be too slow if the branching factor is high.
        # We can limit the depth of the initial branching?
        
        # Alternative: Use the best greedy sequence as the starting point for DFS?
        # If we start from the best greedy sequence, we are already at depth L.
        # We want to find L+1.
        # So we can try to extend the best greedy sequence using DFS.
        # If that fails, we can try to modify the last few elements of the greedy sequence
        # and then extend.
        
        # Let's implement a "Guided DFS" that starts from the best greedy sequence.
        
        # 1. Try to extend the best greedy sequence directly.
        if best_seq:
            prefix_sums, forbidden_sums = update_state(best_seq)
            last_val = best_seq[-1]
            
            # We want to find target_len.
            # If target_len <= len(best_seq), we already have it.
            # So we call dfs_target only if target_len > len(best_seq).
            
            # We modify _dfs to start from the current state.
            # The _dfs above starts from depth 0. We need a version that starts from current state.
            
            pass # We will implement a specific extension DFS below.

    # Let's implement a specific extension DFS that starts from a given sequence.
    
    def extend_sequence(seq, target_len, deadline):
        """
        Tries to extend 'seq' to length 'target_len' using DFS.
        """
        if len(seq) >= target_len:
            return seq[:]
        
        prefix_sums, forbidden_sums = update_state(seq)
        last_val = seq[-1] if seq else 0
        current_seq = seq[:]
        
        def _dfs_ext(depth, last_val, prefix_sums, forbidden_sums, current_seq):
            if time.time() > deadline:
                return None
            
            if depth + (256 - last_val) < target_len:
                return None
            
            if depth == target_len:
                return current_seq[:]
            
            candidates = []
            for x in range(last_val + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_sums):
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Heuristic: Try smaller values first, but shuffle to avoid bias
            # Actually, let's try a few different orders in different runs.
            # For now, random shuffle.
            random.shuffle(candidates)
            
            for x in candidates:
                new_prefix = prefix_sums[-1] + x
                new_diffs = []
                valid = True
                for p in prefix_sums:
                    d = new_prefix - p
                    if d in forbidden_sums:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                prefix_sums.append(new_prefix)
                forbidden_sums.update(new_diffs)
                current_seq.append(x)
                
                result = _dfs_ext(depth + 1, x, prefix_sums, forbidden_sums, current_seq)
                
                if result is not None:
                    return result
                
                current_seq.pop()
                for d in new_diffs:
                    forbidden_sums.discard(d)
                prefix_sums.pop()
                
                if time.time() > deadline:
                    return None
            
            return None

        return _dfs_ext(len(current_seq), last_val, prefix_sums, forbidden_sums, current_seq)

    # Phase 2 Execution:
    # We try to increase best_len one by one.
    # We use the best greedy sequence as a starting point for extension.
    # If extension fails, we might try other greedy sequences or a fresh DFS.
    
    dfs_deadline = start_time + time_limit - 2.0
    
    current_target = best_len + 1
    while current_target <= 256 and time.time() < dfs_deadline:
        # Try to extend the current best_seq to current_target
        # To increase diversity, we can try extending different greedy sequences.
        # But we only kept the best one.
        
        # Let's try extending the best_seq.
        # If it fails, we might need to backtrack further in the greedy sequence.
        # This is complex.
        
        # Alternative: Run a fresh DFS from scratch for the target length?
        # This is expensive.
        
        # Let's try a "Perturbed Extension":
        # Take the best_seq, remove the last k elements, and try to extend to target_len.
        
        found = False
        # Try removing 0, 1, 2, ... elements from the end of best_seq
        for k in range(len(best_seq) + 1):
            if time.time() > dfs_deadline:
                break
            
            base_seq = best_seq[:len(best_seq) - k]
            if len(base_seq) >= current_target:
                continue # We already have a longer sequence
            
            result = extend_sequence(base_seq, current_target, dfs_deadline)
            if result is not None:
                best_seq = result
                best_len = current_target
                found = True
                break
        
        if not found:
            # If we couldn't extend the best_seq (even after truncation),
            # we might be stuck.
            # We could try a fresh greedy + extend loop.
            # But for now, if we can't extend the best, we stop.
            # Or we can try a completely fresh DFS from scratch?
            # Let's try one fresh DFS attempt if time permits.
            
            if time.time() < dfs_deadline - 5:
                # Fresh DFS from scratch
                # This is risky but might find a different structure.
                # We'll use a limited depth or time.
                
                # Actually, the "Perturbed Extension" is quite powerful.
                # If it fails, it's likely that current_target is not reachable
                # from the current best structure.
                
                # We can try a different greedy seed to get a different base structure.
                new_seed = random.randint(0, 10000)
                new_greedy = greedy_construct(seed=new_seed, prob_smallest=0.7)
                if len(new_greedy) > best_len:
                    best_seq = new_greedy
                    best_len = len(new_greedy)
                    current_target = best_len + 1
                    continue # Retry with new best
                
                # If no improvement, break the loop
                break
        
        current_target += 1

    return best_seq
