"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 55.0  # 5s buffer before 60s limit

    def get_valid_candidates(values, sums, last_val):
        """
        Returns a list of all x in (last_val, 256] such that appending x 
        to values does not create any duplicate consecutive block sums.
        """
        valid = []
        # We need to check x from last_val + 1 to 256
        for x in range(last_val + 1, 257):
            # Calculate new tails (suffix sums ending at x)
            # tails[0] = x
            # tails[1] = x + values[-1]
            # tails[2] = x + values[-1] + values[-2]
            # ...
            tails = [0] * (len(values) + 1)
            total = x
            tails[0] = total
            for i in range(len(values) - 1, -1, -1):
                total += values[i]
                tails[len(values) - i] = total
            
            # Check for collision
            # If any tail is in sums, collision
            if not sums.intersection(tails):
                valid.append(x)
        
        return valid

    def greedy_increasing():
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            # Greedy: pick smallest
            x = candidates[0]
            values.append(x)
            last_val = x
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        return values

    def greedy_decreasing():
        """
        Construct a strictly decreasing sequence b_1 > b_2 > ... > b_k
        such that consecutive sums are distinct.
        Then reverse it to get increasing a_1 < ... < a_k.
        The set of consecutive sums is invariant under reversal.
        """
        values = []
        sums = set()
        next_val = 257
        while next_val > 1:
            candidates = []
            for x in range(next_val - 1, 0, -1):
                # Check if x can be appended to the *right* of current decreasing sequence?
                # Wait, if we build decreasing b_1, b_2, ...
                # b_1 is largest.
                # Next b_2 < b_1.
                # The new sums added by appending b_2 to [b_1] are:
                # b_2, b_2+b_1.
                # This is the same logic as increasing, just values are decreasing.
                # Let's just reuse the logic by mapping values to 257-x?
                # No, simpler: just run the same candidate check but iterating downwards.
                
                tails = [0] * (len(values) + 1)
                total = x
                tails[0] = total
                for i in range(len(values) - 1, -1, -1):
                    total += values[i]
                    tails[len(values) - i] = total
                
                if not sums.intersection(tails):
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Greedy decreasing: pick largest valid (which is candidates[0] since we iterated down? 
            # No, range(next_val-1, 0, -1) goes 256, 255...
            # So candidates[0] is the largest valid.
            x = candidates[0]
            values.append(x)
            next_val = x
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        
        # Reverse to get increasing sequence
        return values[::-1]

    def randomized_trial(strategy, rand):
        values = []
        sums = set()
        last_val = 0
        
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            
            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                # Pick from the largest 10% or top 10
                k = min(10, len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                # Pick from the smallest 10% or top 10
                k = min(10, len(candidates))
                x = rand.choice(candidates[:k])
            elif strategy == 'backtrack':
                # This is a heuristic "pseudo-backtrack"
                # If we are stuck, we can't actually backtrack in this linear scan.
                # But we can simulate a "lookahead" by picking a value that leaves the most options.
                # For now, just random, but maybe biased towards values that are "safe".
                # A value x is safer if it doesn't collide with many potential future sums?
                # Hard to compute. Let's just use random for now but maybe with a twist.
                # Twist: Pick x such that (x + sum(values)) is far from existing sums?
                # Or just random.
                x = rand.choice(candidates)
            else:
                x = rand.choice(candidates)
                
            values.append(x)
            last_val = x
            
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
                
        return values

    def dfs_construct(depth_limit, time_deadline):
        """
        Depth-first search with a limit on depth to explore non-greedy branches.
        We try to find a sequence of length > current best.
        """
        best_seq = []
        
        def dfs(values, sums, last_val, depth):
            nonlocal best_seq
            if time.time() > time_deadline:
                return
            if len(values) > len(best_seq):
                best_seq = values[:]
            
            if depth >= depth_limit:
                return
                
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                return
            
            # Heuristic ordering: try smaller first (greedy-like) but allow some randomness?
            # No, DFS needs determinism or a fixed order to be effective without randomness.
            # But pure greedy is already tried. 
            # Maybe we can use DFS to *verify* if a longer sequence exists by exploring
            # branches where we skip the first candidate?
            
            # To make DFS useful, we should explore branches that deviate from greedy.
            # Greedy picks candidates[0].
            # Let's iterate through candidates, but maybe skip the first one sometimes?
            # Or just iterate all, but prune if we know we can't beat best_len?
            
            # Pruning: If len(values) + (256 - last_val) <= len(best_seq), we can't beat it.
            # This is a weak bound.
            
            for x in candidates:
                # If we have already found a sequence of length L, and we are at depth D,
                # can we reach L?
                # Max possible length from here is len(values) + (256 - last_val)
                if len(values) + (256 - last_val) <= len(best_seq):
                    break # Since candidates are increasing, and last_val increases, this bound only gets worse?
                    # Actually, if we pick a larger x, last_val increases, so (256-last_val) decreases.
                    # So if the bound fails for the smallest x, it fails for all larger x.
                    
                # New sums
                new_sums = set(sums)
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                
                dfs(values + [x], new_sums, x, depth + 1)
                
                if time.time() > time_deadline:
                    return

        dfs([], set(), 0, 0)
        return best_seq

    # 1. Baselines
    best_seq = greedy_increasing()
    best_len = len(best_seq)
    
    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)
        
    # 2. Randomized Search
    # We use a mix of strategies to explore the space
    strategies = ['random', 'large', 'small', 'random', 'large', 'backtrack']
    
    trial_count = 0
    while time.time() - start_time < time_limit - 5.0: # Reserve 5s for DFS if possible
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count)
        
        seq = randomized_trial(strategy, rand)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
        trial_count += 1
        
    # 3. DFS Attempt (if time permits)
    # Try to find a sequence slightly longer than best_len using DFS
    # DFS is expensive, so we only try if we have time.
    if time.time() - start_time < time_limit - 1.0:
        # Try to find a sequence of length best_len + 1
        # Depth limit should be best_len + 1
        dfs_seq = dfs_construct(best_len + 1, time.time() + (time_limit - (time.time() - start_time)))
        if len(dfs_seq) > best_len:
            best_seq = dfs_seq
            best_len = len(best_seq)
            
    return best_seq

# EVOLVE-BLOCK-END
