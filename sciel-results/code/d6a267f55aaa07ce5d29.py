import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    best_seq = []
    
    def check_and_get_sums(seq, sums, x):
        # seq is list, sums is set of existing sums
        # returns new_sums set if valid, else None
        new_sums = set(sums)
        # Add sums ending at x
        # Sum of x
        # Sum of last + x
        # etc.
        # Actually, we can just compute the new sums added.
        added_sums = set()
        current_sum = 0
        # Iterate backwards from x
        # x is the new element.
        # Sums are: x, x + seq[-1], x + seq[-1] + seq[-2], ...
        # We need to check if any of these are in `sums`.
        # And we need to add them to `sums` for the next step.
        
        # Optimization: 
        # The new sums are $S_k = x + \sum_{i=k-1}^{n-1} seq[i]$ for $k=0 \dots n-1$
        # where $seq$ is the previous sequence.
        
        prev_seq = seq
        s = 0
        # The sums are x, x+prev_seq[-1], x+prev_seq[-1]+prev_seq[-2], ...
        # Let's just compute them.
        for i in range(len(prev_seq) - 1, -1, -1):
            s += prev_seq[i]
            new_val = x + s
            if new_val in sums:
                return None
            added_sums.add(new_val)
        
        # Also add x itself?
        # The loop above adds x + 0? No.
        # If prev_seq is empty, loop doesn't run.
        # We must add x itself.
        added_sums.add(x)
        
        # Check if x is in sums?
        if x in sums:
            return None
            
        # Check if any added_sums are in sums? (Already done in loop for x+..., but not for x)
        # Actually, the loop checks x + sum(suffix).
        # If suffix is empty, sum is 0, so x.
        # So the loop should handle x if we start with s=0 and add x?
        # Let's rewrite clearly.
        
        return added_sums

    def search(seq, sums, min_val):
        nonlocal best_seq, deadline
        
        if time.time() > deadline:
            return
            
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # Generate candidates
        candidates = []
        for x in range(min_val, 257):
            # Check validity
            # We need to check if x creates duplicate sums.
            # New sums: x, x+seq[-1], x+seq[-1]+seq[-2], ...
            
            # Fast check:
            # 1. x not in sums
            if x in sums:
                continue
                
            new_sums = set()
            s = 0
            valid = True
            # Check sums ending at x
            # x
            new_sums.add(x)
            # x + seq[-1]
            # x + seq[-1] + seq[-2]
            # ...
            for i in range(len(seq) - 1, -1, -1):
                s += seq[i]
                val = x + s
                if val in sums:
                    valid = False
                    break
                new_sums.add(val)
                
            if not valid:
                continue
                
            # Check if new_sums has internal duplicates?
            # Since seq is increasing, the suffix sums are strictly increasing.
            # So x + suffix_sum are strictly increasing.
            # And x is the smallest.
            # So no internal duplicates.
            
            candidates.append((x, new_sums))
            
        if not candidates:
            return
            
        # Pruning: 1-step lookahead
        # For each candidate, check if it allows at least one more step.
        live_candidates = []
        for x, new_sums in candidates:
            # Check if there exists y > x such that y is valid for seq + [x]
            # New sums for y would be y, y+x, y+x+seq[-1], ...
            # We just need to know if ANY y exists.
            
            # To speed up, we can just check the smallest possible y > x?
            # No, we need to check if *any* y works.
            # This check is O(256 * len(seq)).
            # Candidates count is at most 256.
            # Total check cost: 256 * 256 * len(seq) ~ 65k * 15 ~ 1M ops.
            # This is fast enough.
            
            found_next = False
            # We can optimize by checking y in range(x+1, 257)
            # But we can break early if we find one.
            
            # Actually, we can just try to find the smallest valid y > x.
            # If we find one, it's live.
            
            # Let's just iterate y.
            for y in range(x + 1, 257):
                if y in (sums | new_sums):
                    continue
                    
                # Check if y is valid given seq + [x]
                # Sums: y, y+x, y+x+seq[-1], ...
                valid_y = True
                s_y = 0
                # y
                # y + x
                # y + x + seq[-1]
                # ...
                
                # Check y
                # Check y + x
                # Check y + x + seq[-1]
                # ...
                
                # We can reuse the logic.
                # Sums to check:
                # y
                # y + x
                # y + x + seq[-1]
                # y + x + seq[-1] + seq[-2]
                # ...
                
                # Note: y+x is in new_sums?
                # new_sums contains x, x+seq[-1], ...
                # y+x is NOT in new_sums unless y=x (impossible).
                # But y+x might be in `sums`.
                
                # Let's just do the full check for y.
                s_y = 0
                # y
                if y in sums or y in new_sums: continue # y in new_sums is false as y>x.
                
                # y + x
                val = y + x
                if val in sums or val in new_sums:
                    valid_y = False
                    break
                s_y += x
                
                # y + x + seq[-1]
                for i in range(len(seq) - 1, -1, -1):
                    s_y += seq[i]
                    val = y + s_y
                    if val in sums or val in new_sums:
                        valid_y = False
                        break
                
                if valid_y:
                    found_next = True
                    break
                    
            if found_next:
                live_candidates.append((x, new_sums))
                
        if not live_candidates:
            # No live candidates, but we might still be at a local max
            # If we are here, we can't extend.
            return
            
        # Sort live candidates
        # Heuristic: Smallest x first?
        # Or maybe the one that leaves the most options? (Too expensive)
        # Let's try Smallest x first.
        live_candidates.sort(key=lambda item: item[0])
        
        # Explore the best candidate
        # We can explore more than one if time permits.
        # But DFS on the first one is a good start.
        # To beat the seed, we just need to find a longer path.
        # The seed is Greedy Smallest.
        # If Greedy Smallest gets stuck at length L, and we skip the bad choice, we might get L+1.
        
        # Let's explore the first live candidate.
        x, new_sums = live_candidates[0]
        next_seq = seq + [x]
        next_sums = sums | new_sums
        search(next_seq, next_sums, x + 1)
        
        # If time remains, explore the next best candidate?
        # This turns it into a beam search / DFS with backtracking.
        # Given the time limit, we should be careful.
        # Let's add a counter or time check to limit branching.
        
        # For now, let's just do the first one. If we want to improve, we can try the second one if time > 20s left.
        
        if time.time() < deadline - 10:
            if len(live_candidates) > 1:
                x2, new_sums2 = live_candidates[1]
                next_seq2 = seq + [x2]
                next_sums2 = sums | new_sums2
                search(next_seq2, next_sums2, x2 + 1)

    # Initial call
    # We can try starting with different values?
    # The seed starts with 1.
    # If we start with 1, we are constrained.
    # What if we start with 2?
    # The search function handles the choice of x.
    # We just call search([], set(), 1).
    
    # However, the "Greedy Smallest" seed always picks 1 if valid.
    # Is 1 always valid? Yes, sums={1}.
    # So the seed starts with 1.
    # Our search will also consider 1.
    # If 1 is a dead end (no next move), we will skip it and try 2.
    # This is the key improvement.
    
    search([], set(), 1)
    
    return best_seq
