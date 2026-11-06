import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    N = 256

    def check_valid(seq):
        """Return True if all non-empty consecutive block sums are distinct."""
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def bidirectional_construct(values, max_len=None):
        """
        Build a strictly increasing sequence by extending from both ends.
        values: sorted list of candidate values.
        Strategy: at each step, try to insert the smallest unused value at the front
        or the largest unused value at the back, whichever keeps validity.
        Use a running set of block sums for efficiency.
        """
        if max_len is None:
            max_len = len(values)
        
        # We'll build the sequence incrementally
        # Start with a single middle-ish element
        if not values:
            return []
        
        # Try multiple starting points
        best_seq = []
        best_len = 0
        
        # Starting candidates: every kth element
        start_indices = list(range(0, len(values), max(1, len(values)//20)))
        if not start_indices:
            start_indices = [0]
        
        for si in start_indices:
            if time.time() > deadline - 2:
                break
            start_val = values[si]
            seq = [start_val]
            seen_sums = {start_val}
            left_idx = si - 1
            right_idx = si + 1
            
            while len(seq) < max_len and (left_idx >= 0 or right_idx < len(values)):
                if time.time() > deadline - 2:
                    break
                
                added = False
                # Try adding from the right (larger values) first
                if right_idx < len(values):
                    candidate = values[right_idx]
                    new_seq = seq + [candidate]
                    # Check new block sums involving the new element
                    valid = True
                    new_sums = set()
                    # Sums ending at new position
                    s = 0
                    for j in range(len(seq), -1, -1):
                        s += new_seq[j]
                        if s in seen_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if valid:
                        # Also need to check sums that start before and end at new pos
                        # Actually, all new sums are those that include the new element
                        # and end at the new position (since it's appended at end)
                        # Wait, no: consecutive blocks in new_seq that include the new element
                        # are those starting at i (0<=i<=len(seq)) and ending at len(seq)
                        # We already checked those. But we also need to make sure
                        # no existing sum conflicts with these new sums.
                        # The new sums are exactly those computed above.
                        if not any(s in seen_sums for s in new_sums):
                            seq = new_seq
                            seen_sums.update(new_sums)
                            right_idx += 1
                            added = True
                
                if not added and left_idx >= 0:
                    candidate = values[left_idx]
                    new_seq = [candidate] + seq
                    # New block sums are those starting at position 0 and ending at j (0<=j<=len(seq)-1)
                    valid = True
                    new_sums = set()
                    s = 0
                    for j in range(len(new_seq)):
                        s += new_seq[j]
                        if s in seen_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if valid and not any(s in seen_sums for s in new_sums):
                        seq = new_seq
                        seen_sums.update(new_sums)
                        left_idx -= 1
                        added = True
                
                if not added:
                    # Try skipping some values
                    skipped = False
                    if right_idx < len(values):
                        # Try next few values on the right
                        for skip in range(1, min(5, len(values) - right_idx)):
                            if time.time() > deadline - 2:
                                break
                            candidate = values[right_idx + skip]
                            new_seq = seq + [candidate]
                            valid = True
                            new_sums = set()
                            s = 0
                            for j in range(len(seq), -1, -1):
                                s += new_seq[j]
                                if s in seen_sums or s in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s)
                            if valid and not any(s in seen_sums for s in new_sums):
                                seq = new_seq
                                seen_sums.update(new_sums)
                                right_idx += skip + 1
                                skipped = True
                                break
                    if not skipped and left_idx >= 0:
                        for skip in range(1, min(5, left_idx + 1)):
                            if time.time() > deadline - 2:
                                break
                            candidate = values[left_idx - skip]
                            new_seq = [candidate] + seq
                            valid = True
                            new_sums = set()
                            s = 0
                            for j in range(len(new_seq)):
                                s += new_seq[j]
                                if s in seen_sums or s in new_sums:
                                    valid = False
                                    break
                                new_sums.add(s)
                            if valid and not any(s in seen_sums for s in new_sums):
                                seq = new_seq
                                seen_sums.update(new_sums)
                                left_idx -= skip + 1
                                skipped = True
                                break
                    if not skipped:
                        break
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
        
        return best_seq

    def greedy_forward(order):
        """Standard greedy: add elements in given order if valid."""
        seq = []
        seen = set()
        for x in order:
            # Check if adding x (in sorted position) is valid
            # Since we need strictly increasing, x must be inserted in sorted order
            # Find insertion point
            pos = 0
            while pos < len(seq) and seq[pos] < x:
                pos += 1
            if pos < len(seq) and seq[pos] == x:
                continue  # duplicate
            new_seq = seq[:pos] + [x] + seq[pos:]
            # Check validity efficiently: only need to check blocks involving x
            # But for simplicity, do full check if seq is small
            if len(seq) < 30:
                if not check_valid(new_seq):
                    continue
            else:
                # Quick check: compute all sums
                valid = True
                n = len(new_seq)
                temp_seen = set()
                for i in range(n):
                    s = 0
                    for j in range(i, n):
                        s += new_seq[j]
                        if s in temp_seen:
                            valid = False
                            break
                        temp_seen.add(s)
                    if not valid:
                        break
                if not valid:
                    continue
            seq = new_seq
            # Update seen sums (recompute for simplicity)
            seen = set()
            n = len(seq)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += seq[j]
                    seen.add(s)
        return seq

    def local_search_extend(seq, time_budget):
        """
        Try to extend a valid sequence by finding values that can be inserted
        at any position. Use a smarter approach: for each candidate value not in seq,
        try inserting it at the correct sorted position and check validity.
        """
        if not seq:
            return seq
        
        current = seq[:]
        current_len = len(current)
        used = set(current)
        
        # Precompute current block sums
        def compute_sums(s):
            sums = set()
            n = len(s)
            for i in range(n):
                ss = 0
                for j in range(i, n):
                    ss += s[j]
                    sums.add(ss)
            return sums
        
        seen_sums = compute_sums(current)
        
        end_time = time.time() + time_budget
        progress = True
        
        while progress and time.time() < end_time:
            progress = False
            # Try each unused value
            candidates = [x for x in range(1, N + 1) if x not in used]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > end_time:
                    break
                # Find insertion position
                pos = 0
                while pos < len(current) and current[pos] < x:
                    pos += 1
                if pos < len(current) and current[pos] == x:
                    continue
                
                new_seq = current[:pos] + [x] + current[pos:]
                
                # Check if valid: only need to verify new block sums don't conflict
                # New block sums are those that include x
                # These are blocks starting at i and ending at j where i <= pos <= j in new_seq
                # Equivalently, blocks that start in [0, pos] and end in [pos, n-1]
                # and include position pos
                new_sums = set()
                n = len(new_seq)
                for i in range(0, pos + 1):
                    s = 0
                    for j in range(i, n):
                        s += new_seq[j]
                        if j >= pos:
                            # This sum includes x
                            if s in seen_sums or s in new_sums:
                                # Conflict
                                new_sums = None
                                break
                            new_sums.add(s)
                        else:
                            # Sum doesn't include x yet, but might later
                            pass
                    if new_sums is None:
                        break
                
                if new_sums is not None:
                    # Also need to check that sums NOT including x are still fine
                    # They were fine before, and we didn't change them
                    # But wait: inserting x shifts positions. The actual values of
                    # blocks not including x remain the same. So they're still distinct.
                    # The only new sums are those including x.
                    # So if no conflict, it's valid.
                    current = new_seq
                    used.add(x)
                    seen_sums.update(new_sums)
                    current_len += 1
                    progress = True
                    break
        
        return current

    # Phase 1: Bidirectional construction with various value subsets
    best_seq = []
    best_len = 0
    
    # Full range bidirectional
    t0 = time.time()
    seq = bidirectional_construct(list(range(1, N + 1)))
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]
    
    # Try with various subsets (skip some values)
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        # Create a subset by skipping random values
        skip_count = random.randint(0, 50)
        skips = set(random.sample(range(1, N + 1), skip_count))
        values = [x for x in range(1, N + 1) if x not in skips]
        seq = bidirectional_construct(values)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # Phase 2: Local search extension from best
    if time.time() < deadline - 10:
        budget = min(20, deadline - 10 - time.time())
        if budget > 1:
            extended = local_search_extend(best_seq, budget)
            if len(extended) > best_len:
                best_len = len(extended)
                best_seq = extended[:]
    
    # Phase 3: Multiple local search runs from different initializations
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        num_runs = max(1, int(remaining / 8))
        for run in range(num_runs):
            if time.time() > deadline - 5:
                break
            # Create a random initial sequence via greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            init_seq = greedy_forward(order)
            
            # Or start from best and perturb
            if random.random() < 0.5 and best_seq:
                perturb = list(best_seq)
                for _ in range(random.randint(1, 3)):
                    if perturb:
                        idx = random.randint(0, len(perturb) - 1)
                        removed = perturb.pop(idx)
                        available = [x for x in range(1, N + 1) if x not in set(perturb)]
                        if available:
                            perturb.append(random.choice(available))
                init_seq = sorted(set(perturb))
                if not check_valid(init_seq):
                    init_seq = greedy_forward(order)
            
            run_budget = min(6, deadline - 5 - time.time())
            if run_budget < 0.5:
                break
            result = local_search_extend(init_seq, run_budget)
            if len(result) > best_len:
                best_len = len(result)
                best_seq = result[:]
    
    # Final verification
    if best_seq and not check_valid(best_seq):
        # Fallback: find a small valid sequence
        best_seq = []
        for x in range(1, N + 1):
            new_seq = best_seq + [x]
            if check_valid(new_seq):
                best_seq = new_seq
    
    return best_seq
