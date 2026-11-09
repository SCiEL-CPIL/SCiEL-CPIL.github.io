import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    best_seq = []
    best_len = 0
    search_count = 0
    max_searches = 50000  # Safety limit to prevent infinite loops or excessive time

    def is_valid(seq, sums, x):
        """Check if x can be appended to seq without creating duplicate consecutive sums."""
        if x in sums:
            return False
            
        new_sums = set()
        new_sums.add(x)
        
        s = 0
        # Check sums ending at x: x + sum(suffix)
        for i in range(len(seq) - 1, -1, -1):
            s += seq[i]
            val = x + s
            if val in sums:
                return False
            new_sums.add(val)
            
        # Internal duplicates in new_sums?
        # Since seq is strictly increasing, suffix sums are strictly increasing.
        # Thus x + suffix_sum are strictly increasing.
        # x is the smallest (empty suffix sum is 0).
        # So no internal duplicates.
        return True

    def get_new_sums(seq, x):
        """Calculate the set of new sums added by appending x to seq."""
        new_sums = set()
        new_sums.add(x)
        s = 0
        for i in range(len(seq) - 1, -1, -1):
            s += seq[i]
            new_sums.add(x + s)
        return new_sums

    def count_extensions(seq, sums, min_val):
        """Count how many valid next elements exist for the current state.
           Used as a heuristic to prefer branches that keep options open."""
        count = 0
        for x in range(min_val, 257):
            if is_valid(seq, sums, x):
                count += 1
                if count > 5: # Cap the count for speed, we just want a relative measure
                    return count
        return count

    def search(seq, sums, min_val):
        nonlocal best_seq, best_len, search_count, deadline
        
        search_count += 1
        if time.time() > deadline or search_count > max_searches:
            return
            
        current_len = len(seq)
        if current_len > best_len:
            best_len = current_len
            best_seq = seq[:]
            
        # Generate valid candidates
        candidates = []
        for x in range(min_val, 257):
            if is_valid(seq, sums, x):
                new_sums = get_new_sums(seq, x)
                candidates.append((x, new_sums))
                
        if not candidates:
            return
            
        # Heuristic: Score candidates by how many extensions they allow.
        # Prefer candidates that leave more room for future steps.
        # This is a greedy heuristic for the branch-and-bound.
        scored_candidates = []
        for x, new_sums in candidates:
            # Estimate future potential
            # A simple proxy: count how many y > x are valid for seq+[x]
            # This is expensive, so we do a quick check or just use x itself.
            # Let's try: prefer smaller x? Or larger x?
            # Usually, smaller x leaves more room in [1, 256].
            # However, sometimes a slightly larger x avoids a "trap".
            
            # Let's use a 1-step lookahead count as a score.
            # To keep it fast, we limit the check range or count.
            next_count = 0
            # Check a few potential next values
            # We can just check if ANY exists, but counting gives better ranking.
            # Let's check up to 10 potential next values for speed.
            found_any = False
            for y in range(x + 1, 257):
                if is_valid(seq + [x], sums | new_sums, y):
                    next_count += 1
                    if next_count >= 5:
                        break
            
            # Score: higher next_count is better.
            # Tie-breaker: smaller x is better (leaves more upper range).
            scored_candidates.append((next_count, -x, x, new_sums))
            
        # Sort by score (descending count, ascending x)
        scored_candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        
        # Explore top branches
        # We explore the best branch first.
        # If time remains, explore the second best.
        
        # Branch 1
        top_count, neg_x, x, new_sums = scored_candidates[0]
        next_seq = seq + [x]
        next_sums = sums | new_sums
        search(next_seq, next_sums, x + 1)
        
        # Branch 2 if time permits and there's a significant alternative
        if time.time() < deadline - 5 and search_count < max_searches * 0.8:
            if len(scored_candidates) > 1:
                # Only explore second branch if it's not drastically worse?
                # Or just explore it to allow backtracking.
                second_count, neg_x2, x2, new_sums2 = scored_candidates[1]
                # Heuristic: only branch if the second option is also "good" (has extensions)
                if second_count > 0:
                    next_seq2 = seq + [x2]
                    next_sums2 = sums | new_sums2
                    search(next_seq2, next_sums2, x2 + 1)

    # Initial call
    search([], set(), 1)
    
    return best_seq
