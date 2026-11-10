import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def get_valid_candidates(current_seq, sums_set, max_val=256):
        """
        Returns a list of valid next values x that can be appended to current_seq.
        """
        if not current_seq:
            candidates = []
            for x in range(1, max_val + 1):
                # Sums: {x}
                if x not in sums_set:
                    candidates.append(x)
            return candidates
        
        last_val = current_seq[-1]
        candidates = []
        # We only need to check x > last_val
        # Also, the new sums are x, x+last_val, ...
        # To keep the sequence short and sums distinct, smaller x is generally better
        # but we need to check all.
        
        # Optimization: If the sum of the whole sequence + x > max_possible_sum, 
        # it doesn't matter for distinctness, but x must be <= 256.
        
        # Iterate x from last_val + 1 to 256
        # We can break early if we find many candidates? No, we need to pick one.
        # But we can limit the search range if we want to speed up.
        
        # Let's just generate all valid ones.
        for x in range(last_val + 1, max_val + 1):
            # Check new sums
            total = x
            valid = True
            # The new sums are x, x+seq[-1], x+seq[-1]+seq[-2], ...
            # We need to check if any of these are in sums_set
            # Note: x itself is a new sum.
            if x in sums_set:
                continue # Collision with single element sum? 
                         # Wait, sums_set contains sums of ALL blocks in current_seq.
                         # x is a new block [x]. If x is already a sum of some block in current_seq, collision.
            
            # Check longer blocks ending at x
            # total starts as x. We already checked x.
            # Now add seq[-1]
            # We need to check x + seq[-1], x + seq[-1] + seq[-2], etc.
            
            # Efficient check:
            # Precompute suffix sums of current_seq?
            # Let suffix_sums[i] = sum(current_seq[i:])
            # Then new sums are x + suffix_sums[i] for i=0..len-1.
            # We need to check if any (x + suffix_sums[i]) is in sums_set.
            
            # We can pass suffix_sums to this function to avoid recomputation.
            
            # For now, let's assume we have suffix_sums.
            # But inside this loop, we don't have them.
            # Let's refactor: compute suffix sums once per call.
            
            pass # Placeholder
            
        return candidates

    # Better implementation of candidate generation
    
    def generate_candidates(seq, sums_set):
        if not seq:
            # Any x in 1..256 not in sums_set (which is empty)
            return list(range(1, 257))
        
        last_val = seq[-1]
        # Compute suffix sums of seq
        # suffix[i] = sum(seq[i:])
        n = len(seq)
        suffix = [0] * n
        total = 0
        for i in range(n - 1, -1, -1):
            total += seq[i]
            suffix[i] = total
            
        candidates = []
        # x must be > last_val and <= 256
        for x in range(last_val + 1, 257):
            # New sums: x, x + suffix[0], x + suffix[1], ...
            # Wait, the blocks ending at the new element x are:
            # [x] -> sum x
            # [seq[-1], x] -> sum seq[-1] + x
            # [seq[-2], seq[-1], x] -> sum seq[-2] + seq[-1] + x
            # ...
            # [seq[0], ..., seq[-1], x] -> sum seq + x
            
            # So the new sums are x + suffix[i] for i from n-1 down to 0?
            # Let's check indices.
            # suffix[n-1] = seq[-1]. Block [seq[-1], x] sum is x + suffix[n-1].
            # suffix[n-2] = seq[-2] + seq[-1]. Block [seq[-2], seq[-1], x] sum is x + suffix[n-2].
            # ...
            # suffix[0] = sum(seq). Block [seq, x] sum is x + suffix[0].
            
            # So we need to check if {x + suffix[i] for i in range(n)} intersects with sums_set.
            # Note: The block [x] corresponds to sum x. This is NOT in the list above?
            # suffix[i] is sum of seq[i:].
            # If we consider the new element at index n, the block [x] is just x.
            # The block [seq[n-1], x] is x + seq[n-1].
            # So the set of new sums is {x} U {x + suffix[i] for i in range(n)}.
            
            # Check collision
            collision = False
            if x in sums_set:
                collision = True
            else:
                for i in range(n):
                    s = x + suffix[i]
                    if s in sums_set:
                        collision = True
                        break
            
            if not collision:
                candidates.append(x)
                
        return candidates

    # Fast Greedy to get a baseline
    def greedy_construct():
        seq = []
        sums_set = set()
        for x in range(1, 257):
            cands = generate_candidates(seq, sums_set)
            if cands:
                # Greedy: pick the smallest
                chosen = cands[0]
                seq.append(chosen)
                # Update sums
                n = len(seq) - 1
                # New sums are chosen + suffix of previous seq
                # Actually, just recompute or update efficiently
                # Since we picked the smallest, and we are in a loop, 
                # updating sums_set is O(N^2) if done naively.
                # Let's just recompute sums for the whole sequence if we want to be safe, 
                # but that's O(N^3) total.
                # Better: update sums_set incrementally.
                
                # The new sums are:
                # 1. chosen
                # 2. chosen + seq[-2], chosen + seq[-2]+seq[-3], ...
                
                # Let's just use the full sum set update for simplicity in baseline, 
                # as N is small (<= 256).
                # Actually, generate_candidates already computed suffixes.
                # We can just add the new sums to sums_set.
                
                # Re-calculate new sums for the chosen value
                # chosen
                sums_set.add(chosen)
                total = chosen
                for a in reversed(seq[:-1]):
                    total += a
                    sums_set.add(total)
                    
        return seq

    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Randomized Backtracking Search to extend length
    # We try to find a sequence of length L_target
    # We start with L_target = best_len + 1
    
    # To speed up, we can use a "soft" constraint or just random restarts.
    # We will run multiple attempts for each target length.
    
    def search_for_length(target_len, time_budget):
        """
        Attempts to find a valid sequence of length target_len.
        Uses randomized DFS with restarts.
        Returns the sequence if found, else None.
        """
        start = time.time()
        
        # Heuristic: Start with a partial sequence from a greedy run?
        # Or start from scratch?
        # Starting from scratch allows exploring different structures.
        # But greedy is a good guide.
        # Let's try to extend the current best_seq if possible, or start fresh.
        
        # Strategy:
        # 1. Take best_seq.
        # 2. Try to append values to reach target_len.
        # 3. If stuck, backtrack and try different choices.
        
        # However, best_seq might be a "dead end" for longer lengths.
        # So we should also try starting from scratch with randomization.
        
        # Let's do a hybrid:
        # - 50% of time: Extend best_seq using randomized backtracking.
        # - 50% of time: Start from scratch with randomized backtracking.
        
        # Function to do randomized DFS from a given state
        def dfs_randomized(seq, sums_set, depth, max_depth, suffix_cache=None):
            if time.time() - start > time_budget:
                return None
            
            if len(seq) == max_depth:
                return list(seq)
            
            # Get candidates
            cands = generate_candidates(seq, sums_set)
            
            if not cands:
                return None
            
            # Randomize the order of candidates
            # To bias towards smaller values (which are often better for fitting in range),
            # we can shuffle but then sort by value? No, just shuffle.
            # Or pick a random candidate from the first K smallest?
            # Let's just shuffle all candidates.
            random.shuffle(cands)
            
            # Try each candidate
            for x in cands:
                # Check time
                if time.time() - start > time_budget:
                    return None
                
                # Add x
                seq.append(x)
                
                # Update sums
                new_sums = []
                total = x
                new_sums.append(x)
                for a in reversed(seq[:-1]):
                    total += a
                    new_sums.append(total)
                
                # Check for collisions (already done in generate_candidates, 
                # but we need to update sums_set)
                # Wait, generate_candidates checked against the OLD sums_set.
                # So x is safe.
                
                sums_set.update(new_sums)
                
                result = dfs_randomized(seq, sums_set, depth + 1, max_depth)
                if result:
                    return result
                
                # Backtrack
                seq.pop()
                for s in new_sums:
                    sums_set.discard(s)
                    
            return None
        
        # Attempt 1: Extend best_seq
        if len(best_seq) < target_len:
            seq = list(best_seq)
            sums_set = set()
            # Recompute sums for best_seq
            n = len(seq)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += seq[j]
                    sums_set.add(s)
            
            # We need to fill target_len - len(seq) more elements
            # But we can't just append. We might need to change existing elements?
            # No, DFS appends. So this only works if best_seq is a prefix of a valid longer sequence.
            # It might not be.
            
            # Let's try to extend.
            res = dfs_randomized(seq, sums_set, 0, target_len)
            if res:
                return res
        
        # Attempt 2: Start from scratch
        # Use a smaller time budget per restart
        num_restarts = 10
        time_per_restart = time_budget / num_restarts
        
        for _ in range(num_restarts):
            if time.time() - start > time_budget:
                break
            
            seq = []
            sums_set = set()
            res = dfs_randomized(seq, sums_set, 0, target_len)
            if res:
                return res
                
        return None

    # Main search loop
    current_target = best_len + 1
    attempts_for_current_target = 0
    max_attempts_per_target = 5 # Limit attempts before giving up on this length
    
    while current_target <= 256:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for this target length
        # If we are close to deadline, give less time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Give 1 second per attempt, up to 5 attempts
        time_budget = min(1.0, remaining / (max_attempts_per_target - attempts_for_current_target + 1))
        
        found_seq = search_for_length(current_target, time_budget)
        
        if found_seq:
            best_seq = found_seq
            best_len = current_target
            current_target += 1
            attempts_for_current_target = 0
        else:
            attempts_for_current_target += 1
            if attempts_for_current_target >= max_attempts_per_target:
                # Give up on this length, try next? Or stop?
                # Usually if you can't find L+1, you can't find L+2 easily.
                # But maybe a different structure allows L+2? Unlikely.
                # Let's try one more time with a longer budget if time permits.
                if remaining > 5.0:
                    time_budget = 2.0
                    found_seq = search_for_length(current_target, time_budget)
                    if found_seq:
                        best_seq = found_seq
                        best_len = current_target
                        current_target += 1
                        attempts_for_current_target = 0
                    else:
                        break # Stop searching for longer sequences
                else:
                    break
                    
    return best_seq
