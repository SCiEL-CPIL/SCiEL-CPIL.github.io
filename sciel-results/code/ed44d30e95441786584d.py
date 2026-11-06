import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    # Helper to check if adding x to set S (with counts) is valid
    # Returns (valid, new_counts) or (False, None)
    # We modify counts in place? No, better to check first.
    # For speed, we might just check validity without copying counts if we revert.
    
    def is_valid_addition(x, chosen, counts):
        # Check x+x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x+y
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def get_tightness(x, chosen, counts):
        """Return tightness score (number of sums that would hit count 2) or None if invalid."""
        tight = 0
        if counts.get(2*x, 0) >= 2:
            return None
        if counts.get(2*x, 0) == 1:
            tight += 1
        for y in chosen:
            s = x + y
            c = counts.get(s, 0)
            if c >= 2:
                return None
            if c == 1:
                tight += 1
        return tight

    def run_greedy(noise=0.0, time_limit=None):
        """
        Greedy construction.
        Picks the element that causes the least "tightness" (sums reaching count 2).
        Uses combined validity+tightness check for speed.
        """
        chosen = []
        counts = defaultdict(int)
        candidates = set(range(1, N + 1))
        
        if time_limit is None:
            time_limit = deadline
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                tight = get_tightness(x, chosen, counts)
                if tight is None:
                    continue
                
                score = tight + (random.random() * noise if noise > 0 else 0)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            # Add best_x
            counts[2*best_x] = counts.get(2*best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
            
        return chosen

    def local_search(initial_set, time_limit):
        """
        Tries to grow the set by adding elements.
        If stuck, tries swaps and kicks to escape local optima.
        """
        chosen = list(initial_set)
        counts = defaultdict(int)
        
        # Rebuild counts
        for x in chosen:
            counts[2*x] = counts.get(2*x, 0) + 1
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
                
        candidates = set(range(1, N + 1))
        for x in chosen:
            candidates.discard(x)
            
        if not candidates:
            return chosen
            
        stuck_count = 0
        max_stuck = 50  # After this many stuck iterations, try a kick
        
        while time.time() < time_limit:
            # Try to add
            added_any = False
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for x in cand_list:
                if time.time() > time_limit:
                    break
                tight = get_tightness(x, chosen, counts)
                if tight is not None:
                    # Add it
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    candidates.discard(x)
                    added_any = True
                    stuck_count = 0
                    break
            
            if added_any:
                continue
            
            # If we couldn't add, try swaps
            if len(candidates) == 0:
                break
                
            stuck_count += 1
            
            # Try multiple swap candidates before giving up
            swap_success = False
            for _ in range(10):  # Try up to 10 random swaps
                if time.time() > time_limit:
                    break
                    
                x_to_try = random.choice(list(candidates))
                chosen_list = list(chosen)
                random.shuffle(chosen_list)
                
                for y in chosen_list:
                    # Save state
                    old_counts = dict(counts)
                    
                    # Remove y
                    counts[2*y] -= 1
                    if counts[2*y] == 0: del counts[2*y]
                    for z in chosen:
                        if z != y:
                            s = y + z
                            counts[s] -= 1
                            if counts[s] == 0: del counts[s]
                            
                    # Check x against reduced set
                    reduced_chosen = [z for z in chosen if z != y]
                    if get_tightness(x, reduced_chosen, counts) is not None:
                        # Add x
                        counts[2*x] = counts.get(2*x, 0) + 1
                        for z in reduced_chosen:
                            s = x + z
                            counts[s] = counts.get(s, 0) + 1
                            
                        # Update sets
                        chosen.remove(y)
                        chosen.append(x)
                        candidates.discard(x)
                        candidates.add(y)
                        
                        swap_success = True
                        break
                    else:
                        # Revert counts
                        counts.clear()
                        counts.update(old_counts)
                    
                    if swap_success:
                        break
            
            if not swap_success:
                # Try a kick: remove 1-3 random elements, then greedily refill
                if stuck_count >= max_stuck:
                    kick_size = random.randint(1, min(3, len(chosen)))
                    kicked = random.sample(chosen, kick_size)
                    
                    # Remove kicked elements
                    for y in kicked:
                        counts[2*y] -= 1
                        if counts[2*y] == 0: del counts[2*y]
                        for z in chosen:
                            if z != y:
                                s = y + z
                                counts[s] -= 1
                                if counts[s] == 0: del counts[s]
                        chosen.remove(y)
                        candidates.add(y)
                    
                    # Greedily refill
                    refill_time = min(time.time() + 2.0, time_limit)
                    while candidates and time.time() < refill_time:
                        best_x = None
                        best_tight = float('inf')
                        
                        for x in candidates:
                            tight = get_tightness(x, chosen, counts)
                            if tight is not None and tight < best_tight:
                                best_tight = tight
                                best_x = x
                        
                        if best_x is None:
                            break
                        
                        counts[2*best_x] = counts.get(2*best_x, 0) + 1
                        for y in chosen:
                            s = best_x + y
                            counts[s] = counts.get(s, 0) + 1
                        chosen.append(best_x)
                        candidates.discard(best_x)
                    
                    stuck_count = 0
        
        return chosen

    # Main loop
    while time.time() < deadline:
        # 1. Generate a good baseline with Greedy
        # Use varying noise to explore different greedy paths
        noise = random.uniform(0, 2.0)
        baseline = run_greedy(noise=noise, time_limit=min(deadline, time.time() + 3))
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = baseline
            
        # 2. Refine with Local Search
        # Allocate remaining time for this round
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        # Allocate more time to local search since it's where improvements happen
        ls_time = min(remaining * 0.9, 15.0) # Allow up to 15s per LS round
        start_set = best_set if len(best_set) > len(baseline) else baseline
        refined = local_search(start_set, time.time() + ls_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
            
    return best_set
