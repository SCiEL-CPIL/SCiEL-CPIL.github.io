import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    N = 1000
    best_set = None
    best_size = 0
    best_counts = None
    best_pairs = None

    def init_greedy(seed_val):
        random.seed(seed_val)
        chosen_set = set()
        counts = {}
        pairs = {}
        
        def get_pairs_for_sum(s):
            return pairs.get(s, [])

        def can_add(x):
            # Check 2*x
            if counts.get(2 * x, 0) >= 2:
                return False
            # Check x + y for y in chosen_set
            for y in chosen_set:
                s = x + y
                if counts.get(s, 0) >= 2:
                    return False
            return True

        def add_element(x):
            chosen_set.add(x)
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
            pairs.setdefault(s, []).append((x, x))
            
            for y in chosen_set:
                if y == x:
                    continue
                s = x + y
                counts[s] = counts.get(s, 0) + 1
                pairs.setdefault(s, []).append((x, y))

        def remove_element(x):
            chosen_set.remove(x)
            s = 2 * x
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
                if s in pairs:
                    del pairs[s]
            else:
                if s in pairs:
                    # Remove pair (x,x)
                    idx = pairs[s].index((x, x))
                    pairs[s].pop(idx)
            
            for y in chosen_set:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    if s in counts:
                        del counts[s]
                    if s in pairs:
                        del pairs[s]
                else:
                    if s in pairs:
                        # Remove pair (x,y)
                        # Note: order might be (y,x) if y < x? No, we added (x,y) where x was the new one.
                        # But when removing, we just need to find a pair containing x and y.
                        found = False
                        for i, (a, b) in enumerate(pairs[s]):
                            if (a == x and b == y) or (a == y and b == x):
                                pairs[s].pop(i)
                                found = True
                                break
                        if not found:
                            # This should not happen if logic is correct
                            pass

        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x):
                add_element(x)
                
        return chosen_set, counts, pairs

    def local_search(chosen_set, counts, pairs):
        chosen_set = set(chosen_set)
        counts = dict(counts)
        pairs = {s: list(ps) for s, ps in pairs.items()}
        unchosen = set(range(1, N + 1)) - chosen_set
        
        improved = True
        passes = 0
        max_passes = 50 # Increase passes for deeper search
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            unchosen_list = list(unchosen)
            random.shuffle(unchosen_list)
            
            for x in unchosen_list:
                if time.time() > deadline:
                    break
                
                # Check if x can be added directly
                if counts.get(2 * x, 0) < 2:
                    blocked_sums = []
                    can_add_directly = True
                    for y in chosen_set:
                        s = x + y
                        if counts.get(s, 0) >= 2:
                            blocked_sums.append(s)
                            can_add_directly = False
                    if can_add_directly:
                        # Add x
                        chosen_set.add(x)
                        unchosen.remove(x)
                        
                        s = 2 * x
                        counts[s] = counts.get(s, 0) + 1
                        pairs.setdefault(s, []).append((x, x))
                        
                        for y in chosen_set:
                            if y == x:
                                continue
                            s = x + y
                            counts[s] = counts.get(s, 0) + 1
                            pairs.setdefault(s, []).append((x, y))
                        improved = True
                        continue
                
                # If not directly addable, try to find a swap
                # Identify all sums s where count(s) >= 2 and x contributes to s
                # x contributes to 2x (if x in set, but x not in set now, so 2x count doesn't include x yet)
                # Wait, if x is not in set, does x contribute to 2x? No.
                # But adding x WILL contribute to 2x.
                # So if count(2x) == 2, adding x makes it 3. We must reduce count(2x) by 1.
                # The pairs contributing to 2x are in pairs[2x].
                
                # x contributes to x+y for y in chosen_set.
                # If count(x+y) == 2, adding x makes it 3. We must reduce count(x+y) by 1.
                
                required_reductions = []
                
                if counts.get(2 * x, 0) >= 2:
                    required_reductions.append(2 * x)
                    
                for y in chosen_set:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        required_reductions.append(s)
                
                if not required_reductions:
                    # This case shouldn't happen if can_add_directly failed, but just in case
                    continue

                # We need to remove a set of elements R from chosen_set such that
                # for every s in required_reductions, at least one element from R is in a pair summing to s.
                
                # Try 1-swap: find y in chosen_set that hits all required_reductions
                found_swap = False
                # Sort chosen_set by something? Maybe just iterate.
                # To speed up, we can filter candidates y that are in at least one of the required pairs.
                candidate_ys = set()
                for s in required_reductions:
                    if s in pairs:
                        for (a, b) in pairs[s]:
                            candidate_ys.add(a)
                            candidate_ys.add(b)
                
                # Try each candidate y
                for y in candidate_ys:
                    if y not in chosen_set:
                        continue
                    
                    # Check if removing y reduces count of all s in required_reductions
                    valid_swap = True
                    for s in required_reductions:
                        # Does y participate in any pair summing to s?
                        participates = False
                        if s in pairs:
                            for (a, b) in pairs[s]:
                                if a == y or b == y:
                                    participates = True
                                    break
                        if not participates:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap: remove y, add x
                        # Remove y
                        chosen_set.remove(y)
                        unchosen.add(y)
                        
                        # Update counts and pairs for removal of y
                        s = 2 * y
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                            if s in pairs:
                                del pairs[s]
                        else:
                            if s in pairs:
                                # Remove (y,y)
                                idx = pairs[s].index((y, y))
                                pairs[s].pop(idx)
                        
                        for z in chosen_set:
                            s = y + z
                            counts[s] -= 1
                            if counts[s] == 0:
                                if s in counts:
                                    del counts[s]
                                if s in pairs:
                                    del pairs[s]
                            else:
                                if s in pairs:
                                    # Remove pair (y,z) or (z,y)
                                    found = False
                                    for i, (a, b) in enumerate(pairs[s]):
                                        if (a == y and b == z) or (a == z and b == y):
                                            pairs[s].pop(i)
                                            found = True
                                            break
                                    if not found:
                                        pass
                        
                        # Add x
                        chosen_set.add(x)
                        unchosen.remove(x)
                        
                        s = 2 * x
                        counts[s] = counts.get(s, 0) + 1
                        pairs.setdefault(s, []).append((x, x))
                        
                        for z in chosen_set:
                            if z == x:
                                continue
                            s = x + z
                            counts[s] = counts.get(s, 0) + 1
                            pairs.setdefault(s, []).append((x, z))
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue # Continue to next x

        return chosen_set, counts, pairs

    # Run multiple restarts with different seeds
    for i in range(20):
        if time.time() > deadline:
            break
        
        seed_val = 12345 + i
        chosen_set, counts, pairs = init_greedy(seed_val)
        
        # Local search
        chosen_set, counts, pairs = local_search(chosen_set, counts, pairs)
        
        current_size = len(chosen_set)
        if current_size > best_size:
            best_size = current_size
            best_set = list(chosen_set)

    if best_set is None:
        best_set = []
        
    return best_set

random.seed(42)
