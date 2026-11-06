import random
import time

def construct():
    mandatory = {1, 2, 5, 11}
    max_val = 284
    
    def is_valid(S):
        sums = set()
        L = sorted(S)
        for i in range(len(L)):
            for j in range(i, len(L)):
                s = L[i] + L[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_sums(S):
        sums = set()
        L = sorted(S)
        for i in range(len(L)):
            for j in range(i, len(L)):
                sums.add(L[i] + L[j])
        return sums

    def try_add(S, sums, x):
        for y in S:
            if x + y in sums:
                return False
        if 2*x in sums:
            return False
        return True

    def greedy_initial():
        S = set(mandatory)
        sums = get_sums(S)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in S:
                continue
            if try_add(S, sums, x):
                S.add(x)
                for y in S:
                    sums.add(x + y)
                sums.add(2*x)
        
        # Fill remaining with sorted order if possible? 
        # Greedy random is usually okay for init.
        return S, sums

    def local_search_2swap(S, sums, candidates, time_limit=5.0):
        start_time = time.time()
        best_S = set(S)
        best_size = len(S)
        best_sums = set(sums)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1-Opt: Try to add
            cand_list = list(candidates)
            random.shuffle(cand_list)
            for x in cand_list:
                if x in S: continue
                if try_add(S, sums, x):
                    S.add(x)
                    for y in S:
                        sums.add(x + y)
                    sums.add(2*x)
                    candidates.remove(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        best_sums = set(sums)
                    break
            
            if improved:
                continue

            # 1-Opt: Try to swap (remove one, add one)
            removable = [x for x in S if x not in mandatory]
            if removable:
                random.shuffle(removable)
                for x_out in removable:
                    # Temporarily remove
                    temp_S = S - {x_out}
                    # Recalculate sums for temp_S? 
                    # It's faster to just check validity of (S - {x_out}) U {x_in}
                    # Let's optimize: 
                    # Sums in S are unique.
                    # When we remove x_out, we lose sums {x_out + y for y in S}.
                    # We want to add x_in.
                    # New sums = (Old Sums - Removed Sums) + New Sums
                    # This is complex to track incrementally without a full rebuild or careful diff.
                    # Given N=20, rebuilding sums is O(N^2) = 400 ops. Very fast.
                    
                    valid_swap_found = False
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    
                    for x_in in cand_list:
                        if x_in in S: continue
                        new_S = temp_S | {x_in}
                        # Check if new_S is valid
                        if is_valid(new_S):
                            # Accept
                            S = new_S
                            sums = get_sums(S)
                            candidates.remove(x_in)
                            candidates.add(x_out)
                            improved = True
                            valid_swap_found = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                                best_sums = set(sums)
                            break
                    if valid_swap_found:
                        break
            
            if improved:
                continue

            # 2-Opt: Try to swap 2 (remove two, add two)
            removable = [x for x in S if x not in mandatory]
            if len(removable) >= 2:
                # Pick 2 to remove
                # To limit search, pick a small subset of pairs
                # Or just try a few random pairs
                attempts = 0
                while attempts < 50: # Limit 2-opt attempts per iteration
                    attempts += 1
                    idx1, idx2 = random.sample(range(len(removable)), 2)
                    x_out1 = removable[idx1]
                    x_out2 = removable[idx2]
                    
                    if x_out1 == x_out2: continue
                    if x_out1 in mandatory or x_out2 in mandatory: continue
                    
                    temp_S = S - {x_out1, x_out2}
                    
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    
                    # Try to add 2 from candidates
                    # This is O(C^2) which is large. 
                    # Instead, iterate candidates for first, then second.
                    found = False
                    for x_in1 in cand_list[:50]: # Limit to 50 candidates for speed
                        if x_in1 in S: continue
                        if x_in1 == x_out1 or x_in1 == x_out2: continue
                        
                        # Check if x_in1 can be added to temp_S
                        # We need to check conflicts with temp_S
                        # And later with x_in2
                        # Let's just construct new_S and check validity
                        # But checking validity of 20 elements is cheap.
                        
                        # To avoid O(N^2) inside O(C^2), we can do:
                        # 1. Check if x_in1 is compatible with temp_S
                        # 2. Check if x_in2 is compatible with temp_S U {x_in1}
                        
                        # Let's just use is_valid for simplicity first, it's N=20.
                        # 50 * 50 * 400 ops = 1M ops. Might be too slow if done often.
                        # Let's reduce candidate list for 2-opt.
                        
                        for x_in2 in cand_list[:50]:
                            if x_in2 in S: continue
                            if x_in2 == x_out1 or x_in2 == x_out2: continue
                            if x_in2 == x_in1: continue
                            
                            new_S = temp_S | {x_in1, x_in2}
                            if is_valid(new_S):
                                # Accept
                                S = new_S
                                sums = get_sums(S)
                                candidates.remove(x_in1)
                                candidates.remove(x_in2)
                                candidates.add(x_out1)
                                candidates.add(x_out2)
                                improved = True
                                found = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                    best_sums = set(sums)
                                break
                        if found:
                            break
                    if found:
                        break
            else:
                # Cannot do 2-opt
                pass

            if not improved:
                # Stuck
                break

        return best_S, best_sums, best_size

    best_overall_S = set(mandatory)
    best_overall_size = len(mandatory)
    
    start_global = time.time()
    
    while time.time() - start_global < 110:
        # Restart
        S, sums = greedy_initial()
        
        # Local Search
        S, sums, size = local_search_2swap(S, sums, set(range(1, max_val+1)) - S, time_limit=5.0)
        
        if size > best_overall_size:
            best_overall_size = size
            best_overall_S = set(S)
            
        # Perturbation: Randomly swap a few elements to escape
        if best_overall_size < 20:
            # Try to perturb the current best or the local best?
            # Usually perturb the local best.
            # Let's just rely on the restart loop.
            pass
            
    return sorted(best_overall_S)
