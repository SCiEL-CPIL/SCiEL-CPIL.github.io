import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # LCM of 1..22
    # 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
    lcm_val = 232792560
    
    # Precompute weights
    weights = {i: lcm_val // i for i in range(1, 23)}
    
    def get_subset_sums(S):
        """Returns a set of all subset sums of weights for S."""
        if not S:
            return {0}
        sums = {0}
        for x in S:
            w = weights[x]
            # Add w to all existing sums
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        return sums
        
    def can_add(sums, x):
        """Checks if x can be added to a set with current sums without collision.
        Returns (bool, new_sums)"""
        w = weights[x]
        # Collision if any (s + w) is in sums
        for s in sums:
            if (s + w) in sums:
                return False, None
        # If no collision, return new sums
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + w)
        return True, new_sums

    def is_dissociated(S):
        """Checks if all subset sums are distinct."""
        return len(get_subset_sums(S)) == (1 << len(S))
        
    def greedy_construct(order):
        S = []
        sums = {0}
        remaining = set(range(1, 23))
        
        for x in order:
            if x not in remaining:
                continue
            ok, new_sums = can_add(sums, x)
            if ok:
                S.append(x)
                sums = new_sums
                remaining.remove(x)
                
        return S, sums

    def local_search(S, sums, time_limit):
        S = list(S)
        sums = set(sums)
        remaining = set(range(1, 23)) - set(S)
        
        while time.time() < time_limit:
            added = False
            # 1. Try to add
            rem_list = list(remaining)
            random.shuffle(rem_list)
            for x in rem_list:
                ok, new_sums = can_add(sums, x)
                if ok:
                    S.append(x)
                    sums = new_sums
                    remaining.remove(x)
                    added = True
                    break
            
            if added:
                continue
                
            # 2. Try Swap 1-for-1
            if S and remaining:
                for _ in range(10):
                    if time.time() > time_limit: break
                    i = random.randint(0, len(S)-1)
                    j = random.choice(list(remaining))
                    
                    old_x = S[i]
                    new_x = j
                    
                    candidate_S = list(S)
                    candidate_S[i] = new_x
                    
                    candidate_sums = get_subset_sums(candidate_S)
                    if len(candidate_sums) == (1 << len(candidate_S)):
                        S = candidate_S
                        sums = candidate_sums
                        remaining = set(range(1, 23)) - set(S)
                        added = True
                        break
            
            if added:
                continue

            # 3. Try 2-for-2 Swap
            if len(S) >= 2 and len(remaining) >= 2:
                for _ in range(5):
                    if time.time() > time_limit: break
                    
                    i1, i2 = random.sample(range(len(S)), 2)
                    j1, j2 = random.sample(list(remaining), 2)
                    
                    cand = []
                    for idx in range(len(S)):
                        if idx != i1 and idx != i2:
                            cand.append(S[idx])
                    cand.append(j1)
                    cand.append(j2)
                    
                    candidate_sums = get_subset_sums(cand)
                    if len(candidate_sums) == (1 << len(cand)):
                        S = cand
                        sums = candidate_sums
                        remaining = set(range(1, 23)) - set(S)
                        added = True
                        break
                        
        return S, sums

    best_set = []
    best_size = 0
    
    # Run multiple restarts with different initial orders
    orders = [
        list(range(1, 23)),              # Ascending index (Descending weight)
        list(range(22, 0, -1)),          # Descending index (Ascending weight)
    ]
    
    # Add a few random orders
    for _ in range(10):
        rand_order = list(range(1, 23))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        S, sums = greedy_construct(order)
        
        # Local search for a short duration or until improvement
        # Let's give each restart a time slice
        slice_deadline = min(deadline - 2, time.time() + 3)
        S, sums = local_search(S, sums, slice_deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # If we have time left, do a more aggressive local search on the best set found
    if best_set and time.time() < deadline - 5:
        best_sums = get_subset_sums(best_set)
        best_set, best_sums = local_search(best_set, best_sums, deadline - 2)
        
    # Final validation
    while best_set and not is_dissociated(best_set):
        best_set.pop()
        
    return best_set
