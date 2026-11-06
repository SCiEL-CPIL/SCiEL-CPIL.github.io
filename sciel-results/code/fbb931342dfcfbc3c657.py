import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 1000
    max_sum = 2 * N
    sum_counts = [0] * (max_sum + 1)
    
    def can_add(x, chosen_list):
        if sum_counts[2*x] >= 2:
            return False
        for y in chosen_list:
            if sum_counts[x + y] >= 2:
                return False
        return True

    def add(x, chosen_list, chosen_set):
        sum_counts[2*x] += 1
        for y in chosen_list:
            sum_counts[x + y] += 1
        chosen_list.append(x)
        chosen_set.add(x)

    def remove(x, chosen_list, chosen_set):
        sum_counts[2*x] -= 1
        for y in chosen_list:
            if y != x:
                sum_counts[x + y] -= 1
        for i in range(len(chosen_list)):
            if chosen_list[i] == x:
                chosen_list[i] = chosen_list[-1]
                chosen_list.pop()
                break
        chosen_set.discard(x)

    def run_greedy(seed, order=None):
        rng = random.Random(seed)
        for i in range(max_sum + 1):
            sum_counts[i] = 0
        chosen_list = []
        chosen_set = set()
        
        if order is None:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        else:
            candidates = order[:]
            rng.shuffle(candidates)
            
        for x in candidates:
            if time.time() > deadline - 10:
                break
            if can_add(x, chosen_list):
                add(x, chosen_list, chosen_set)
                
        return list(chosen_set)

    # 1. Find a good starting set
    best_start = []
    best_start_size = 0
    
    # Try multiple random seeds
    for s in range(50):
        candidate_set = run_greedy(s)
        if len(candidate_set) > best_start_size:
            best_start_size = len(candidate_set)
            best_start = candidate_set
            
    # Try ascending and descending
    asc_set = run_greedy(0, list(range(1, N + 1)))
    if len(asc_set) > best_start_size:
        best_start_size = len(asc_set)
        best_start = asc_set
        
    desc_set = run_greedy(1, list(range(N, 0, -1)))
    if len(desc_set) > best_start_size:
        best_start_size = len(desc_set)
        best_start = desc_set

    # Initialize state
    for i in range(max_sum + 1):
        sum_counts[i] = 0
    current_list = []
    current_set = set()
    for x in best_start:
        add(x, current_list, current_set)
    
    best_size = len(current_list)
    best_result = list(current_set)
    
    rng = random.Random(42)
    
    # Local search phase with Limited Refill
    # We kick out elements and try to refill with a limited number of candidates
    
    while time.time() < deadline:
        saved_counts = sum_counts[:]
        saved_list = current_list[:]
        saved_set = set(current_set)
        
        kick_size = rng.randint(1, 3)
        
        if len(current_list) > kick_size:
            indices = rng.sample(range(len(current_list)), kick_size)
            indices.sort(reverse=True)
            for idx in indices:
                u = current_list[idx]
                remove(u, current_list, current_set)
            
            # Limited Refill
            # Generate a random subset of candidates to try
            candidates = list(set(range(1, N + 1)) - current_set)
            rng.shuffle(candidates)
            
            # Try only the first 100 candidates to speed up iteration
            # If we find a large improvement, we might need more, 
            # but usually small kicks allow small refills
            limit = 100 
            if len(candidates) < limit:
                limit = len(candidates)
                
            for x in candidates[:limit]:
                if time.time() > deadline - 1:
                    break
                if can_add(x, current_list):
                    add(x, current_list, current_set)
            
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_result = list(current_set)
            else:
                # Revert
                sum_counts[:] = saved_counts
                current_list = saved_list
                current_set = saved_set
        else:
            # If set is small, just try adding
            candidates = list(set(range(1, N + 1)) - current_set)
            rng.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline - 1:
                    break
                if can_add(x, current_list):
                    add(x, current_list, current_set)
            if len(current_list) > best_size:
                best_size = len(current_list)
                best_result = list(current_set)
            else:
                sum_counts[:] = saved_counts
                current_list = saved_list
                current_set = saved_set

    # Final validation
    if not best_result:
        best_result = run_greedy(0)
        
    # Verify validity
    test_counts = [0] * (max_sum + 1)
    valid = True
    n = len(best_result)
    for i in range(n):
        for j in range(i, n):
            s = best_result[i] + best_result[j]
            test_counts[s] += 1
            if test_counts[s] > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        best_result = run_greedy(0, list(range(1, N + 1)))
        
    return best_result
# EVOLVE-BLOCK-END
