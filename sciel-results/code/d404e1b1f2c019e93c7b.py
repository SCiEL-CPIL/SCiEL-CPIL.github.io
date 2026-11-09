import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    ap_ids_for_element = [[] for _ in range(N + 1)]
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_id = len(aps)
            aps.append((a, b, c, e))
            for x in (a, b, c, e):
                ap_ids_for_element[x].append(ap_id)
                
    num_aps = len(aps)
    
    def can_add(x, ap_counts):
        for ap_id in ap_ids_for_element[x]:
            if ap_counts[ap_id] >= 3:
                return False
        return True

    def greedy_fill(current_set, ap_counts):
        """Add any valid elements to current_set."""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x not in current_set and can_add(x, ap_counts):
                    current_set.add(x)
                    for ap_id in ap_ids_for_element[x]:
                        ap_counts[ap_id] += 1
                    improved = True
        return current_set, ap_counts

    def local_search(initial_set):
        current_set = set(initial_set)
        ap_counts = [0] * num_aps
        for x in current_set:
            for ap_id in ap_ids_for_element[x]:
                ap_counts[ap_id] += 1
        
        # Initial fill
        current_set, ap_counts = greedy_fill(current_set, ap_counts)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try all single-element removals and see if we can gain more by adding
            best_gain = 0
            best_state = None
            
            for x_remove in list(current_set):
                # Remove x_remove
                for ap_id in ap_ids_for_element[x_remove]:
                    ap_counts[ap_id] -= 1
                
                # Try to fill greedily
                temp_set = set(current_set)
                temp_set.remove(x_remove)
                temp_counts = ap_counts[:]
                
                temp_set, temp_counts = greedy_fill(temp_set, temp_counts)
                
                gain = len(temp_set) - len(current_set)
                
                if gain > best_gain:
                    best_gain = gain
                    best_state = (temp_set, temp_counts)
                
                # Restore counts
                for ap_id in ap_ids_for_element[x_remove]:
                    ap_counts[ap_id] += 1

            if best_gain > 0:
                current_set, ap_counts = best_state
                improved = True
                
        return current_set

    def random_greedy():
        current_set = set()
        ap_counts = [0] * num_aps
        # Random order
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        for x in elems:
            if can_add(x, ap_counts):
                current_set.add(x)
                for ap_id in ap_ids_for_element[x]:
                    ap_counts[ap_id] += 1
        return current_set

    # Main Loop
    best_set = set()
    best_size = 0
    
    # Initial few random greedy starts
    for _ in range(20):
        s = random_greedy()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Local Search on the best one
    if best_size > 0:
        best_set = local_search(best_set)
        best_size = len(best_set)
        
    # Iterative Improvement with Perturbation
    while time.time() < deadline:
        # Perturb the best set
        if best_size > 2:
            k = random.randint(1, 3)
            to_remove = random.sample(list(best_set), k)
            perturbed = set(best_set)
            for x in to_remove:
                perturbed.remove(x)
        else:
            perturbed = random_greedy()
            
        candidate = local_search(perturbed)
        
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
        else:
            # Occasionally try a fresh start
            if random.random() < 0.05:
                fresh = random_greedy()
                fresh = local_search(fresh)
                if len(fresh) > best_size:
                    best_size = len(fresh)
                    best_set = fresh

    return sorted(best_set)
