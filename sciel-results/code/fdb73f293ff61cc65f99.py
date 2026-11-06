import time
import random
from itertools import combinations

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations and triple-to-K mapping.
    Returns:
    - all_triples: list of tuples
    - triple_to_k222s: list of lists (for each triple index, list of K indices it belongs to)
    - k222_list: list of frozensets of triple indices
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    triple_to_k222s = [[] for _ in range(len(all_triples))]
    k222_list = []
    
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # There are 15 perfect matchings for 6 points
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # matching is list of 3 tuples: (a,a'), (b,b'), (c,c')
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = set()
            for t in k_triples:
                k_indices.add(triple_to_idx[t])
            
            k222_idx = len(k222_list)
            k222_list.append(frozenset(k_indices))
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
    
    return all_triples, triple_to_k222s, k222_list

def _is_valid_add(idx, present, triple_to_k222s, k222_list):
    """Check if adding idx to present is valid."""
    for k_idx in triple_to_k222s[idx]:
        k_set = k222_list[k_idx]
        # Check if all other members of this K are in present
        all_others_present = True
        for other_idx in k_set:
            if other_idx != idx and other_idx not in present:
                all_others_present = False
                break
        if all_others_present:
            return False
    return True

def _greedy_seed(n_triples, triple_to_k222s, k222_list):
    """Create an initial valid set using greedy random order."""
    order = list(range(n_triples))
    random.shuffle(order)
    present = set()
    for idx in order:
        if _is_valid_add(idx, present, triple_to_k222s, k222_list):
            present.add(idx)
    return present

def _simulated_annealing(initial_set, all_triples, triple_to_k222s, k222_list, time_limit=10.0):
    """
    Simulated Annealing to maximize the size of the independent set.
    Moves: Remove one triple, add one or more triples (if valid).
    Accepts worse solutions with probability exp(-delta_E / T).
    """
    current_set = set(initial_set)
    best_set = set(current_set)
    best_size = len(current_set)
    
    start_time = time.time()
    n_triples = len(all_triples)
    
    # Initial temperature
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = 1.0 # Restart annealing schedule if we cool down too much, or just break? 
            # Let's just break and let the outer loop restart if needed, 
            # but here we run for a fixed time. Let's keep T low.
            # Actually, better to just let it run. If T is too low, it's greedy.
            
        # Generate a neighbor
        # 1. Remove a random element from current_set
        if not current_set:
            break
        
        u_idx = random.choice(list(current_set))
        trial_set = set(current_set)
        trial_set.discard(u_idx)
        
        # 2. Try to add some random elements to trial_set
        # We try to add 1 to 3 elements to potentially increase size
        num_add_attempts = random.randint(1, 3)
        added_count = 0
        for _ in range(num_add_attempts):
            t_idx = random.randint(0, n_triples - 1)
            attempts = 0
            while t_idx in trial_set and attempts < 20:
                t_idx = random.randint(0, n_triples - 1)
                attempts += 1
            
            if t_idx not in trial_set:
                if _is_valid_add(t_idx, trial_set, triple_to_k222s, k222_list):
                    trial_set.add(t_idx)
                    added_count += 1
        
        # Calculate energy change (size difference)
        current_size = len(current_set)
        trial_size = len(trial_set)
        delta_e = trial_size - current_size # We want to maximize size, so positive is better
        
        # Acceptance probability
        if delta_e >= 0:
            current_set = trial_set
            if trial_size > best_size:
                best_size = trial_size
                best_set = set(trial_set)
        else:
            # Accept with probability exp(delta_e / T)
            # delta_e is negative
            prob = random.expovariate(1.0) # Exponential distribution
            # Or simply: if random.random() < exp(delta_e / T)
            # Using log for stability:
            # ln(random) < delta_e / T
            # Since delta_e is negative, let's use the standard formula
            if random.random() < pow(2.71828, delta_e / T):
                current_set = trial_set
        
        T *= cooling_rate
        
        # If T gets too small, reset it occasionally to allow exploration?
        # Or just let it converge.
        
    return best_set

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 16
    
    all_triples, triple_to_k222s, k222_list = _precompute_k222s(n)
    n_triples = len(all_triples)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Generate a good initial seed using greedy
    while time.time() - start_time < deadline - 60.0:
        seed_set = _greedy_seed(n_triples, triple_to_k222s, k222_list)
        if len(seed_set) > best_size:
            best_size = len(seed_set)
            best_set = set(seed_set)
            
        # Run Simulated Annealing on this seed
        # Allocate remaining time minus a safety buffer for the next restart
        remaining = deadline - 10.0 - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        # Split time between SA runs
        # We will run SA for a portion of the remaining time, then restart
        # For simplicity, let's run SA for a fixed chunk, say 5-10 seconds, 
        # but capped by remaining time.
        sa_time = min(10.0, remaining * 0.8)
        
        sa_set = _simulated_annealing(seed_set, all_triples, triple_to_k222s, k222_list, time_limit=sa_time)
        
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = set(sa_set)
            
        # Check time
        if time.time() - start_time > deadline - 20.0:
            break
            
    # Phase 2: Final intensive SA on the best set found
    if time.time() - start_time < deadline - 5.0:
        remaining = deadline - 5.0 - (time.time() - start_time)
        if remaining > 1.0:
            final_set = _simulated_annealing(best_set, all_triples, triple_to_k222s, k222_list, time_limit=remaining)
            if len(final_set) > best_size:
                best_size = len(final_set)
                best_set = set(final_set)

    # Final validation
    # Re-verify best_set to ensure no K(2,2,2) is complete
    valid = True
    for k_idx in range(len(k222_list)):
        if k222_list[k_idx].issubset(best_set):
            valid = False
            break
            
    if not valid:
        # Fallback: remove one triple from each violating K
        # This is a simple repair
        for k_idx in range(len(k222_list)):
            if k222_list[k_idx].issubset(best_set):
                # Remove a random triple from this K
                to_remove = random.choice(list(k222_list[k_idx]))
                best_set.discard(to_remove)
        
        # Re-verify
        for k_idx in range(len(k222_list)):
            if k222_list[k_idx].issubset(best_set):
                # If still invalid, return empty or minimal valid set
                # This should be rare
                best_set = set()
                break

    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
