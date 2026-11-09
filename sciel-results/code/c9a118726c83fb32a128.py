import time
import random
import itertools
import sys

# Precomputation
N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Generate all K(2,2,2) configurations
CONFIG_TRIPLES = []
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]

def precompute():
    global CONFIG_TRIPLES, TRIPLE_CONFIGS
    CONFIG_TRIPLES = []
    TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]
    
    six_subsets = list(itertools.combinations(range(N_POINTS), 6))
    
    for pts in six_subsets:
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            remaining = [x for x in rest if x != p1]
            r0, r1, r2, r3 = remaining
            pairings = [
                [(p0, p1), (r0, r1), (r2, r3)],
                [(p0, p1), (r0, r2), (r1, r3)],
                [(p0, p1), (r0, r3), (r1, r2)]
            ]
            for pairs in pairings:
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                
                transversals = [
                    (a, b, c),
                    (a, b, c_p),
                    (a, b_p, c),
                    (a, b_p, c_p),
                    (a_p, b, c),
                    (a_p, b, c_p),
                    (a_p, b_p, c),
                    (a_p, b_p, c_p)
                ]
                
                idxs = []
                for t in transversals:
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is None:
                        continue
                    idxs.append(idx)
                
                config_id = len(CONFIG_TRIPLES)
                CONFIG_TRIPLES.append(idxs)
                for idx in idxs:
                    TRIPLE_CONFIGS[idx].append(config_id)

precompute()
N_CONFIGS = len(CONFIG_TRIPLES)

# Precompute for each config, the list of its 8 triple indices (already done in CONFIG_TRIPLES)
# For fast validity check, we need to know for a given triple, which configs it's in (TRIPLE_CONFIGS)
# And for a config, how many of its 8 triples are currently in the set (config_counts)

def is_valid_addition(triple_idx, config_counts):
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def update_counts(triple_idx, delta, config_counts):
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        config_counts[cfg_id] += delta

def greedy_construct_with_order(order, time_limit, start_time):
    """
    Greedily add triples in the given order.
    Returns (set_of_triples, config_counts)
    """
    current_set = set()
    config_counts = [0] * N_CONFIGS
    
    for t_idx in order:
        if time.time() - start_time > time_limit:
            break
        if is_valid_addition(t_idx, config_counts):
            current_set.add(t_idx)
            update_counts(t_idx, 1, config_counts)
    
    return current_set, config_counts

def greedy_construct(num_restarts=100, time_limit=None):
    best_set = set()
    best_size = 0
    start_time = time.time()
    
    for r in range(num_restarts):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        current_set, _ = greedy_construct_with_order(order, time_limit, start_time)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return best_set, best_size

def local_search(initial_set, time_limit):
    """
    Refine the set using local search with add, swap, and perturb moves.
    """
    current_set = initial_set.copy()
    config_counts = [0] * N_CONFIGS
    
    for t_idx in current_set:
        update_counts(t_idx, 1, config_counts)
        
    best_set = current_set.copy()
    best_size = len(current_set)
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Try to add any valid triple (greedy fill)
        # We iterate through all triples not in the set, in random order
        candidates = [t for t in range(N_TRIPLES) if t not in current_set]
        random.shuffle(candidates)
        
        added_any = False
        for t_idx in candidates:
            if time.time() - start_time > time_limit:
                break
            if is_valid_addition(t_idx, config_counts):
                current_set.add(t_idx)
                update_counts(t_idx, 1, config_counts)
                added_any = True
        
        if added_any:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            continue # Restart to check for more additions
        
        # 2. No single addition possible. Try exchanges.
        # Strategy: Remove 1, Add 2 (Net +1) or Remove 1, Add 1 (Neutral, for perturbation)
        
        # We try a few random 1-out-2-in moves
        found_improvement = False
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            if not current_set:
                break
            
            t_out = random.choice(list(current_set))
            
            # Remove t_out
            current_set.remove(t_out)
            update_counts(t_out, -1, config_counts)
            
            # Find valid additions
            # We want to add 2 triples if possible.
            # We can collect all valid triples first, then pick 2?
            # Or greedily add 1, then check for 2nd.
            
            valid_adds = []
            # To speed up, we can just check candidates in random order
            # But checking all N_TRIPLES is expensive.
            # Let's just iterate through a random subset or all?
            # N_TRIPLES is 560. Checking 560 * 100 attempts = 56k checks.
            # Each check is O(degree). Degree is small.
            # This is feasible.
            
            candidates = [t for t in range(N_TRIPLES) if t not in current_set]
            random.shuffle(candidates)
            
            added = []
            for t_idx in candidates:
                if len(added) == 2:
                    break
                if is_valid_addition(t_idx, config_counts):
                    current_set.add(t_idx)
                    update_counts(t_idx, 1, config_counts)
                    added.append(t_idx)
            
            if len(added) == 2:
                # Net gain +1
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                found_improvement = True
                break # Accept and restart main loop
            elif len(added) == 1:
                # Net gain 0. Accept to perturb?
                # Let's accept with probability 0.5 to escape local minima
                if random.random() < 0.5:
                    found_improvement = True # Mark as "moved" to restart
                    break
                else:
                    # Revert
                    t1 = added[0]
                    current_set.remove(t1)
                    update_counts(t1, -1, config_counts)
                    current_set.add(t_out)
                    update_counts(t_out, 1, config_counts)
            else:
                # Could not add anything. Restore t_out
                current_set.add(t_out)
                update_counts(t_out, 1, config_counts)
        
        if found_improvement:
            continue
        
        # 3. Stuck. Perturbation: Remove k random triples, re-greedy
        if len(current_set) > 10:
            k = min(5, len(current_set) // 2)
            to_remove = random.sample(list(current_set), k)
            for t in to_remove:
                current_set.remove(t)
                update_counts(t, -1, config_counts)
            
            # Greedy re-add
            candidates = [t for t in range(N_TRIPLES) if t not in current_set]
            random.shuffle(candidates)
            for t_idx in candidates:
                if time.time() - start_time > time_limit:
                    break
                if is_valid_addition(t_idx, config_counts):
                    current_set.add(t_idx)
                    update_counts(t_idx, 1, config_counts)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
        
        # If still stuck and time is running out, break
        if (time.time() - start_time) > time_limit - 1.0:
            break

    return best_set, best_size

def construct():
    start_time = time.time()
    total_time_limit = 115.0 # Leave 5s margin for 120s limit
    
    # Phase 1: Greedy
    greedy_time = min(15.0, total_time_limit * 0.15)
    best_set, best_size = greedy_construct(num_restarts=200, time_limit=greedy_time)
    
    current_time = time.time() - start_time
    if current_time > total_time_limit:
        return [list(ALL_TRIPLES[i]) for i in best_set]
    
    # Phase 2: Local Search
    remaining_time = total_time_limit - current_time
    best_set, best_size = local_search(best_set, remaining_time)
    
    # Convert to list of lists of points
    result = [list(ALL_TRIPLES[i]) for i in best_set]
    
    return result
