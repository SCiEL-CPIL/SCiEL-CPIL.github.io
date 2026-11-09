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
# A K(2,2,2) is defined by 3 disjoint pairs: {a, a'}, {b, b'}, {c, c'}
# The 8 transversals are all combinations of picking one from each pair.
# We need to map each configuration to the list of 8 triple indices.

# To speed up generation, we can iterate over partitions of 6 points into 3 pairs.
# But simpler: iterate over all sets of 6 points, and all 15 pairings.
# Number of sets of 6: C(16,6) = 8008.
# 15 pairings per set.
# Total 120,120 configurations.

# We will store for each triple index, the list of configuration indices it belongs to.
# And for each configuration index, the 8 triple indices.

# This precomputation might take a second or two.
# 120k * 8 = 1M operations. Very fast.

CONFIG_TRIPLES = [] # List of lists of 8 triple indices
TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)] # List of lists of config indices

def precompute():
    global CONFIG_TRIPLES, TRIPLE_CONFIGS
    CONFIG_TRIPLES = []
    TRIPLE_CONFIGS = [[] for _ in range(N_TRIPLES)]
    
    # Generate all 6-subsets
    six_subsets = list(itertools.combinations(range(N_POINTS), 6))
    
    for pts in six_subsets:
        # Generate all 15 perfect matchings (pairings) of these 6 points
        # Fixed point p0. Pair with p1, p2, p3, p4, p5.
        # Let's just use a helper to generate pairings.
        # There are 15 ways.
        # We can generate them by fixing the partner of the first element.
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            remaining = [x for x in rest if x != p1]
            # Now pair the remaining 4 points.
            # 3 ways to pair 4 points:
            # (r0,r1)(r2,r3), (r0,r2)(r1,r3), (r0,r3)(r1,r2)
            r0, r1, r2, r3 = remaining
            pairings = [
                [(p0, p1), (r0, r1), (r2, r3)],
                [(p0, p1), (r0, r2), (r1, r3)],
                [(p0, p1), (r0, r3), (r1, r2)]
            ]
            for pairs in pairings:
                # pairs is a list of 3 tuples
                # Generate the 8 transversals
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                
                # The 8 combinations
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
                
                # Get indices
                idxs = []
                for t in transversals:
                    # t is a tuple of 3 distinct points, sorted?
                    # itertools.combinations produces sorted tuples.
                    # Our transversals might not be sorted.
                    # TRIPLE_INDEX expects sorted tuples from itertools.combinations.
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is None:
                        # This should not happen as all points are in 0..15
                        continue
                    idxs.append(idx)
                
                config_id = len(CONFIG_TRIPLES)
                CONFIG_TRIPLES.append(idxs)
                for idx in idxs:
                    TRIPLE_CONFIGS[idx].append(config_id)

# Run precomputation at module load time to ensure it's ready
# It's fast enough.
precompute()

def is_valid_addition(triple_idx, in_set, config_counts):
    """
    Check if adding triple_idx to in_set is valid.
    in_set: set of triple indices
    config_counts: list of counts of triples in each config that are in in_set
    """
    if triple_idx in in_set:
        return False
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        # If the config already has 7 other triples, adding this one makes 8
        if config_counts[cfg_id] == 7:
            return False
    return True

def update_counts(triple_idx, delta, config_counts):
    """
    Update config counts for adding (delta=1) or removing (delta=-1) triple_idx.
    """
    for cfg_id in TRIPLE_CONFIGS[triple_idx]:
        config_counts[cfg_id] += delta

def greedy_construct(num_restarts=50, time_limit=None):
    best_set = set()
    best_size = 0
    start_time = time.time()
    
    for r in range(num_restarts):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # Random order of triples
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        current_set = set()
        config_counts = [0] * len(CONFIG_TRIPLES)
        
        for t_idx in order:
            if is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                update_counts(t_idx, 1, config_counts)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return best_set, best_size

def local_search(initial_set, time_limit):
    """
    Refine the set using local search.
    """
    current_set = initial_set.copy()
    config_counts = [0] * len(CONFIG_TRIPLES)
    
    # Initialize counts
    for t_idx in current_set:
        update_counts(t_idx, 1, config_counts)
        
    best_set = current_set.copy()
    best_size = len(current_set)
    
    start_time = time.time()
    
    # Candidate list for additions
    # We can just iterate over all not in set
    
    while (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Try to add any valid triple
        # Randomize the order of candidates to avoid bias
        candidates = [t for t in range(N_TRIPLES) if t not in current_set]
        random.shuffle(candidates)
        
        for t_idx in candidates:
            if is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                update_counts(t_idx, 1, config_counts)
                improved = True
                # Don't break, try to add more if possible?
                # Greedy addition is fine.
        
        if improved:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            continue # Restart the loop to check for more additions
        
        # 2. No single addition possible. Try exchanges.
        # Strategy: Remove 1, Add 1 (Swap) or Remove 1, Add 2 (Net +1)
        # Since no single add is possible, we must remove something to free up constraints.
        
        # Try a few random swaps/exchanges
        for _ in range(50): # Limit attempts per phase to avoid wasting time
            if (time.time() - start_time) > time_limit:
                break
                
            # Pick a random triple to remove
            if not current_set:
                break
            t_out = random.choice(list(current_set))
            
            # Remove it
            current_set.remove(t_out)
            update_counts(t_out, -1, config_counts)
            
            # Try to add 2 random triples
            # We need to find 2 triples that are now valid.
            # Note: Removing t_out might make some previously invalid triples valid.
            
            added = []
            # Get candidates that were invalid before but might be valid now?
            # Simpler: Just try random candidates from the whole pool not in current_set
            
            candidates = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out]
            random.shuffle(candidates)
            
            # Try to add 2
            for t1 in candidates:
                if is_valid_addition(t1, current_set, config_counts):
                    current_set.add(t1)
                    update_counts(t1, 1, config_counts)
                    added.append(t1)
                    break # Only add 1 for now, then check for 2nd?
            
            # If we added 1, try to add another
            if len(added) == 1:
                # Re-shuffle or continue
                candidates2 = [t for t in range(N_TRIPLES) if t not in current_set and t != t_out and t != added[0]]
                random.shuffle(candidates2)
                for t2 in candidates2:
                    if is_valid_addition(t2, current_set, config_counts):
                        current_set.add(t2)
                        update_counts(t2, 1, config_counts)
                        added.append(t2)
                        break
            
            # Evaluate result
            if len(added) > 1:
                # Net gain +1
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                break # Accept and restart main loop
            elif len(added) == 1:
                # Net gain 0. Accept if it helps?
                # Accept with some probability or if it changes the set significantly?
                # Let's accept it to move the state.
                improved = True # Mark as improved to restart addition phase
                break
            else:
                # Could not add anything. Restore t_out
                current_set.add(t_out)
                update_counts(t_out, 1, config_counts)
                
        if not improved:
            # Stuck. Try a larger perturbation: Remove 5, Re-greedy
            if len(current_set) > 5:
                to_remove = random.sample(list(current_set), 5)
                for t in to_remove:
                    current_set.remove(t)
                    update_counts(t, -1, config_counts)
                
                # Greedy re-add
                candidates = [t for t in range(N_TRIPLES) if t not in current_set]
                random.shuffle(candidates)
                for t in candidates:
                    if is_valid_addition(t, current_set, config_counts):
                        current_set.add(t)
                        update_counts(t, 1, config_counts)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()

    return best_set, best_size

def construct():
    # Deadline guard
    start_time = time.time()
    total_time_limit = 110.0 # Leave 10s margin for 120s limit
    
    # Phase 1: Greedy
    greedy_time = min(20.0, total_time_limit * 0.2)
    best_set, best_size = greedy_construct(num_restarts=100, time_limit=greedy_time)
    
    current_time = time.time() - start_time
    if current_time > total_time_limit:
        return [list(ALL_TRIPLES[i]) for i in best_set]
    
    # Phase 2: Local Search
    remaining_time = total_time_limit - current_time
    best_set, best_size = local_search(best_set, remaining_time)
    
    # Convert to list of lists of points
    result = [list(ALL_TRIPLES[i]) for i in best_set]
    
    return result

# Ensure the function is defined even if precomputation is slow (it's not, but just in case)
# The precomputation runs at import.
