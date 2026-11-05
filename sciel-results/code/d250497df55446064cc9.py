import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []  # List of lists of 8 triple indices

    points = range(N)
    for pts in itertools.combinations(points, 6):
        # Generate all partitions of these 6 points into 3 pairs
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            # Normalize pair order and pair internal order for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)

                # Find the 8 transversal triples
                (a, b), (c, d), (e, f) = norm_pairs
                tris = []
                for x in (a, b):
                    for y in (c, d):
                        for z in (e, f):
                            t = tuple(sorted((x, y, z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)

    num_configs = len(configs)

    # For each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)

    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)

    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1

    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] >= 7:
                return False
        return True

    def get_violations():
        """Return list of config indices that have count >= 8."""
        return [cid for cid in range(num_configs) if config_counts[cid] >= 8]

    def repair_violations():
        """Remove triples to fix all K(2,2,2) violations. Returns number removed."""
        removed = 0
        max_iter = 1000
        for _ in range(max_iter):
            violations = get_violations()
            if not violations:
                break
            # Pick the most violated config
            cid = max(violations, key=lambda c: config_counts[c])
            # Find a triple in this config to remove that helps the most
            tris_in_cid = triples_in_config[cid]
            # Score each candidate: how many other violations would be reduced
            best_t = None
            best_score = -1
            for t_idx in tris_in_cid:
                if not present[t_idx]:
                    continue
                # Count how many other violating configs this triple is in
                score = 0
                for other_cid in triple_to_configs[t_idx]:
                    if config_counts[other_cid] >= 8:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_t = t_idx
            if best_t is not None:
                remove_triple(best_t)
                removed += 1
            else:
                break
        return removed

    def greedy_fill(preferred_order=None):
        """Fill with triples in given order (or random if None), respecting constraints."""
        present[:] = False
        config_counts[:] = 0
        if preferred_order is None:
            preferred_order = list(range(num_triples))
            random.shuffle(preferred_order)
        count = 0
        for t_idx in preferred_order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                count += 1
        return count

    def local_search_improve():
        """Try to add missing triples and do swaps to improve."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any missing triple
            for t_idx in range(num_triples):
                if not present[t_idx] and is_valid_addition(t_idx):
                    add_triple(t_idx)
                    improved = True
            if improved:
                continue
            # Try swaps: remove one, add two
            # Sample a few random present triples to try removing
            present_list = [i for i in range(num_triples) if present[i]]
            if not present_list:
                break
            # Try a limited number of swap attempts
            num_swap_attempts = min(200, len(present_list))
            random.shuffle(present_list)
            for victim in present_list[:num_swap_attempts]:
                remove_triple(victim)
                # Try to add back as many as possible
                added = 0
                for t_idx in range(num_triples):
                    if not present[t_idx] and t_idx != victim and is_valid_addition(t_idx):
                        add_triple(t_idx)
                        added += 1
                if added > 1:
                    improved = True
                    break
                else:
                    # Restore
                    if added == 1:
                        # Remove the one we added
                        for t_idx in range(num_triples):
                            if present[t_idx] and t_idx != victim:
                                # Check if this was the one we just added
                                pass
                        # Actually, let's just re-add victim and undo the addition
                        # Find which one was added (it's the one that's present and wasn't before)
                        # Simpler: just restore victim
                        add_triple(victim)
                    else:
                        add_triple(victim)
        return sum(1 for i in range(num_triples) if present[i])

    # --- Main strategy: Algebraic constructions + repair + local search ---

    best_count = 0
    best_set = set()

    # Strategy 1: Random greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        count = greedy_fill()
        if count > best_count:
            best_count = count
            best_set = set(i for i in range(num_triples) if present[i])

    # Strategy 2: Structured constructions
    # Use Z_4 x Z_4 representation: point (x, y) -> 4*x + y
    # Try various algebraic families

    # Construction A: All triples that form lines in AG(2,4) - but AG(2,4) has 16 points
    # Lines in AG(2,4): for each direction (slope), lines through each point
    # In F_4, we have elements {0, 1, w, w^2} where w^2 = w + 1
    # Let's use a simpler algebraic structure

    # Construction B: Based on 2x2x2x2 cube structure
    # Points are (a,b,c,d) in {0,1}^4, mapped to 8a+4b+2c+d
    # Triples: pick 3 coordinates to vary, fix 1
    # For each fixed coordinate value, we get 8 triples per face... 
    # Actually, let's try: for each pair of coordinates to fix, 
    # the remaining 2 coordinates give 4 points, not 3.
    
    # Better: Use the fact that 16 = 4*4, use Z_4 x Z_4
    # Triples where (x1-x2, y1-y2), (x2-x3, y2-y3) are "independent" in some sense

    # Let's try a different structured approach:
    # Start with all 560 triples, then remove minimal set to break all K(2,2,2)
    # This is a hitting set / vertex cover problem on the hypergraph of configs

    # Strategy 3: Start from all triples, remove minimal to be K(2,2,2)-free
    # This is equivalent to finding a maximum K(2,2,2)-free subhypergraph
    # Greedy: repeatedly remove a triple from the most violated config
    
    if time.time() < deadline - 15:
        # Start with all triples
        present[:] = True
        config_counts[:] = 0
        for cid, tris in enumerate(triples_in_config):
            config_counts[cid] = 8  # All 8 are present
        
        # Remove triples to break all configs
        # Greedy: at each step, remove the triple that breaks the most configs
        max_removals = num_triples
        for step in range(max_removals):
            if time.time() > deadline - 15:
                break
            violations = get_violations()
            if not violations:
                break
            # Find triple that appears in the most violating configs
            best_t = None
            best_score = 0
            # Only consider triples in violating configs
            candidates = set()
            for cid in violations:
                for t_idx in triples_in_config[cid]:
                    if present[t_idx]:
                        candidates.add(t_idx)
            
            for t_idx in candidates:
                score = 0
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] >= 8:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_t = t_idx
            
            if best_t is not None and best_score > 0:
                remove_triple(best_t)
            else:
                break
        
        count = sum(1 for i in range(num_triples) if present[i])
        if count > best_count:
            best_count = count
            best_set = set(i for i in range(num_triples) if present[i])
        
        # Now try to add back triples that were removed but can now be added
        # (after removals, some constraints are relaxed)
        for t_idx in range(num_triples):
            if not present[t_idx] and is_valid_addition(t_idx):
                add_triple(t_idx)
        
        count = sum(1 for i in range(num_triples) if present[i])
        if count > best_count:
            best_count = count
            best_set = set(i for i in range(num_triples) if present[i])

    # Strategy 4: Randomized construction with bias toward structured patterns
    # Use a "warm start" from the best found so far, then perturb and repair
    if time.time() < deadline - 10:
        # Take best_set and try to improve via local search
        present[:] = False
        config_counts[:] = 0
        for t_idx in best_set:
            add_triple(t_idx)
        
        count = local_search_improve()
        if count > best_count:
            best_count = count
            best_set = set(i for i in range(num_triples) if present[i])

    # Strategy 5: Multiple random starts with structured initialization
    # Initialize with a random subset, then greedily add
    for seed in range(30):
        if time.time() > deadline - 8:
            break
        random.seed(1000 + seed)
        present[:] = False
        config_counts[:] = 0
        
        # Start with a random subset of ~40% of triples
        initial_size = num_triples // 2
        initial_triples = random.sample(range(num_triples), initial_size)
        for t_idx in initial_triples:
            add_triple(t_idx)
        
        # Repair violations
        repair_violations()
        
        # Greedily add more
        for t_idx in range(num_triples):
            if not present[t_idx] and is_valid_addition(t_idx):
                add_triple(t_idx)
        
        count = sum(1 for i in range(num_triples) if present[i])
        if count > best_count:
            best_count = count
            best_set = set(i for i in range(num_triples) if present[i])

    # Final local search on best
    if time.time() < deadline - 5:
        present[:] = False
        config_counts[:] = 0
        for t_idx in best_set:
            add_triple(t_idx)
        
        count = local_search_improve()
        if count > best_count:
            best_count = count
            best_set = set(i for i in range(num_triples) if present[i])

    # Convert indices back to triples
    result = [all_triples[i] for i in sorted(best_set)]

    return result
