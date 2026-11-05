import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit, allowing 120s+ buffer but safe

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    # Efficiently generate unique K(2,2,2) structures
    # A K(2,2,2) is defined by 6 points partitioned into 3 pairs.
    # There are C(16,6) * 15 (partitions) configurations, but many are duplicates in terms of the set of 8 triples?
    # No, different pairings of the same 6 points yield different sets of 8 triples.
    # Total unique configurations:
    # Number of ways to choose 6 points: C(16,6) = 8008
    # Number of ways to partition 6 points into 3 pairs: 15
    # Total configs = 8008 * 15 = 120120
    # This is manageable.
    
    for pts in itertools.combinations(points, 6):
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
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
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
        # Check if any config it belongs to already has 7 others
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] >= 7:
                return False
        return True

    def get_safety_margin(t_idx):
        """
        Returns the minimum number of slots remaining in any config containing t_idx.
        If t_idx is already present, returns -1.
        If adding t_idx would violate, returns 0.
        Higher is safer.
        """
        if present[t_idx]:
            return -1
        min_remaining = 8
        for cid in triple_to_configs[t_idx]:
            remaining = 8 - config_counts[cid]
            if remaining < min_remaining:
                min_remaining = remaining
            if min_remaining <= 0:
                return 0
        return min_remaining

    def deterministic_greedy_init():
        """
        Iteratively add the triple with the highest safety margin.
        Ties broken by index for determinism.
        """
        present[:] = False
        config_counts[:] = 0
        
        # We can use a priority queue, but since N is small, simple scan is okay.
        # However, 560 triples * 560 steps = 313k operations. Very fast.
        # To make it faster, we can just scan all missing triples each time.
        
        for _ in range(num_triples):
            best_t = -1
            best_margin = -1
            for t_idx in range(num_triples):
                if not present[t_idx]:
                    margin = get_safety_margin(t_idx)
                    if margin > best_margin:
                        best_margin = margin
                        best_t = t_idx
                    elif margin == best_margin and t_idx < best_t:
                        best_t = t_idx
            if best_t == -1:
                break
            if best_margin <= 0:
                break
            add_triple(best_t)
            
        return np.sum(present)

    def local_search_improve(state, time_limit):
        """
        Try to add missing triples if possible.
        If not, try to swap (remove 1, add 2) to increase count.
        This is a specialized local search.
        """
        current_score = np.sum(state)
        
        # Phase 1: Greedy Add
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Find any valid missing triple
            # To be efficient, we can just scan.
            for t_idx in range(num_triples):
                if not state[t_idx]:
                    if is_valid_addition(t_idx):
                        add_triple(t_idx)
                        current_score += 1
                        changed = True
                        break # Restart scan to pick up new possibilities? 
                        # Actually, just continuing is fine, but restart ensures we don't miss a better one?
                        # No, any valid add is good.
        
        # Phase 2: Swaps (Remove 1, Add 2)
        # This is harder. We need to find a triple r such that removing it allows adding two new triples a, b.
        # Net gain +1.
        
        # We can iterate over present triples r.
        # For each r, tentatively remove it.
        # Check if any two missing triples become valid.
        
        for r_idx in range(num_triples):
            if time.time() > time_limit:
                break
            if not state[r_idx]:
                continue
                
            # Tentatively remove r_idx
            remove_triple(r_idx)
            
            # Find candidates for addition
            candidates = []
            for a_idx in range(num_triples):
                if not state[a_idx]:
                    if is_valid_addition(a_idx):
                        candidates.append(a_idx)
                        
            if len(candidates) >= 2:
                # Check if we can add 2 distinct ones
                # Since we removed r, the counts changed.
                # We need to add a1, then a2.
                # a1 must be valid.
                # After adding a1, a2 must be valid.
                
                # Sort candidates by "safety" or just try first two?
                # Let's try to find a pair (a1, a2) that works.
                # This is O(K^2) where K is number of candidates.
                # K is likely small.
                
                found = False
                for i in range(len(candidates)):
                    a1 = candidates[i]
                    add_triple(a1)
                    for j in range(len(candidates)):
                        if i == j: continue
                        a2 = candidates[j]
                        if is_valid_addition(a2):
                            add_triple(a2)
                            current_score += 1 # -1 (remove r) + 2 (add a1, a2)
                            found = True
                            break
                    if found:
                        break
                    # If not found, remove a1 to try next pair
                    remove_triple(a1)
                
                if found:
                    # We successfully improved
                    # Continue searching for more improvements?
                    # Yes, restart the loop or continue.
                    # For simplicity, we'll just let the outer loop handle restarts.
                    # But we are in a specific r_idx loop.
                    # Let's just break out to re-evaluate from scratch?
                    # Or continue to next r_idx.
                    pass
            else:
                # No candidates, restore r_idx
                add_triple(r_idx)
                continue
            
            if not found:
                # Restore r_idx
                add_triple(r_idx)
                
        return current_score

    best_score = 0
    best_state = np.zeros(num_triples, dtype=bool)
    
    # 1. Deterministic Initialization
    init_score = deterministic_greedy_init()
    if init_score > best_score:
        best_score = init_score
        best_state = present.copy()
        
    # 2. Local Search Improvement on the deterministic structure
    improved_score = local_search_improve(present, deadline - 10)
    if improved_score > best_score:
        best_score = improved_score
        best_state = present.copy()
        
    # 3. Perturbation / Diversification
    # If we have time, try perturbing the best state found so far.
    # We can do this by removing a random set of triples and re-running greedy fill.
    
    if time.time() < deadline - 5:
        random.seed(42)
        current_best = best_score
        current_state = best_state.copy()
        
        # Try a few perturbations
        for _ in range(50): # Limit number of perturbations
            if time.time() > deadline - 2:
                break
                
            # Start from current best
            present[:] = current_state
            config_counts[:] = 0
            for i in range(num_triples):
                if present[i]:
                    add_triple(i)
            
            # Remove 5-10 random triples to create space
            num_removals = random.randint(5, 10)
            present_indices = [i for i in range(num_triples) if present[i]]
            if len(present_indices) < num_removals:
                continue
                
            removals = random.sample(present_indices, num_removals)
            for r in removals:
                remove_triple(r)
                
            # Re-run deterministic greedy fill
            for _ in range(num_triples):
                best_t = -1
                best_margin = -1
                for t_idx in range(num_triples):
                    if not present[t_idx]:
                        margin = get_safety_margin(t_idx)
                        if margin > best_margin:
                            best_margin = margin
                            best_t = t_idx
                        elif margin == best_margin and t_idx < best_t:
                            best_t = t_idx
                if best_t == -1:
                    break
                if best_margin <= 0:
                    break
                add_triple(best_t)
                
            score = np.sum(present)
            
            # Try local search on this new state
            score = local_search_improve(present, deadline - 2)
            
            if score > best_score:
                best_score = score
                best_state = present.copy()
                current_best = score
                current_state = best_state.copy()

    # Convert indices back to triples
    result = [all_triples[i] for i in range(num_triples) if best_state[i]]
    
    return result
