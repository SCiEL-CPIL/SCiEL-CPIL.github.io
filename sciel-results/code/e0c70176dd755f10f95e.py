"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points without the complete 3-partite K(2,2,2).
    Uses a Max-SAT style local search with efficient candidate management.
    """
    n = 16
    num_triples = 560  # C(16,3)
    
    # Precompute all possible triples
    all_triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple = (i, j, k)
                triple_index[triple] = len(all_triples)
                all_triples.append(triple)
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    
    def get_pairings(points):
        points = list(points)
        pairings = []
        def rec(remaining, current):
            if not remaining:
                pairings.append(current[:])
                return
            first = remaining[0]
            rest = remaining[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [(first, second)])
        rec(points, [])
        return pairings
    
    seen_configs = set()
    for six_points in combinations(range(n), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_index[triple]
                        triples_in_config.append(idx)
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(k222_configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
            
    random.seed(42)
    start_time = time.time()
    time_limit = 115
    
    best_solution = None
    best_count = -1
    
    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_configs
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
        return sol, counts, sum(sol)

    def greedy_fill(sol, counts, inactive_triples):
        """
        Greedily adds any triple from inactive_triples that is safe.
        Returns the set of triples that could NOT be added (still inactive).
        """
        still_inactive = []
        added = 0
        for idx in inactive_triples:
            if sol[idx]:
                continue
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
                added += 1
            else:
                still_inactive.append(idx)
        return still_inactive, added

    # Precompute list of active triples for quick random access? 
    # We will maintain a list of active indices to sample from.
    
    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        sol, counts, current_count = greedy_init()
        
        # List of inactive triples
        inactive = [i for i in range(num_triples) if not sol[i]]
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            
        stagnation_counter = 0
        
        while time.time() - start_time < time_limit:
            # Try to fill
            inactive, added = greedy_fill(sol, counts, inactive)
            current_count += added
            
            if added > 0:
                stagnation_counter = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                continue
            
            # Stuck: no more triples can be added.
            # We need to remove some triples to unlock others.
            # Identify tight configs (count == 7)
            tight_triples = set()
            for c_idx, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in k222_configs[c_idx]:
                        if sol[t_idx]:
                            tight_triples.add(t_idx)
            
            if not tight_triples:
                # No tight configs, but stuck? This implies we added everything possible?
                # If we added everything possible and there are no tight configs, 
                # it means all remaining inactive triples are blocked by something?
                # Actually, if a triple is inactive, it must be blocked by at least one config with count 7.
                # So if no config has count 7, all inactive triples should be addable.
                # This case should not happen if greedy_fill is correct.
                break

            # Perform a kick: remove k random triples from tight_triples
            # k is adaptive: start small, increase if stuck
            k = 2 + (stagnation_counter // 5)
            k = min(k, 10) # Cap the kick size
            
            tight_list = list(tight_triples)
            if len(tight_list) < k:
                k = len(tight_list)
            
            to_remove = random.sample(tight_list, k)
            
            # Save state to rollback if improvement is not found
            # Instead of full rollback, we just try to fill and compare.
            # But we need to know which triples were removed to add them back to 'inactive'
            
            # Remove them
            for idx in to_remove:
                sol[idx] = False
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] -= 1
                current_count -= 1
            
            # Add removed triples to inactive list
            inactive.extend(to_remove)
            
            # Try to fill again
            inactive, added = greedy_fill(sol, counts, inactive)
            current_count += added
            
            if current_count > best_count:
                best_count = current_count
                best_solution = sol[:]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                # If we didn't improve, we might want to revert to the previous state?
                # Or just continue from this new state? 
                # Continuing from a worse state is risky. 
                # Let's revert if no improvement after a few tries?
                # Simpler: Just keep going. If we get stuck again, we kick again.
                # However, drifting to worse solutions can be bad.
                # Let's add a simple check: if we are significantly worse than local best, restart?
                # For now, let's just continue. The random nature will help.
                
                if stagnation_counter > 20:
                    break # Restart this search
    
    if best_solution is None:
        best_solution, _, _ = greedy_init()
        best_count = sum(best_solution)

    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    return result
# EVOLVE-BLOCK-END
