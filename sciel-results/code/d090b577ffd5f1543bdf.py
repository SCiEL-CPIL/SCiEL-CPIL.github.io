import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute K(2,2,2) configurations
    # Each config is a set of 8 triples (indices)
    # We need to ensure that for any config, we select at most 7 of its 8 triples.
    # Equivalently, if we select 8 triples that form a K(2,2,2), it's invalid.
    # So we are looking for a maximum subset of triples such that no subset of 8 forms a K(2,2,2).
    # This is a hypergraph independent set problem where hyperedges are the K(2,2,2) configs.
    # Wait, the condition is "no three disjoint pairs have all eight transversal triples present".
    # This means for every K(2,2,2) configuration (which consists of 8 specific triples),
    # we cannot have ALL 8 of them in our solution.
    # So we need to find a maximum subset of triples such that for every K(2,2,2) config,
    # at least one triple from that config is missing.
    # This is the "Hitting Set" or "Independent Set in Hypergraph" problem.
    # Actually, it's finding the maximum subset of vertices (triples) such that no hyperedge (config) is fully contained.
    # This is equivalent to: Minimize the number of removed triples such that every K(2,2,2) config is hit.
    # Let R be the set of removed triples. We want |R| to be minimal such that for every config C, C ∩ R != empty.
    # Then the answer is num_triples - min_hitting_set_size.
    
    # Let's build the list of configs.
    configs = []
    triple_to_configs = [[] for _ in range(num_triples)]
    
    # There are C(16,6) * 15 configs?
    # Number of ways to choose 6 points is C(16,6) = 8008.
    # For each set of 6 points, there are 15 ways to partition into 3 pairs.
    # So total configs = 8008 * 15 = 120120.
    # Each config has 8 triples.
    # Total triple-config incidences = 120120 * 8 = 960960.
    # This is manageable.
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append(tuple(sorted((x, y, z))))
            
            trans_indices = set()
            for t in transversals:
                trans_indices.add(triple_to_idx[t])
            
            config_idx = len(configs)
            configs.append(trans_indices)
            
            for idx in trans_indices:
                triple_to_configs[idx].append(config_idx)
                
    num_configs = len(configs)
    
    # We want to maximize the number of selected triples.
    # Let's use a greedy heuristic with local search (simulated annealing or ILS).
    # State: set of selected triples.
    # Objective: maximize |selected|.
    # Constraint: for every config, selected ∩ config != config (i.e., not all 8 are selected).
    # Violation: number of configs where all 8 are selected. We want this to be 0.
    
    # Heuristic:
    # 1. Start with all triples selected.
    # 2. While there are violations (configs fully selected), remove one triple from a violating config.
    #    To maximize the final set, we should remove triples that are in many violating configs (high degree in violation graph).
    #    This is a greedy hitting set approach.
    
    def greedy_hitting_set():
        # selected[i] = True if triple i is in the solution
        selected = [True] * num_triples
        # count[v] = number of selected triples in config v
        config_counts = [8] * num_configs  # Initially all 8 are selected
        
        # Active configs: those with count == 8
        # We need to reduce count of every config to <= 7.
        
        # We maintain a list of "active" configs (count == 8).
        # When we remove a triple, we update counts of its configs.
        # If a config count drops to 7, it's no longer active.
        
        # Priority queue of triples to remove?
        # Score of a triple: number of active configs it belongs to.
        # If we remove a triple, it resolves all active configs it belongs to.
        
        # Let's use a simple greedy:
        # 1. Calculate "weight" of each triple: number of active configs it hits.
        # 2. Pick the triple with highest weight.
        # 3. Remove it (set selected to False).
        # 4. Update counts.
        # 5. Repeat until no active configs.
        
        # To be efficient, we can recompute weights or update them incrementally.
        # Given num_triples=560, num_configs=120k, simple recomputation might be ok if iterations are small.
        # But iterations could be up to 560. 560 * 120k = 67M ops. Might be slow in Python.
        
        # Let's try a different approach: Local Search on the "removed" set.
        # Or, use the "additive" approach: Start with empty set, add triples greedily.
        # But "remove" from full set is often better for hitting set.
        
        # Let's optimize the removal loop.
        # We can maintain `triple_active_config_count` (number of active configs for each triple).
        # And `config_active` (bool).
        
        config_active = [True] * num_configs
        triple_hits = [0] * num_triples
        
        for i in range(num_triples):
            for c in triple_to_configs[i]:
                triple_hits[i] += 1
        
        removed = [False] * num_triples
        removed_count = 0
        
        while True:
            # Find triple with max triple_hits
            # If max is 0, no active configs left.
            max_hits = 0
            best_triple = -1
            for i in range(num_triples):
                if not removed[i] and triple_hits[i] > max_hits:
                    max_hits = triple_hits[i]
                    best_triple = i
            
            if best_triple == -1:
                break
                
            # Remove best_triple
            removed[best_triple] = True
            removed_count += 1
            
            # Update triple_hits for this triple to 0 (it's removed, so it doesn't matter)
            # But we need to update triple_hits for OTHER triples.
            # And update config_active.
            
            for c in triple_to_configs[best_triple]:
                if config_active[c]:
                    # This config was active (count 8). Now count 7.
                    config_active[c] = False
                    # Decrement triple_hits for all other triples in this config
                    for t in configs[c]:
                        if t != best_triple and not removed[t]:
                            triple_hits[t] -= 1
                            
        # Return the set of selected triples
        selected = [not r for r in removed]
        return selected, num_triples - removed_count

    # The greedy hitting set might not be optimal.
    # We can improve it with Local Search (ILS).
    # Move: Swap a removed triple with a selected triple?
    # Or: Re-add a removed triple and remove another to fix violations.
    
    def local_search(selected, size):
        # selected: list of bools
        # We want to try to re-add removed triples.
        # If we re-add triple t, it might create violations in configs where it was the only missing one.
        # Specifically, if we re-add t, for each config c containing t:
        #   if all other 7 triples in c are selected, then c becomes a violation.
        # To fix this, we must remove at least one of those 7 triples.
        # So a move is: Re-add t, remove s (where s is one of the neighbors in the violation).
        # This is a 1-for-1 swap. Size stays same.
        # But we want to INCREASE size.
        # Can we do a 1-for-0 move? Re-add t if it doesn't create any violation.
        # i.e., for all c in triple_to_configs[t], count_selected_in_c < 8.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Try to add removed triples (1-for-0)
            for t in range(num_triples):
                if not selected[t]:
                    # Check if adding t creates any violation
                    safe = True
                    for c in triple_to_configs[t]:
                        # Count selected in c
                        # If all 8 are selected (including t which is not yet, so 7 others must be selected)
                        # Wait, t is not selected. So we check if the other 7 are selected.
                        # If yes, adding t makes it 8 -> violation.
                        cnt = 0
                        for u in configs[c]:
                            if u != t and selected[u]:
                                cnt += 1
                        if cnt == 7:
                            safe = False
                            break
                    if safe:
                        selected[t] = True
                        size += 1
                        improved = True
                        # Don't break, continue to find more? 
                        # Adding t might block others, but might also unblock? No, adding only increases counts.
                        # So if t is safe, it stays safe until we add more.
                        # But adding t might make other removed triples UNSAFE.
                        # So we should restart or just continue.
                        # Let's just continue.
            
            if improved:
                continue
                
            # Phase 2: Try swaps (1-for-1) to enable future additions?
            # Or just try to find a better configuration.
            # Actually, if we are stuck in Phase 1, it means every removed triple t has at least one config c
            # where the other 7 are selected.
            # To add t, we must remove one of those 7.
            # This is a swap.
            # Does a swap help? It keeps size same.
            # But it might change the structure such that a different triple becomes addable.
            # This is complex.
            
            # Alternative: Random restarts of the greedy.
            # Or Simulated Annealing on the "removed" set.
            
            # Let's try a simple random perturbation and re-greedy?
            # No, that's expensive.
            
            # Let's try to find a swap that reduces the "potential" for violations.
            # This is getting complicated.
            # Let's stick to the greedy result for now and see the score.
            # The greedy hitting set is usually quite good.
            
            break
            
        return selected, size

    # Run greedy
    selected, size = greedy_hitting_set()
    
    # Try to improve with local search (adding safe triples)
    selected, size = local_search(selected, size)
    
    # If time remains, try multiple random restarts of the greedy?
    # The greedy is deterministic. To randomize, we can shuffle the order of triple selection or tie-breaking.
    # Let's modify greedy to accept a random seed for tie-breaking.
    
    def greedy_hitting_set_random(seed):
        random.seed(seed)
        selected = [True] * num_triples
        config_active = [True] * num_configs
        triple_hits = [0] * num_triples
        
        for i in range(num_triples):
            for c in triple_to_configs[i]:
                triple_hits[i] += 1
        
        removed = [False] * num_triples
        removed_count = 0
        
        while True:
            # Find candidates with max hits
            max_hits = 0
            for i in range(num_triples):
                if not removed[i] and triple_hits[i] > max_hits:
                    max_hits = triple_hits[i]
            
            if max_hits == 0:
                break
                
            candidates = [i for i in range(num_triples) if not removed[i] and triple_hits[i] == max_hits]
            best_triple = random.choice(candidates)
            
            removed[best_triple] = True
            removed_count += 1
            
            for c in triple_to_configs[best_triple]:
                if config_active[c]:
                    config_active[c] = False
                    for t in configs[c]:
                        if t != best_triple and not removed[t]:
                            triple_hits[t] -= 1
                            
        selected = [not r for r in removed]
        return selected, num_triples - removed_count

    best_selected, best_size = selected, size
    
    # Try a few random restarts
    while time.time() < deadline - 10:
        seed = random.randint(0, 1000000)
        sel, sz = greedy_hitting_set_random(seed)
        sel, sz = local_search(sel, sz)
        
        if sz > best_size:
            best_size = sz
            best_selected = sel
        # If we are close to deadline, break
        if time.time() > deadline - 15:
            break

    result = []
    for i in range(num_triples):
        if best_selected[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
