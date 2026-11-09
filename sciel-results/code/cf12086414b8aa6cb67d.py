import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16
    
    # 1. Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(n), 3))
    T = len(all_triples) # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # 2. Identify all forbidden configurations (K_{2,2,2} transversals)
    # A forbidden config is defined by 3 disjoint pairs (p1, p2, p3).
    # The 8 transversals are the triples formed by picking one element from each pair.
    # We need to ensure that for any such set of 8 triples, NOT ALL are present.
    
    # Precompute the bitmask for each forbidden configuration.
    # There are C(16, 2) * C(14, 2) * C(12, 2) / 3! ways to choose 3 disjoint pairs?
    # Actually, we can just iterate through all sets of 6 distinct points and partition them into 3 pairs.
    # Or simpler: iterate all combinations of 6 points, and all ways to partition into 3 pairs.
    
    forbidden_masks = []
    
    # To avoid duplicates and ensure efficiency, we iterate over all 6-subsets of points.
    # For each 6-subset, there are 15 ways to partition into 3 pairs.
    # Each partition defines a K_{2,2,2} configuration.
    # The 8 transversals correspond to the 8 vertices of a cube in the 3D product space of the pairs.
    
    for six_pts in itertools.combinations(range(n), 6):
        # Generate all pairings of 6 elements
        # (0,1), (2,3), (4,5) is one.
        # We can generate all perfect matchings of K_6.
        p0, p1, p2, p3, p4, p5 = six_pts
        # 15 matchings:
        matchings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)]
        ]
        
        for match in matchings:
            # match is [(a1,a2), (b1,b2), (c1,c2)]
            a1, a2 = match[0]
            b1, b2 = match[1]
            c1, c2 = match[2]
            
            # The 8 transversals are:
            # (a1,b1,c1), (a1,b1,c2), (a1,b2,c1), (a1,b2,c2)
            # (a2,b1,c1), (a2,b1,c2), (a2,b2,c1), (a2,b2,c2)
            
            trans_indices = []
            trans_indices.append(triple_to_idx[tuple(sorted((a1,b1,c1)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a1,b1,c2)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a1,b2,c1)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a1,b2,c2)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a2,b1,c1)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a2,b1,c2)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a2,b2,c1)))])
            trans_indices.append(triple_to_idx[tuple(sorted((a2,b2,c2)))])
            
            # Create a bitmask for these 8 triples
            mask = 0
            for idx in trans_indices:
                mask |= (1 << idx)
            
            # Store the mask and the list of indices for quick checking
            # We only need to store unique masks to save memory/time later? 
            # Actually, different matchings on the same 6 points might produce different sets of 8 triples.
            # Yes, they do. So we store all.
            forbidden_masks.append((mask, trans_indices))
            
    # Number of forbidden configs: C(16,6) * 15 = 8008 * 15 = 120120.
    # This is a manageable number.
    
    # 3. Local Search Strategy
    # State: A boolean array of length T indicating which triples are selected.
    # Objective: Maximize sum(selected) such that no forbidden mask is fully covered by selected.
    # i.e., for all (mask, indices) in forbidden_masks: sum(selected[i] for i in indices) < 8.
    
    # We represent the state as a numpy array of 0s and 1s.
    # We maintain a count for each forbidden config: how many of its 8 triples are currently selected.
    # If any count == 8, the state is invalid.
    
    # Initialization:
    # Start with a random valid configuration or an empty one?
    # Empty is valid (0 triples).
    # We can try to build up a dense random set and repair it.
    
    best_score = 0
    best_state = None
    
    # Helper to check validity and get score
    def get_score(state):
        return np.sum(state)
    
    # Helper to check if a state is valid
    # We maintain counts for efficiency.
    
    def run_local_search():
        # Initialize state
        # Randomly select ~50% of triples? Might be too many invalids.
        # Let's start with a sparse random set, e.g., 20% density.
        state = np.zeros(T, dtype=np.int8)
        # Pick random indices to start
        num_initial = int(T * 0.2)
        init_indices = np.random.choice(T, num_initial, replace=False)
        state[init_indices] = 1
        
        # Initialize counts for each forbidden config
        # counts[j] = number of selected triples in forbidden config j
        # This array has size 120,120.
        counts = np.zeros(len(forbidden_masks), dtype=np.int8)
        
        # Precompute for each triple index, which forbidden configs it belongs to.
        # This is the reverse index.
        # triple_to_configs[i] = list of indices j in forbidden_masks where triple i is present.
        triple_to_configs = [[] for _ in range(T)]
        for j, (mask, indices) in enumerate(forbidden_masks):
            for idx in indices:
                triple_to_configs[idx].append(j)
        
        # Calculate initial counts
        for i in init_indices:
            for j in triple_to_configs[i]:
                counts[j] += 1
        
        current_score = np.sum(state)
        
        # If initial state is invalid, we need to repair it or start fresh.
        # For simplicity of the inner loop, let's assume we only move if it stays valid or improves.
        # Actually, standard approach: 
        # 1. If invalid, remove a triple from a violated config.
        # 2. If valid, try to add a triple. If it causes violation, reject or swap.
        
        # Let's use a simpler "Add if valid, else try to remove" heuristic first, 
        # but that might get stuck. 
        # Better: Simulated Annealing.
        # Move: Flip a bit.
        # If new state is valid and score increases -> accept.
        # If new state is valid and score decreases -> accept with prob exp(-delta/T).
        # If new state is invalid -> reject (or accept with very low prob? No, invalid is -inf).
        
        # To speed up, we only consider flips that might improve things.
        # Adding a triple (0->1):
        #   Must not cause any count to reach 8.
        # Removing a triple (1->0):
        #   Always valid (reduces counts).
        
        # Algorithm:
        # 1. Start with a valid state. (If initial is invalid, repair by removing random triples from violated configs until valid).
        
        # Repair initial state
        while True:
            violated = np.where(counts == 8)[0]
            if len(violated) == 0:
                break
            # Pick a random violated config
            v_idx = violated[np.random.randint(len(violated))]
            # Pick a random triple from this config to remove
            cfg_mask, cfg_indices = forbidden_masks[v_idx]
            # Find a triple in this config that is currently 1
            candidates = [i for i in cfg_indices if state[i] == 1]
            if candidates:
                rem_idx = candidates[np.random.randint(len(candidates))]
                state[rem_idx] = 0
                current_score -= 1
                # Update counts
                for j in triple_to_configs[rem_idx]:
                    counts[j] -= 1
            else:
                # This shouldn't happen if counts are consistent, but just in case
                break
        
        best_local_score = current_score
        best_local_state = state.copy()
        
        temp = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 2:
            # Choose a random triple to flip
            i = np.random.randint(T)
            
            if state[i] == 0:
                # Try to add
                # Check if adding i causes any violation
                # It affects counts in triple_to_configs[i]
                valid_add = True
                affected_configs = triple_to_configs[i]
                for j in affected_configs:
                    if counts[j] == 7:
                        # Adding this would make it 8
                        valid_add = False
                        break
                
                if valid_add:
                    # Apply change
                    state[i] = 1
                    current_score += 1
                    for j in affected_configs:
                        counts[j] += 1
                    
                    # Update best
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_state = state.copy()
                else:
                    # Reject
                    pass
            else:
                # Try to remove
                # Always valid
                state[i] = 1
                current_score -= 1
                affected_configs = triple_to_configs[i]
                for j in affected_configs:
                    counts[j] -= 1
                
                # Update best (unlikely to be best, but keep track)
                # We don't really want to remove unless we're stuck or swapping.
                # Pure removal reduces score. We should prefer swaps.
                # Let's refine: if we can't add, maybe we should try to remove a "bad" one and add a "good" one?
                # For now, simple flip.
                
            # Cooling
            temp *= cooling
            if temp < 0.1:
                break
        
        return best_local_score, best_local_state

    # Run multiple restarts
    random.seed(42)
    np.random.seed(42)
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        score, state = run_local_search()
        
        if score > best_score:
            best_score = score
            best_state = state.copy()
        
        # If we found a very high score, maybe stop early?
        # Max possible is 560.
        # If best_score > 400, it's quite good.
        
    if best_state is None:
        return []
    
    # Convert state back to list of triples
    result = []
    for i in range(T):
        if best_state[i] == 1:
            result.append(list(all_triples[i]))
            
    return result

