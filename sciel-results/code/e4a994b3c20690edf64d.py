import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Generate all K(2,2,2) configurations efficiently
    configs = []
    config_id_map = {}
    triples_in_config = []
    points = list(range(N))

    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairings = [
            ((p0, p1), (p2, p3), (p4, p5)),
            ((p0, p1), (p2, p4), (p3, p5)),
            ((p0, p1), (p2, p5), (p3, p4)),
            ((p0, p2), (p1, p3), (p4, p5)),
            ((p0, p2), (p1, p4), (p3, p5)),
            ((p0, p2), (p1, p5), (p3, p4)),
            ((p0, p3), (p1, p2), (p4, p5)),
            ((p0, p3), (p1, p4), (p2, p5)),
            ((p0, p3), (p1, p5), (p2, p4)),
            ((p0, p4), (p1, p2), (p3, p5)),
            ((p0, p4), (p1, p3), (p2, p5)),
            ((p0, p4), (p1, p5), (p2, p3)),
            ((p0, p5), (p1, p2), (p3, p4)),
            ((p0, p5), (p1, p3), (p2, p4)),
            ((p0, p5), (p1, p4), (p2, p3))
        ]
        
        for pair_set in pairings:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
    
    num_configs = len(configs)

    # Build reverse index: Triple -> List of Config IDs
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # Precompute conflict graph (adjacency) for triples that share a config
    # This is used for local search
    # Since num_triples = 560, a 560x560 adjacency matrix is small (313k entries)
    # Or we can just use the config counts for validity checks which is faster.
    
    # Data structures for the optimization
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int32)
    
    # Helper functions for state updates
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

    def get_initial_lines():
        """
        Randomized greedy initialization with a twist:
        We use a weighted random selection where the weight is inversely proportional
        to the 'tightness' of the configs, or simply random shuffle but with multiple
        restarts to find a good starting point for the local search.
        """
        present[:] = False
        config_counts[:] = 0
        lines = []
        remaining = list(range(num_triples))
        random.shuffle(remaining)
        
        for t_idx in remaining:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                lines.append(t_idx)
                
        return lines

    def walk_sat_search(lines, num_steps=5000):
        """
        Local search with '1-out, k-in' moves.
        If we remove a triple, we might be able to add multiple triples.
        We evaluate the net gain.
        """
        lines_set = set(lines)
        
        for step in range(num_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random triple to remove
            t_in = random.choice(list(lines_set))
            remove_triple(t_in)
            lines_set.discard(t_in)
            
            # Try to add as many triples as possible
            added = []
            # We want to prioritize triples that are now valid
            # A simple strategy: iterate through all non-present triples, 
            # but that's slow. 
            # Better: check triples that were invalid due to t_in.
            # Actually, just checking all is fine for 560 triples if we do it smart.
            # But we can just try to add any valid triple.
            # To maximize gain, we should try to add triples that are "most blocked" by t_in?
            # No, t_in is removed, so it unblocks others.
            
            # Let's try to add triples in a random order, or ordered by potential.
            # For speed, let's just iterate through a random subset or all.
            # Since 560 is small, iterating all is ~O(560 * avg_degree).
            # Let's try to add any valid triple.
            
            potential_adds = []
            for t_out in range(num_triples):
                if t_out not in lines_set and is_valid_addition(t_out):
                    potential_adds.append(t_out)
            
            if not potential_adds:
                # No improvement possible, put it back
                add_triple(t_in)
                lines_set.add(t_in)
                continue
                
            # How many to add?
            # We can add up to len(potential_adds).
            # But adding one might block others? No, adding increases counts, so it blocks more.
            # We want to maximize the number added.
            # This is a small set packing problem on the candidates.
            # Greedy: pick the one with lowest degree among candidates?
            # Or just pick one randomly?
            
            # Let's try a greedy fill:
            # While there are valid candidates, pick one and add it.
            # To avoid O(N^2) inner loop, we'll just do a few random attempts or a simple greedy.
            
            added_count = 0
            # Shuffle candidates
            random.shuffle(potential_adds)
            for t_out in potential_adds:
                if is_valid_addition(t_out):
                    add_triple(t_out)
                    lines_set.add(t_out)
                    added_count += 1
                if added_count >= 1: # If we added at least 1, it's an improvement
                    # We can stop if we added more than 1? 
                    # The move is "remove 1, add k". If k > 1, it's a strict improvement.
                    # If k == 1, it's neutral (size same).
                    # If k == 0, it's a decrease.
                    pass
            
            # If added_count > 1, we improved. Keep it.
            # If added_count == 1, size is same. Might help future moves? 
            # Standard WalkSAT accepts neutral moves sometimes.
            # If added_count == 0, revert.
            
            if added_count <= 1:
                # Revert the addition and the removal?
                # If added_count == 1, we added 1 for 1 removed. Net 0.
                # Let's revert to keep the original state unless we get a strict improvement.
                # Actually, allowing neutral moves can help escape plateaus.
                # But for simplicity and strict improvement first:
                # If added_count <= 1, revert everything.
                for t_out in lines_set:
                    if t_out != t_in:
                        # We need to know which ones were added in this step
                        pass
                # This rollback is messy. 
                # Better approach: 
                # 1. Save state
                # 2. Remove t_in
                # 3. Try to add
                # 4. If size increased, accept. Else revert.
                
                # Let's refactor the move evaluation.
                pass

        return list(lines_set)
    
    # Refined Local Search
    def local_search(lines, max_iters=2000):
        lines_set = set(lines)
        current_size = len(lines_set)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Try to add any valid triple (immediate improvement)
            added = False
            # Random order to avoid bias
            candidates = [t for t in range(num_triples) if t not in lines_set]
            random.shuffle(candidates)
            for t in candidates:
                if is_valid_addition(t):
                    add_triple(t)
                    lines_set.add(t)
                    current_size += 1
                    added = True
                    break
                    
            if added:
                continue
                
            # If no simple add, try 1-out, k-in
            t_in = random.choice(list(lines_set))
            remove_triple(t_in)
            lines_set.discard(t_in)
            
            added_cnt = 0
            # Try to add triples. We want to add as many as possible to ensure k > 1.
            # But adding one might block another.
            # Simple greedy: pick valid ones one by one.
            potential = [t for t in range(num_triples) if t not in lines_set]
            random.shuffle(potential)
            
            # We only care if we can add at least 2 to get a net gain of +1.
            # Or if we add 1, it's neutral.
            # Let's try to add up to 3.
            for t_out in potential:
                if added_cnt >= 2:
                    break
                if is_valid_addition(t_out):
                    add_triple(t_out)
                    lines_set.add(t_out)
                    added_cnt += 1
                    
            if added_cnt > 1:
                current_size += (added_cnt - 1)
                # Accept
            else:
                # Revert
                for t_out in list(lines_set):
                    if t_out != t_in:
                        # We need to identify which were added
                        pass
                # Rollback logic:
                # Remove t_outs we added
                # Add t_in back
                # We don't have a log of added t_outs easily here without saving.
                # Let's save the state before the move.
                pass
        
        return list(lines_set)

    # Even simpler and more robust local search:
    def fast_local_search(lines, max_iters=10000):
        lines_set = set(lines)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # 1. Try to add
            improved = False
            candidates = [t for t in range(num_triples) if t not in lines_set]
            random.shuffle(candidates)
            for t in candidates:
                if is_valid_addition(t):
                    add_triple(t)
                    lines_set.add(t)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-out, k-in (k>=2)
            t_in = random.choice(list(lines_set))
            
            # Save current config counts to allow cheap rollback? 
            # No, just remove, try, and if fail, revert.
            
            remove_triple(t_in)
            lines_set.discard(t_in)
            
            added_t = []
            # Try to add 2 triples
            # We need to find 2 triples that are mutually compatible and compatible with the rest.
            # This is a small subproblem.
            # Let's just try to add any valid triple. If we find 2, we win.
            potential = [t for t in range(num_triples) if t not in lines_set]
            random.shuffle(potential)
            
            for t_out in potential:
                if len(added_t) >= 2:
                    break
                if is_valid_addition(t_out):
                    add_triple(t_out)
                    lines_set.add(t_out)
                    added_t.append(t_out)
            
            if len(added_t) >= 2:
                # Success, keep
                pass
            else:
                # Failure, revert
                for t_out in added_t:
                    remove_triple(t_out)
                    lines_set.discard(t_out)
                add_triple(t_in)
                lines_set.add(t_in)
                
        return list(lines_set)

    best_set = []
    best_count = 0
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() < deadline - 10:
        # 1. Initialize
        lines = get_initial_lines()
        count = len(lines)
        
        # 2. Local Search
        if time.time() < deadline - 10:
            lines = fast_local_search(lines, max_iters=5000)
            count = len(lines)
            
        if count > best_count:
            best_count = count
            best_set = lines
            num_restarts += 1
            
    # Final polish on best set
    if best_set and time.time() < deadline - 5:
        # Reset state to best_set
        present[:] = False
        config_counts[:] = 0
        for t in best_set:
            add_triple(t)
            
        lines = fast_local_search(best_set, max_iters=2000)
        if len(lines) > best_count:
            best_set = lines
            best_count = len(lines)

    result = [all_triples[i] for i in best_set]
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
