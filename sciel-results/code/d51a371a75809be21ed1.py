import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is formed by 3 disjoint pairs (a,b), (c,d), (e,f)
    # The 8 transversal triples are {x,y,z} where x in {a,b}, y in {c,d}, z in {e,f}
    
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    # We need to iterate over all sets of 6 distinct points
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        # There are 15 ways to partition 6 points into 3 pairs
        # But we just need to iterate over all distinct sets of 3 disjoint pairs
        # Actually, the definition of K(2,2,2) in hypergraphs usually refers to the complete 3-partite 3-uniform hypergraph.
        # The vertices are partitioned into 3 sets of size 2.
        # Let's generate all partitions of 6 points into 3 pairs.
        
        # To avoid duplicates, we can fix the order of the pairs? 
        # No, the pairs are unordered, but the set of 3 pairs is what matters.
        # Let's just generate all combinations of 3 disjoint pairs from the 6 points.
        
        # Pairs from 6 points:
        # (0,1), (2,3), (4,5)
        # (0,1), (2,4), (3,5)
        # (0,1), (2,5), (3,4)
        # (0,2), (1,3), (4,5)
        # (0,2), (1,4), (3,5)
        # (0,2), (1,5), (3,4)
        # (0,3), (1,2), (4,5)
        # (0,3), (1,4), (2,5)
        # (0,3), (1,5), (2,4)
        # (0,4), (1,2), (3,5)
        # (0,4), (1,3), (2,5)
        # (0,4), (1,5), (2,3)
        # (0,5), (1,2), (3,4)
        # (0,5), (1,3), (2,4)
        # (0,5), (1,4), (2,3)
        
        pairs_list = [
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
        
        for pair_set in pairs_list:
            # Normalize the pairs for deduplication
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
    
    # Build reverse mapping: for each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State variables
    # present[i] is True if triple i is in the set
    present = np.zeros(num_triples, dtype=bool)
    # config_counts[c] is the number of triples from config c that are present
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def get_initial_lines():
        # Affine lines in F2^4
        lines = []
        seen = set()
        for x in range(N):
            for y in range(N):
                if x == y: continue
                z = x ^ y
                if z == x or z == y: continue
                t = tuple(sorted((x, y, z)))
                if t not in seen:
                    seen.add(t)
                    lines.append(t)
        return lines

    def solve_csp(initial_set, seed):
        """
        Uses randomized backtracking with forward checking to extend the initial set.
        """
        random.seed(seed)
        
        # Reset state
        present[:] = False
        config_counts[:] = 0
        
        current_set = set()
        
        # Add initial set
        for t in initial_set:
            idx = triple_to_idx[t]
            if is_valid_addition(idx):
                add_triple(idx)
                current_set.add(idx)
        
        count = len(current_set)
        
        # Candidates to add: all triples not in current_set
        # We will use a greedy randomized approach with backtracking logic
        
        # To make it efficient, we don't do full backtracking on 560 variables.
        # We do a "greedy with lookahead" or "randomized greedy with restarts"
        # which is effectively what the current code does, but we can improve the ordering.
        
        # Strategy:
        # 1. Identify "critical" triples: those that are in many configs (high degree).
        #    Actually, we want to add triples that are "safe".
        #    A triple is safe if it doesn't complete any config.
        #    Let's prioritize triples that are in fewer configs? 
        #    No, let's prioritize triples that are "forced" or "highly connected" in a different way?
        
        # Let's try a different heuristic:
        # Randomly select a subset of missing triples.
        # Try to add them in a random order, but with a check.
        
        # Better: Simulated Annealing style?
        # Let's stick to a robust local search but with a better move set:
        # Move: Remove a set of k triples, add a set of m triples (m > k) if possible.
        
        # Given the time constraint, a simple high-quality greedy + local search is best.
        # The key improvement: Use "weighted" random selection for adding triples.
        # Weight = 1 / (number of configs it is in). 
        # This favors triples that are "less restrictive" to add, potentially allowing more total triples?
        # Or weight = 1 (uniform).
        
        # Let's try: Randomize the order of missing triples, but bias towards those
        # that share points with already selected triples? (Clustering)
        
        # Let's implement a specific "WalkSAT" variant for Max-SAT.
        # Variables: x_i (is triple i selected?)
        # Clauses: For each config C, NOT (AND of all 8 x_i in C).
        # This is a Max-SAT problem where we want to maximize sum(x_i) subject to clauses.
        
        # We will use a simple local search that:
        # 1. Starts with a valid set.
        # 2. Repeatedly tries to add a random missing triple.
        # 3. If it fails, it identifies the conflicting configs.
        # 4. It removes a random triple from one of the conflicting configs.
        # 5. It retries the addition.
        
        # This is the same as the current code's "kick" mechanism but applied more systematically.
        
        improved = True
        no_improve_count = 0
        max_no_improve = 50
        kick_threshold = 20
        
        while improved and time.time() < deadline - 5:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > kick_threshold:
                # Perform a kick: remove a few random triples to escape local optimum
                kick_size = random.randint(2, 5)
                if count > kick_size:
                    to_remove = random.sample(list(current_set), kick_size)
                    for t in to_remove:
                        remove_triple(t)
                        current_set.remove(t)
                    count -= kick_size
                    improved = True
                    no_improve_count = 0
                    continue

            missing = [i for i in range(num_triples) if not present[i]]
            if not missing:
                break
            
            # Randomize order
            random.shuffle(missing)
            
            added_any = False
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    count += 1
                    added_any = True
                    improved = True
                    no_improve_count = 0
                    if time.time() > deadline - 5:
                        break
            
            if not added_any and time.time() < deadline - 5:
                # Try to swap: remove one, add one
                # Pick a random missing triple
                t_idx = random.choice(missing)
                conflicts = []
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        conflicts.append(cid)
                        
                if conflicts:
                    cid = random.choice(conflicts)
                    present_in_cid = []
                    for t in triples_in_config[cid]:
                        if present[t]:
                            present_in_cid.append(t)
                            
                    if present_in_cid:
                        u_idx = random.choice(present_in_cid)
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        count -= 1
                        
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                            improved = True
                            no_improve_count = 0
                        else:
                            # Revert
                            add_triple(u_idx)
                            current_set.add(u_idx)
                            count += 1
                            
        return count, list(current_set)

    # Initial set: Affine lines
    initial_lines = get_initial_lines()
    
    best_count = 0
    best_set = []
    
    # Run multiple iterations with different seeds to find the best local optimum
    # The initial set is the same, but the randomization in the search varies
    
    num_runs = 10
    for i in range(num_runs):
        if time.time() > deadline - 10:
            break
        count, s = solve_csp(initial_lines, seed=1000 + i)
        if count > best_count:
            best_count = count
            best_set = s
            
    # If time remains, try random greedy starts
    if time.time() < deadline - 20:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            # Reset
            present[:] = False
            config_counts[:] = 0
            current_set = set()
            
            # Random greedy
            order = list(range(num_triples))
            random.shuffle(order)
            for t_idx in order:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
            
            init_triples = [all_triples[i] for i in current_set]
            count2, s2 = solve_csp(init_triples, seed=2000 + i)
            
            if count2 > best_count:
                best_count = count2
                best_set = s2

    result = [all_triples[i] for i in best_set]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
