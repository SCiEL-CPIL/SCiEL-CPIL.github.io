import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    timeout = 85.0
    
    N = 128
    all_numbers = list(range(1, N + 1))
    
    # Precompute LCMs
    lcm_table = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_table[(a, b)] = lcm(a, b)
            
    # Find all bad triples
    bad_triples = []
    for a, b, c in combinations(all_numbers, 3):
        if lcm_table[(a, b)] == lcm_table[(a, c)] == lcm_table[(b, c)]:
            bad_triples.append((a, b, c))
            
    if not bad_triples:
        return all_numbers

    # Map vertex to list of triple indices
    vertex_to_triple_indices = {x: [] for x in all_numbers}
    for i, t in enumerate(bad_triples):
        for x in t:
            vertex_to_triple_indices[x].append(i)
            
    # Global state
    best_removed = N
    best_solution = set()
    
    def update_best(available, removed_count):
        nonlocal best_removed, best_solution
        kept = N - removed_count
        if kept > N - best_removed:
            best_removed = removed_count
            best_solution = set(available)
            
    def greedy_vertex_cover():
        removed = set()
        available = set(all_numbers)
        active_indices = set(range(len(bad_triples)))
        
        while active_indices:
            # Pick a triple
            # Heuristic: pick triple with highest sum of degrees in active set
            # For speed, just pick the first one
            t_idx = next(iter(active_indices))
            a, b, c = bad_triples[t_idx]
            
            # Count active degrees
            def active_deg(x):
                count = 0
                for idx in vertex_to_triple_indices[x]:
                    if idx in active_indices:
                        count += 1
                return count
                
            da, db, dc = active_deg(a), active_deg(b), active_deg(c)
            max_d = max(da, db, dc)
            candidates = []
            if da == max_d: candidates.append(a)
            if db == max_d: candidates.append(b)
            if dc == max_d: candidates.append(c)
            
            to_remove = candidates[0]
            removed.add(to_remove)
            available.remove(to_remove)
            
            # Update active_indices: remove all triples containing to_remove
            for idx in vertex_to_triple_indices[to_remove]:
                if idx in active_indices:
                    active_indices.remove(idx)
                    
        return removed

    # Initialize best with greedy
    init_removed = greedy_vertex_cover()
    best_removed = len(init_removed)
    best_solution = set(all_numbers) - init_removed
    
    # B&B
    def branch_and_bound(available, removed_count, active_indices):
        nonlocal best_removed
        
        if time.time() - start_time > timeout:
            return
            
        if not active_indices:
            update_best(available, removed_count)
            return
            
        # Matching Bound
        used = set()
        match_count = 0
        for idx in active_indices:
            a, b, c = bad_triples[idx]
            if a not in used and b not in used and c not in used:
                used.add(a)
                used.add(b)
                used.add(c)
                match_count += 1
                
        if removed_count + match_count >= best_removed:
            return
            
        # Pick a triple to branch on
        # Heuristic: Pick triple with max sum of active degrees
        best_idx = None
        best_score = -1
        for idx in active_indices:
            a, b, c = bad_triples[idx]
            # Approx score
            score = len(vertex_to_triple_indices[a]) + len(vertex_to_triple_indices[b]) + len(vertex_to_triple_indices[c])
            if score > best_score:
                best_score = score
                best_idx = idx
                # Early break for speed? 
                # Let's check a few
                # break 
                
        if best_idx is None:
            update_best(available, removed_count)
            return
            
        a, b, c = bad_triples[best_idx]
        
        # Branch
        # Order by active degree (approx)
        def active_deg(x):
            count = 0
            for idx in vertex_to_triple_indices[x]:
                if idx in active_indices:
                    count += 1
            return count
            
        da, db, dc = active_deg(a), active_deg(b), active_deg(c)
        
        candidates = [(da, a), (db, b), (dc, c)]
        candidates.sort(key=lambda x: -x[0])
        
        for _, x_to_remove in candidates:
            new_available = available - {x_to_remove}
            # New active indices: remove all triples containing x_to_remove
            new_active = set(active_indices)
            for idx in vertex_to_triple_indices[x_to_remove]:
                if idx in new_active:
                    new_active.remove(idx)
                    
            branch_and_bound(new_available, removed_count + 1, new_active)
            if time.time() - start_time > timeout:
                return

    branch_and_bound(set(all_numbers), 0, set(range(len(bad_triples))))
    
    return list(best_solution) if best_solution else [1]
