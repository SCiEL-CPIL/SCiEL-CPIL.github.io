import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill
    
    d = 7
    n = 1 << d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Build all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_idx[e] = len(all_edges)
                all_edges.append(e)
    
    # Build all 4-cycles (squares)
    four_cycles = []
    seen_squares = set()
    for v in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                e1 = v ^ (1 << i)
                e2 = v ^ (1 << j)
                e3 = v ^ (1 << i) ^ (1 << j)
                edges = [
                    (min(v, e1), max(v, e1)),
                    (min(v, e2), max(v, e2)),
                    (min(e1, e3), max(e1, e3)),
                    (min(e2, e3), max(e2, e3)),
                ]
                sq_key = frozenset(edges)
                if sq_key in seen_squares:
                    continue
                seen_squares.add(sq_key)
                four_cycles.append(edges)
    
    # For each edge, list of cycle indices it belongs to
    edge_cycles = [[] for _ in range(len(all_edges))]
    for ci, cycle in enumerate(four_cycles):
        for e in cycle:
            edge_cycles[edge_to_idx[e]].append(ci)
    
    # Precompute for each edge, the set of other edges in its squares (for fast candidate generation)
    edge_conflicts = [set() for _ in range(len(all_edges))]
    for ei in range(len(all_edges)):
        for ci in edge_cycles[ei]:
            for e in four_cycles[ci]:
                oi = edge_to_idx[e]
                if oi != ei:
                    edge_conflicts[ei].add(oi)
    
    # Greedy with random ordering
    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(len(all_edges)))
        rng.shuffle(order)
        chosen = [False] * len(all_edges)
        cycle_count = [0] * len(four_cycles)
        
        for ei in order:
            ok = True
            for ci in edge_cycles[ei]:
                if cycle_count[ci] == 3:
                    ok = False
                    break
            if ok:
                chosen[ei] = True
                for ci in edge_cycles[ei]:
                    cycle_count[ci] += 1
        return chosen, cycle_count
    
    # Local search: add edges, then 1-flip and 2-flip
    def local_search(chosen, max_time=100.0):
        n_edges = len(all_edges)
        n_cycles = len(four_cycles)
        cycle_count = [0] * n_cycles
        for ei in range(n_edges):
            if chosen[ei]:
                for ci in edge_cycles[ei]:
                    cycle_count[ci] += 1
        
        # First, try to add any edge that doesn't complete a cycle
        changed = True
        while changed:
            changed = False
            for ei in range(n_edges):
                if not chosen[ei]:
                    ok = True
                    for ci in edge_cycles[ei]:
                        if cycle_count[ci] == 3:
                            ok = False
                            break
                    if ok:
                        chosen[ei] = True
                        for ci in edge_cycles[ei]:
                            cycle_count[ci] += 1
                        changed = True
        
        # Now try 1-flip moves: remove one edge, add as many as possible
        end_time = time.time() + min(max_time, deadline - time.time() - 10)
        rng = random.Random(42)
        
        # Precompute list of chosen indices for fast access
        chosen_indices = [i for i in range(n_edges) if chosen[i]]
        
        # 1-flip: remove one, add many
        improved = True
        while improved and time.time() < end_time:
            improved = False
            rng.shuffle(chosen_indices)
            
            for e_out in chosen_indices:
                if time.time() > end_time:
                    break
                
                # Remove e_out
                chosen[e_out] = False
                for ci in edge_cycles[e_out]:
                    cycle_count[ci] -= 1
                
                # Try to add edges that were previously blocked
                # Candidates: edges in squares containing e_out
                candidates = list(edge_conflicts[e_out])
                candidates = [c for c in candidates if not chosen[c]]
                rng.shuffle(candidates)
                
                added_count = 0
                for c in candidates:
                    ok = True
                    for ci in edge_cycles[c]:
                        if cycle_count[ci] == 3:
                            ok = False
                            break
                    if ok:
                        chosen[c] = True
                        for ci in edge_cycles[c]:
                            cycle_count[ci] += 1
                        added_count += 1
                
                if added_count > 1:
                    # Improvement: removed 1, added > 1
                    improved = True
                    # Rebuild chosen_indices
                    chosen_indices = [i for i in range(n_edges) if chosen[i]]
                elif added_count == 1:
                    # Neutral: accept to explore
                    chosen_indices = [i for i in range(n_edges) if chosen[i]]
                else:
                    # No improvement, revert
                    chosen[e_out] = True
                    for ci in edge_cycles[e_out]:
                        cycle_count[ci] += 1
        
        # 2-flip moves: remove two, add two or more
        for _ in range(5000):
            if time.time() > end_time:
                break
            
            chosen_indices = [i for i in range(n_edges) if chosen[i]]
            if len(chosen_indices) < 2:
                break
            
            e1, e2 = rng.sample(chosen_indices, 2)
            
            # Remove them
            for ei in (e1, e2):
                chosen[ei] = False
                for ci in edge_cycles[ei]:
                    cycle_count[ci] -= 1
            
            # Try to add two random unchosen edges
            unchosen = [i for i in range(n_edges) if not chosen[i]]
            if len(unchosen) < 2:
                for ei in (e1, e2):
                    chosen[ei] = True
                    for ci in edge_cycles[ei]:
                        cycle_count[ci] += 1
                continue
            
            improved = False
            for _ in range(3):
                a, b = rng.sample(unchosen, 2)
                # Check if adding a and b is valid
                affected = set(edge_cycles[a] + edge_cycles[b])
                valid = True
                for ci in affected:
                    cnt = cycle_count[ci]
                    if a in [edge_to_idx[e] for e in four_cycles[ci]]:
                        cnt += 1
                    if b in [edge_to_idx[e] for e in four_cycles[ci]]:
                        cnt += 1
                    if cnt >= 4:
                        valid = False
                        break
                
                if valid:
                    chosen[a] = True
                    chosen[b] = True
                    for ci in edge_cycles[a]:
                        cycle_count[ci] += 1
                    for ci in edge_cycles[b]:
                        cycle_count[ci] += 1
                    improved = True
                    break
            
            if not improved:
                for ei in (e1, e2):
                    chosen[ei] = True
                    for ci in edge_cycles[ei]:
                        cycle_count[ci] += 1
        
        return chosen
    
    # Try multiple greedy constructions
    best_chosen = None
    best_count = 0
    for s in range(30):
        if time.time() > deadline - 40:
            break
        chosen, _ = greedy_random(s)
        cnt = sum(chosen)
        if cnt > best_count:
            best_count = cnt
            best_chosen = chosen[:]
    
    if best_chosen is None:
        best_chosen, _ = greedy_random(0)
        best_count = sum(best_chosen)
    
    # Local search on best
    best_chosen = local_search(best_chosen[:], max_time=80.0)
    best_count = sum(best_chosen)
    
    # Try a few more random restarts with local search
    for s in range(30, 50):
        if time.time() > deadline - 30:
            break
        chosen, _ = greedy_random(s)
        improved = local_search(chosen, max_time=30.0)
        cnt = sum(improved)
        if cnt > best_count:
            best_count = cnt
            best_chosen = improved
    
    # Final: try to add any remaining edges
    cycle_count = [0] * len(four_cycles)
    for ei in range(len(all_edges)):
        if best_chosen[ei]:
            for ci in edge_cycles[ei]:
                cycle_count[ci] += 1
    for ei in range(len(all_edges)):
        if not best_chosen[ei]:
            ok = True
            for ci in edge_cycles[ei]:
                if cycle_count[ci] == 3:
                    ok = False
                    break
            if ok:
                best_chosen[ei] = True
                for ci in edge_cycles[ei]:
                    cycle_count[ci] += 1
    
    result = [[all_edges[ei][0], all_edges[ei][1]] for ei in range(len(all_edges)) if best_chosen[ei]]
    return result
