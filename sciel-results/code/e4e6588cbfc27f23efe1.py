"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    d = 7
    n = 1 << d  # 128 vertices
    m = n * d // 2  # 448 edges
    
    # Build edge list and index mapping
    edges = []
    edge_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v, i))
                edge_idx[(u, v)] = idx
                edge_idx[(v, u)] = idx
                idx += 1
    
    # Build conflict structure: for each edge, which other edges form a C4 with it
    # Edge (u, v) with direction i. For each other direction j:
    #   u' = u ^ (1<<j), v' = v ^ (1<<j)
    #   The C4 is u-v-v'-u'-u, and the other 3 edges are:
    #   (u, u'), (v, v'), (u', v')
    conflicts = [set() for _ in range(m)]
    for idx, (u, v, i) in enumerate(edges):
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # The three other edges in the C4
            for a, b in [(u, u2), (v, v2), (u2, v2)]:
                other = edge_idx[(min(a, b), max(a, b))]
                conflicts[idx].add(other)
                conflicts[other].add(idx)
    
    # Convert to sorted lists for speed
    conflict_lists = [sorted(c) for c in conflicts]
    
    def greedy_random(seed):
        """Greedy with random edge ordering."""
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        
        chosen = [False] * m
        count = 0
        for idx in order:
            # Check if any conflict is already chosen
            ok = True
            for c in conflict_lists[idx]:
                if chosen[c]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
        return count, chosen
    
    def local_improve(chosen, seed):
        """Local search: try to add blocked edges by removing one conflicting edge."""
        rng = random.Random(seed)
        count = sum(chosen)
        chosen_set = set(i for i in range(m) if chosen[i])
        
        # Build list of unchosen edges and their conflicts
        unchosen = [i for i in range(m) if not chosen[i]]
        
        improved = True
        while improved:
            improved = False
            rng.shuffle(unchosen)
            for idx in unchosen:
                if chosen[idx]:
                    continue
                # Find conflicts that are chosen
                chosen_conflicts = [c for c in conflict_lists[idx] if chosen[c]]
                if not chosen_conflicts:
                    # Can add directly
                    chosen[idx] = True
                    chosen_set.add(idx)
                    count += 1
                    improved = True
                    unchosen.remove(idx)
                    continue
                
                # Try removing one conflict to add this edge
                # Only beneficial if removing one allows adding this + potentially others
                # Simple heuristic: try each conflict, see if net gain > 0
                for c in chosen_conflicts:
                    # If we remove c, can we add idx?
                    # Check if any other chosen conflict remains
                    remaining = [cc for cc in conflict_lists[idx] if cc != c and chosen[cc]]
                    if remaining:
                        continue
                    # Remove c, add idx
                    # But c might be needed by other edges... simple 1-swap
                    chosen[c] = False
                    chosen_set.discard(c)
                    chosen[idx] = True
                    chosen_set.add(idx)
                    # Count doesn't change (removed 1, added 1)
                    # But this might unblock other edges
                    improved = True
                    break
        
        # Second pass: try to add any unchosen edge whose conflicts are all unchosen
        for idx in list(unchosen):
            if chosen[idx]:
                continue
            ok = True
            for c in conflict_lists[idx]:
                if chosen[c]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                chosen_set.add(idx)
                count += 1
                improved = True
        
        return count, chosen
    
    best_count = 0
    best_chosen = None
    
    # Phase 1: Multiple greedy starts
    for seed in range(20):
        if time.time() > deadline:
            break
        count, chosen = greedy_random(seed)
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
    
    # Phase 2: Local improvement on best and near-best solutions
    if best_chosen is not None:
        count, chosen = local_improve(best_chosen[:], 42)
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
    
    # Phase 3: More greedy + local improve cycles
    for seed in range(20, 100):
        if time.time() > deadline:
            break
        count, chosen = greedy_random(seed)
        count2, chosen2 = local_improve(chosen, seed)
        if count2 > best_count:
            best_count = count2
            best_chosen = chosen2[:]
    
    # Phase 4: Additional local improvement passes on the best
    if best_chosen is not None:
        for seed in range(100, 120):
            if time.time() > deadline:
                break
            count, chosen = local_improve(best_chosen[:], seed)
            if count > best_count:
                best_count = count
                best_chosen = chosen[:]
    
    # Convert to output format
    result = []
    for i in range(m):
        if best_chosen[i]:
            u, v, _ = edges[i]
            result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
