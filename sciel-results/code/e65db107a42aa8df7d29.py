import time
import random
from itertools import combinations
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute K(2,2,2) configurations
    configs = []
    triple_to_configs = [[] for _ in range(num_triples)]
    
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
            configs.append(list(trans_indices))
            
            for idx in trans_indices:
                triple_to_configs[idx].append(config_idx)
                
    num_configs = len(configs)
    
    # Precompute config sizes (should all be 8)
    config_size = [len(c) for c in configs]
    
    def greedy_hitting_set(seed=None):
        """Greedy hitting set with better scoring: prioritize triples that hit many active configs
        but appear in fewer total configs (less 'cost' to remove)."""
        if seed is not None:
            random.seed(seed)
        
        selected = [True] * num_triples
        config_active = [True] * num_configs
        config_count = [config_size[c] for c in range(num_configs)]  # selected count in each config
        
        # For each triple, track how many active configs it's in
        triple_active_count = [0] * num_triples
        # Total configs each triple appears in
        triple_total_count = [len(triple_to_configs[i]) for i in range(num_triples)]
        
        for i in range(num_triples):
            for c in triple_to_configs[i]:
                triple_active_count[i] += 1
        
        removed = [False] * num_triples
        removed_count = 0
        
        # Use a max-heap with score = active_count / (total_count + 1) to balance
        # We want high active_count but low total_count
        # Score: active_count * 1000 - total_count (to prioritize active configs first)
        
        def make_heap():
            heap = []
            for i in range(num_triples):
                if not removed[i] and triple_active_count[i] > 0:
                    score = triple_active_count[i] * 10000 - triple_total_count[i]
                    heapq.heappush(heap, (-score, i))
            return heap
        
        heap = make_heap()
        
        while heap and time.time() < deadline - 5:
            neg_score, t = heapq.heappop(heap)
            if removed[t] or triple_active_count[t] == 0:
                continue
            
            # Remove triple t
            removed[t] = True
            removed_count += 1
            
            # Update configs containing t
            for c in triple_to_configs[t]:
                config_count[c] -= 1
                if config_count[c] < config_size[c] and config_active[c]:
                    # Config is no longer active (not all 8 selected)
                    config_active[c] = False
                    # Decrement active count for other triples in this config
                    for u in configs[c]:
                        if u != t and not removed[u]:
                            triple_active_count[u] -= 1
                            if triple_active_count[u] > 0:
                                new_score = triple_active_count[u] * 10000 - triple_total_count[u]
                                heapq.heappush(heap, (-new_score, u))
            
            # Mark t as having 0 active count
            triple_active_count[t] = 0
        
        selected = [not r for r in removed]
        return selected, num_triples - removed_count

    def local_search(selected, size, max_time=None):
        """Local search: try to add safe triples, and use swaps to enable additions."""
        if max_time is None:
            max_time = deadline - 5
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Phase 1: Try to add removed triples that are safe
            for t in range(num_triples):
                if not selected[t]:
                    # Check if adding t creates any violation
                    safe = True
                    for c in triple_to_configs[t]:
                        # Count selected in c (excluding t which is not selected)
                        cnt = 0
                        for u in configs[c]:
                            if u != t and selected[u]:
                                cnt += 1
                        if cnt == config_size[c] - 1:
                            safe = False
                            break
                    if safe:
                        selected[t] = True
                        size += 1
                        improved = True
            
            if improved:
                continue
            
            # Phase 2: Try swaps to enable future additions
            # For each removed triple t, find a blocker config where all others are selected
            # Then try swapping one of the blockers with t
            swap_found = False
            for t in range(num_triples):
                if not selected[t] and time.time() > max_time:
                    break
                    
                # Find blocker configs
                blockers = []
                for c in triple_to_configs[t]:
                    cnt = 0
                    for u in configs[c]:
                        if u != t and selected[u]:
                            cnt += 1
                    if cnt == config_size[c] - 1:
                        blockers.append(c)
                
                if not blockers:
                    continue
                
                # Try swapping: remove one blocker triple s, add t
                for c in blockers:
                    for s in configs[c]:
                        if s == t or not selected[s]:
                            continue
                        
                        # Check if removing s and adding t is valid
                        # After swap: t is added, s is removed
                        # Check all configs containing t: need count < size
                        # Check all configs containing s: need count < size after removing s
                        
                        valid = True
                        
                        # Check configs containing t
                        for ct in triple_to_configs[t]:
                            cnt = 0
                            for u in configs[ct]:
                                if u == t:
                                    continue  # t will be added
                                if u == s:
                                    continue  # s will be removed
                                if selected[u]:
                                    cnt += 1
                            if cnt == config_size[ct] - 1:
                                # After adding t, count would be config_size[ct] -> violation
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Check configs containing s (but not t)
                        for cs in triple_to_configs[s]:
                            if t in configs[cs]:
                                continue  # Already checked
                            cnt = 0
                            for u in configs[cs]:
                                if u == s:
                                    continue  # s will be removed
                                if selected[u]:
                                    cnt += 1
                            if cnt == config_size[cs]:
                                # After removing s, count is still config_size[cs] -> but s was selected,
                                # so current count includes s. After removing s, count = cnt
                                # We need cnt < config_size[cs], which is always true since cnt <= config_size[cs]-1
                                # Actually this check is always valid since removing s decreases count
                                pass
                        
                        if valid:
                            # Perform swap
                            selected[s] = False
                            selected[t] = True
                            # Size stays the same
                            improved = True
                            swap_found = True
                            break
                    if swap_found:
                        break
                if swap_found:
                    break
            
            # Phase 3: If no swaps found, try random perturbations
            if not improved and time.time() < max_time - 2:
                # Randomly remove a few triples and re-add greedily
                if random.random() < 0.3:
                    # Perturb: remove 2-5 random selected triples
                    selected_indices = [i for i in range(num_triples) if selected[i]]
                    if len(selected_indices) > 5:
                        num_remove = random.randint(2, min(5, len(selected_indices) // 4))
                        to_remove = random.sample(selected_indices, num_remove)
                        for t in to_remove:
                            selected[t] = False
                            size -= 1
                        
                        # Now try to re-add
                        for t in range(num_triples):
                            if not selected[t]:
                                safe = True
                                for c in triple_to_configs[t]:
                                    cnt = 0
                                    for u in configs[c]:
                                        if u != t and selected[u]:
                                            cnt += 1
                                    if cnt == config_size[c] - 1:
                                        safe = False
                                        break
                                if safe:
                                    selected[t] = True
                                    size += 1
                        
                        improved = True
        
        return selected, size

    # Run initial greedy
    selected, size = greedy_hitting_set()
    selected, size = local_search(selected, size)
    
    best_selected = list(selected)
    best_size = size
    
    # Try multiple random restarts with different seeds
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline - 15:
        seed = rng.randint(0, 10000000)
        sel, sz = greedy_hitting_set(seed)
        sel, sz = local_search(sel, sz, max_time=deadline - 10)
        
        if sz > best_size:
            best_size = sz
            best_selected = list(sel)
        
        restart_count += 1
        # If we've done enough restarts or time is running out
        if time.time() > deadline - 20:
            break
        if restart_count > 50:
            break
    
    result = []
    for i in range(num_triples):
        if best_selected[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
