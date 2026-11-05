import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    d = 7
    n = 2 ** d  # 128 vertices

    # Precompute all edges: (u, v, dim) where u < v and dim is the bit that differs
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    total_edges = len(all_edges)  # 448

    # Build a mapping from edge tuple to index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx

    # For each edge, precompute the "forbidden triples" - the other 3 edges that would complete a C4
    edge_to_forbidden_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        forbidden = []
        for j in range(d):
            if j != i:
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                e1 = (min(u, c), max(u, c))
                e2 = (min(v, e), max(v, e))
                e3 = (min(c, e), max(c, e))
                i1 = edge_to_idx.get(e1)
                i2 = edge_to_idx.get(e2)
                i3 = edge_to_idx.get(e3)
                if i1 is not None and i2 is not None and i3 is not None:
                    forbidden.append((i1, i2, i3))
        edge_to_forbidden_idx[idx] = forbidden

    # Precompute: for each edge, which edges block it (i.e., appear in a forbidden triple)
    # This helps with faster "blocking set" computation
    edge_blockers = [set() for _ in range(total_edges)]
    for idx in range(total_edges):
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            edge_blockers[idx].add(i1)
            edge_blockers[idx].add(i2)
            edge_blockers[idx].add(i3)

    def is_addable(idx, chosen_arr):
        """Check if edge idx can be added to chosen set (numpy bool array) without creating a C4"""
        for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
            if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                return False
        return True

    def greedy_randomized(seed=None):
        """Randomized greedy: shuffle edges and add if possible"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = np.zeros(total_edges, dtype=bool)
        order = list(range(total_edges))
        rng.shuffle(order)
        
        for idx in order:
            if is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
        
        return chosen_arr

    def greedy_mcs(seed=None):
        """Maximum Cardinality Search heuristic"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = np.zeros(total_edges, dtype=bool)
        remaining = set(range(total_edges))
        
        while remaining:
            if time.time() - start_time > time_limit - 5:
                break
            
            best_idx = None
            best_score = None
            
            if len(remaining) > 100:
                sample = rng.sample(list(remaining), 100)
            else:
                sample = list(remaining)
            
            for idx in sample:
                score = 0
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    c = (chosen_arr[i1] and chosen_arr[i2]) or \
                        (chosen_arr[i1] and chosen_arr[i3]) or \
                        (chosen_arr[i2] and chosen_arr[i3])
                    if c:
                        score += 1
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        score = 1000000
                        break
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None and best_score < 1000000:
                chosen_arr[best_idx] = True
                remaining.discard(best_idx)
            else:
                found = False
                for idx in sample:
                    if is_addable(idx, chosen_arr):
                        chosen_arr[idx] = True
                        remaining.discard(idx)
                        found = True
                        break
                if not found:
                    break
        
        return chosen_arr

    def local_search(chosen_arr, max_iters=10000, seed=None):
        """Local search: try to add unchosen edges, removing blocking edges if needed"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) == 0:
                break
            
            idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
            
            if is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
                current_size += 1
            else:
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        blocking.append((i1, i2, i3))
                
                if not blocking:
                    chosen_arr[idx] = True
                    current_size += 1
                else:
                    triple = blocking[rng.randint(0, len(blocking) - 1)]
                    for rem_idx in triple:
                        chosen_arr[rem_idx] = False
                        if is_addable(idx, chosen_arr):
                            chosen_arr[idx] = True
                            break
                        else:
                            chosen_arr[rem_idx] = True
        
        return chosen_arr

    def simulated_annealing(chosen_arr, max_iters=20000, seed=None):
        """Simulated Annealing"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        best_arr = chosen_arr.copy()
        best_size = current_size
        
        T_start = 5.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
        T = T_start
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            T *= cooling_rate
            
            unchosen = np.where(~chosen_arr)[0]
            chosen_indices = np.where(chosen_arr)[0]
            
            if len(unchosen) == 0:
                break
            
            if rng.random() < 0.7 and len(unchosen) > 0:
                idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
                
                if is_addable(idx, chosen_arr):
                    chosen_arr[idx] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_arr = chosen_arr.copy()
                else:
                    blocking = []
                    for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                        if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                            blocking.append((i1, i2, i3))
                    
                    if blocking:
                        triple = blocking[rng.randint(0, len(blocking) - 1)]
                        rem_idx = triple[rng.randint(0, 2)]
                        chosen_arr[rem_idx] = False
                        if is_addable(idx, chosen_arr):
                            chosen_arr[idx] = True
                            if current_size > best_size:
                                best_size = current_size
                                best_arr = chosen_arr.copy()
                        else:
                            chosen_arr[rem_idx] = True
            else:
                if len(chosen_indices) > 0:
                    rem_idx = int(chosen_indices[rng.randint(0, len(chosen_indices) - 1)])
                    old_size = current_size
                    chosen_arr[rem_idx] = False
                    current_size -= 1
                    
                    delta = -1
                    accept_prob = np.exp(-delta / T) if T > 1e-6 else 0.0
                    if rng.random() < accept_prob:
                        if current_size > best_size:
                            best_size = current_size
                            best_arr = chosen_arr.copy()
                    else:
                        chosen_arr[rem_idx] = True
                        current_size = old_size
        
        return best_arr

    def swap_and_add(chosen_arr, max_iters=5000, seed=None):
        """Try to swap 1 edge for 2 edges (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) < 2:
                break
            
            s = rng.sample(range(len(unchosen)), 2)
            idx1 = int(unchosen[s[0]])
            idx2 = int(unchosen[s[1]])
            
            if is_addable(idx1, chosen_arr) and is_addable(idx2, chosen_arr):
                chosen_arr[idx1] = True
                chosen_arr[idx2] = True
                current_size += 2
            else:
                block1 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block1.update([i1, i2, i3])
                
                block2 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block2.update([i1, i2, i3])
                
                candidates = list(block1 | block2)
                rng.shuffle(candidates)
                
                for rem_idx in candidates[:5]:
                    chosen_arr[rem_idx] = False
                    if is_addable(idx1, chosen_arr) and is_addable(idx2, chosen_arr):
                        chosen_arr[idx1] = True
                        chosen_arr[idx2] = True
                        current_size += 1
                        break
                    else:
                        chosen_arr[rem_idx] = True
        
        return chosen_arr

    def two_opt_search(chosen_arr, max_iters=3000, seed=None):
        """Try 2-Opt: remove 2 edges, add 2 edges (net 0), or remove 1 add 2 (net +1)"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) < 2:
                break
            
            idx1, idx2 = rng.sample(unchosen.tolist(), 2)
            idx1, idx2 = int(idx1), int(idx2)
            
            if is_addable(idx1, chosen_arr) and is_addable(idx2, chosen_arr):
                chosen_arr[idx1] = True
                chosen_arr[idx2] = True
                current_size += 2
            else:
                block1 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx1]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block1.update([i1, i2, i3])
                
                block2 = set()
                for (i1, i2, i3) in edge_to_forbidden_idx[idx2]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        block2.update([i1, i2, i3])
                
                # Try removing 1 edge to enable adding both
                candidates = list(block1 | block2)
                rng.shuffle(candidates)
                
                for rem_idx in candidates[:8]:
                    chosen_arr[rem_idx] = False
                    if is_addable(idx1, chosen_arr) and is_addable(idx2, chosen_arr):
                        chosen_arr[idx1] = True
                        chosen_arr[idx2] = True
                        current_size += 1
                        break
                    else:
                        chosen_arr[rem_idx] = True
        
        return chosen_arr

    def multi_start_local_search(chosen_arr, max_iters=8000, seed=None):
        """Multiple local search passes with random restarts from perturbed states"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        best_arr = chosen_arr.copy()
        best_size = int(chosen_arr.sum())
        
        for restart in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Perturb: randomly remove a few edges
            perturbed = best_arr.copy()
            chosen_indices = np.where(perturbed)[0]
            if len(chosen_indices) > 0:
                n_remove = max(1, len(chosen_indices) // 50)
                to_remove = rng.sample(chosen_indices.tolist(), min(n_remove, len(chosen_indices)))
                for idx in to_remove:
                    perturbed[idx] = False
            
            improved = local_search(perturbed, max_iters=max_iters // 3, seed=seed + restart)
            if int(improved.sum()) > best_size:
                best_size = int(improved.sum())
                best_arr = improved
        
        return best_arr

    best_solution = None
    best_size = 0

    # Phase 1: Multiple greedy strategies with different seeds
    num_restarts = 0
    while time.time() - start_time < time_limit - 40:
        seed = num_restarts
        
        if num_restarts % 3 == 0:
            solution = greedy_randomized(seed)
        elif num_restarts % 3 == 1:
            solution = greedy_mcs(seed)
        else:
            solution = greedy_randomized(seed + 1000)
        
        size = int(solution.sum())
        if size > best_size:
            best_size = size
            best_solution = solution
        num_restarts += 1
    
    # Phase 2: Local search on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 30:
        improved = local_search(best_solution, max_iters=15000, seed=42)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 3: Simulated Annealing
    if best_solution is not None and time.time() - start_time < time_limit - 20:
        improved = simulated_annealing(best_solution, max_iters=25000, seed=789)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 4: Swap and add
    if best_solution is not None and time.time() - start_time < time_limit - 15:
        improved = swap_and_add(best_solution, max_iters=8000, seed=123)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 5: 2-opt search
    if best_solution is not None and time.time() - start_time < time_limit - 10:
        improved = two_opt_search(best_solution, max_iters=5000, seed=456)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Phase 6: Multi-start local search
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        improved = multi_start_local_search(best_solution, max_iters=10000, seed=789)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    # Final polish: more local search iterations
    if best_solution is not None and time.time() - start_time < time_limit - 3:
        improved = local_search(best_solution, max_iters=5000, seed=12345)
        if int(improved.sum()) > best_size:
            best_size = int(improved.sum())
            best_solution = improved
    
    if best_solution is None:
        best_solution = greedy_randomized(0)
    
    # Convert to output format
    result = []
    for idx in np.where(best_solution)[0]:
        u, v, i = all_edges[idx]
        result.append([u, v])
    
    return result
