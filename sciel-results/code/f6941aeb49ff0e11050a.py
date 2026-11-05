import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute bitmask for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    # Precompute conflict graph: two triples conflict if they share 2+ points
    # Use bitmasks for fast intersection
    conflicts = [0] * n_triples  # bitmask of conflicting triples
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            common = mi & mj
            if bin(common).count('1') >= 2:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    def is_conflict_free(triple_set):
        """Check if a set of triples is conflict-free (linear)."""
        for i in triple_set:
            if conflicts[i] & triple_set:
                return False
        return True
    
    def compute_alpha(triple_set):
        """Compute independence number: largest subset of points containing no triple."""
        if not triple_set:
            return 16
        
        ts = set(triple_set)
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                has_triple = False
                for ti in ts:
                    if (triple_masks[ti] & smask) == triple_masks[ti]:
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0
    
    def compute_alpha_fast(triple_list):
        """Faster alpha using precomputed masks."""
        if not triple_list:
            return 16
        
        ts = set(triple_list)
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                has_triple = False
                for ti in ts:
                    if (triple_masks[ti] & smask) == triple_masks[ti]:
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0
    
    def find_max_independent_set_size(triple_list):
        """Find the size of the maximum independent set."""
        if not triple_list:
            return 16
        
        ts = set(triple_list)
        
        # Try from largest to smallest
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                has_triple = False
                for ti in ts:
                    if (triple_masks[ti] & smask) == triple_masks[ti]:
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0
    
    def greedy_construct(heuristic='random'):
        """Greedy construction of a linear triple system."""
        selected = []
        selected_set = set()
        point_usage = [0] * 16
        
        available = list(range(n_triples))
        if heuristic == 'random':
            random.shuffle(available)
        elif heuristic == 'balanced':
            # Prefer triples with less-used points
            def score(idx):
                t = all_triples[idx]
                return sum(point_usage[p] for p in t)
            available.sort(key=score)
        elif heuristic == 'dense':
            # Prefer triples that are more "central"
            random.shuffle(available)
        
        for idx in available:
            if time.time() > deadline - 5:
                break
            
            # Check if this triple conflicts with any selected triple
            can_add = True
            for sel in selected_set:
                if conflicts[idx] & (1 << sel):
                    can_add = False
                    break
            
            if can_add:
                selected.append(idx)
                selected_set.add(idx)
                for p in all_triples[idx]:
                    point_usage[p] += 1
        
        return selected
    
    def local_search(triple_list, max_iters=200):
        """Local search with 1-swap moves to minimize alpha."""
        current = list(triple_list)
        current_set = set(current)
        current_alpha = compute_alpha_fast(current)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            if time.time() > deadline - 5:
                break
            
            # Try removing one triple
            for i in range(len(current)):
                new_set = current_set - {current[i]}
                new_alpha = compute_alpha_fast(list(new_set))
                if new_alpha < current_alpha:
                    current = list(new_set)
                    current_set = new_set
                    current_alpha = new_alpha
                    improved = True
                    break
            
            # Try adding one triple
            if not improved:
                # Find candidates that don't conflict
                candidates = []
                for idx in range(n_triples):
                    if idx in current_set:
                        continue
                    # Check conflicts
                    can_add = True
                    for sel in current_set:
                        if conflicts[idx] & (1 << sel):
                            can_add = False
                            break
                    if can_add:
                        candidates.append(idx)
                
                random.shuffle(candidates)
                for idx in candidates[:30]:
                    new_set = current_set | {idx}
                    new_alpha = compute_alpha_fast(list(new_set))
                    if new_alpha < current_alpha:
                        current = list(new_set)
                        current_set = new_set
                        current_alpha = new_alpha
                        improved = True
                        break
            
            # Try 1-swap: remove one, add another
            if not improved and len(current) > 0:
                for i in range(len(current)):
                    removed = current[i]
                    temp_set = current_set - {removed}
                    
                    # Find candidates that don't conflict with temp_set
                    candidates = []
                    for idx in range(n_triples):
                        if idx in temp_set:
                            continue
                        # Check conflicts
                        can_add = True
                        for sel in temp_set:
                            if conflicts[idx] & (1 << sel):
                                can_add = False
                                break
                        if can_add:
                            candidates.append(idx)
                    
                    random.shuffle(candidates)
                    for idx in candidates[:20]:
                        new_set = temp_set | {idx}
                        new_alpha = compute_alpha_fast(list(new_set))
                        if new_alpha < current_alpha:
                            current = list(new_set)
                            current_set = new_set
                            current_alpha = new_alpha
                            improved = True
                            break
                    if improved:
                        break
        
        return current, current_alpha
    
    best_alpha = 16
    best_triples = []
    
    # Phase 1: Multiple diverse greedy constructions + local search
    heuristics = ['random', 'balanced', 'dense']
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        heuristic = heuristics[restart % len(heuristics)]
        system = greedy_construct(heuristic)
        
        if not system:
            continue
        
        optimized, alpha = local_search(system, max_iters=50)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = optimized[:]
            
            if best_alpha <= 6:
                break
    
    # Phase 2: If we haven't done well, try a more systematic branch-and-bound
    if best_alpha > 7 and time.time() < deadline - 30:
        # Try to build a system that covers many small subsets
        # Use a greedy approach: pick triples that hit the most "uncovered" subsets
        
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            # Start empty and greedily add triples
            current = []
            current_set = set()
            
            # Score each triple by how many potential independent sets it hits
            point_usage = [0] * 16
            
            for _ in range(25):  # Add up to 25 triples
                if time.time() > deadline - 5:
                    break
                
                best_idx = None
                best_score = -1
                
                # Try a random subset of triples for speed
                sample_size = min(300, n_triples)
                sample = random.sample(range(n_triples), sample_size)
                
                for idx in sample:
                    if idx in current_set:
                        continue
                    
                    # Check conflicts
                    can_add = True
                    for sel in current_set:
                        if conflicts[idx] & (1 << sel):
                            can_add = False
                            break
                    
                    if not can_add:
                        continue
                    
                    # Score: prefer triples that use less-used points
                    t = all_triples[idx]
                    score = sum(1 - point_usage[p] / 6 for p in t)
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx is None:
                    break
                
                current.append(best_idx)
                current_set.add(best_idx)
                for p in all_triples[best_idx]:
                    point_usage[p] += 1
            
            # Evaluate
            alpha = compute_alpha_fast(current)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = current[:]
                
                if best_alpha <= 6:
                    break
    
    # Phase 3: Final local search refinement
    if best_triples and time.time() < deadline - 5:
        refined, final_alpha = local_search(best_triples, max_iters=100)
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            best_triples = refined
    
    # Convert indices back to triples
    result = [all_triples[i] for i in best_triples]
    
    return result
