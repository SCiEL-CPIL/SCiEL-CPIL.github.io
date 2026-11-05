import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Map each triple to its index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute bitmask for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    # Precompute: for each point, which triples contain it
    point_to_triples = [[] for _ in range(16)]
    for i, t in enumerate(all_triples):
        for p in t:
            point_to_triples[p].append(i)
    
    # Precompute conflict: two triples conflict if they share 2+ points
    # For each triple, store set of conflicting triple indices
    conflicts = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        for j in range(i + 1, n_triples):
            common = triple_masks[i] & triple_masks[j]
            if bin(common).count('1') >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    def compute_alpha(triple_indices):
        """Compute the independence number given a set of triple indices."""
        if not triple_indices:
            return 16
        
        # Create a set for fast lookup
        tset = set(triple_indices)
        
        # For each subset of points, check if it contains no triple
        # We want the largest such subset
        # Try from largest to smallest
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                # Check if any triple is fully contained in this subset
                has_triple = False
                for ti in tset:
                    if (triple_masks[ti] & smask) == triple_masks[ti]:
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0
    
    def compute_alpha_fast(triple_indices):
        """Faster alpha computation using precomputed point-to-triples."""
        if not triple_indices:
            return 16
        
        tset = frozenset(triple_indices)
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                has_triple = False
                for ti in tset:
                    if (triple_masks[ti] & smask) == triple_masks[ti]:
                        has_triple = True
                        break
                if not has_triple:
                    return k
        return 0
    
    def find_largest_independent_set(triple_indices):
        """Find the actual largest independent set."""
        if not triple_indices:
            return 16, set(points)
        
        tset = set(triple_indices)
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                has_triple = False
                for ti in tset:
                    if (triple_masks[ti] & smask) == triple_masks[ti]:
                        has_triple = True
                        break
                if not has_triple:
                    return k, set(subset)
        return 0, set()
    
    best_alpha = 16
    best_triples = []
    
    # Strategy: Use a greedy approach that builds the system by adding triples
    # that cover as many "dangerous" subsets as possible
    
    # Alternative: Start from a random partial STS and optimize
    
    # Let's use a different approach: 
    # 1. Generate a random linear triple system
    # 2. Use local search to minimize alpha
    
    def generate_random_system():
        """Generate a random linear triple system."""
        available = list(range(n_triples))
        random.shuffle(available)
        
        selected = []
        used_points = [0] * 16  # Count of times each point is used
        
        for idx in available:
            t = all_triples[idx]
            # Check if this triple conflicts with any selected triple
            can_add = True
            for sel_idx in selected:
                if idx in conflicts[sel_idx]:
                    can_add = False
                    break
            
            if can_add:
                selected.append(idx)
        
        return selected
    
    def local_search(triple_indices, max_iterations=500):
        """Local search to minimize alpha."""
        current = list(triple_indices)
        current_set = set(current)
        current_alpha = compute_alpha(current)
        
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            
            if time.time() > deadline - 5:
                break
            
            # Try removing one triple
            for i in range(len(current)):
                new_set = current_set - {current[i]}
                new_alpha = compute_alpha(list(new_set))
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
                        if idx in conflicts[sel]:
                            can_add = False
                            break
                    if can_add:
                        candidates.append(idx)
                
                random.shuffle(candidates)
                for idx in candidates[:20]:  # Limit to 20 random candidates
                    new_set = current_set | {idx}
                    new_alpha = compute_alpha(list(new_set))
                    if new_alpha < current_alpha:
                        current = list(new_set)
                        current_set = new_set
                        current_alpha = new_alpha
                        improved = True
                        break
        
        return current, current_alpha
    
    # Phase 1: Multiple random restarts with local search
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate a random system
        system = generate_random_system()
        
        # Local search
        optimized, alpha = local_search(system, max_iterations=100)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = optimized[:]
            
            if best_alpha <= 7:
                break
    
    # Phase 2: If we haven't done well, try a more systematic approach
    if best_alpha > 8 and time.time() < deadline - 30:
        # Try to build a system that covers many small subsets
        # Use a greedy approach: pick triples that hit the most "uncovered" subsets
        
        # Start with a good initial guess
        # Known good configurations often use symmetric structures
        
        # Try a specific construction: partition points into groups
        # and create triples that cross groups
        
        # Let's try a different heuristic: 
        # For each point, ensure it's in enough triples to break up independent sets
        
        # Greedy: at each step, pick the triple that covers the most "dangerous" points
        # where dangerous means points that are in many potential independent sets
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Start empty and greedily add triples
            current = []
            current_set = set()
            
            # Score each triple by how many potential independent sets it hits
            # Approximation: prefer triples that use points not yet heavily covered
            
            point_usage = [0] * 16
            
            for _ in range(20):  # Add up to 20 triples
                if time.time() > deadline - 5:
                    break
                
                best_idx = None
                best_score = -1
                
                # Try a random subset of triples for speed
                sample_size = min(200, n_triples)
                sample = random.sample(range(n_triples), sample_size)
                
                for idx in sample:
                    if idx in current_set:
                        continue
                    
                    # Check conflicts
                    can_add = True
                    for sel in current_set:
                        if idx in conflicts[sel]:
                            can_add = False
                            break
                    
                    if not can_add:
                        continue
                    
                    # Score: prefer triples that use less-used points
                    t = all_triples[idx]
                    score = sum(1 - point_usage[p] / 5 for p in t)
                    
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
            alpha = compute_alpha(current)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = current[:]
                
                if best_alpha <= 7:
                    break
    
    # Convert indices back to triples
    result = [all_triples[i] for i in best_triples]
    
    return result
