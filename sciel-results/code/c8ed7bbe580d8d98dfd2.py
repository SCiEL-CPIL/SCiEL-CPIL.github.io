import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
    
    # Fast independence number calculation using branch and bound
    def calc_alpha(triples_list):
        """Calculate independence number using branch and bound with bitsets."""
        if not triples_list:
            return 16
        
        # Build adjacency: for each vertex, which triples contain it
        # And for each triple, which vertices it contains
        n = 16
        # vertex_to_triples: list of triple indices for each vertex
        v2t = [[] for _ in range(n)]
        for i, t in enumerate(triples_list):
            for v in t:
                v2t[v].append(i)
        
        # For fast checking: a set S is independent if no triple is subset of S
        # Use recursive search with pruning
        
        best = [0]
        
        def is_independent(vertices):
            """Check if vertices (tuple) is independent."""
            if len(vertices) <= best[0]:
                return False
            for t in triples_list:
                if all(v in vertices for v in t):
                    return False
            return True
        
        # Better: use bitset representation
        # Represent each triple as a bitmask
        triple_masks = []
        for t in triples_list:
            mask = 0
            for v in t:
                mask |= (1 << v)
            triple_masks.append(mask)
        
        # For a candidate independent set S (as bitmask), check if any triple is subset
        # triple_mask & S == triple_mask means triple is subset of S
        
        def check_independent(mask):
            for tm in triple_masks:
                if tm & mask == tm:
                    return False
            return True
        
        # Branch and bound
        # Order vertices by degree (number of triples containing them)
        deg = [len(v2t[v]) for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 5:
                return
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == n:
                return
            
            v = order[idx]
            
            # Option 1: don't include v
            search(idx + 1, current_mask, current_size)
            
            # Option 2: include v
            new_mask = current_mask | (1 << v)
            # Check if still independent
            valid = True
            for tm in triple_masks:
                if tm & new_mask == tm:
                    valid = False
                    break
            if valid:
                search(idx + 1, new_mask, current_size + 1)
        
        search(0, 0, 0)
        return best[0]
    
    # Even faster: use a different approach
    # For each candidate size k from 16 down to 1, check if there's an independent set of size k
    # Use combinatorial search with pruning
    
    def calc_alpha_fast(triples_list):
        if not triples_list:
            return 16
        
        n = 16
        triple_masks = []
        for t in triples_list:
            mask = 0
            for v in t:
                mask |= (1 << v)
            triple_masks.append(mask)
        
        # Precompute: for each subset of size k, is it independent?
        # Too expensive for large k. Use branch and bound.
        
        # Sort vertices by degree
        deg = [0] * n
        for tm in triple_masks:
            for v in range(n):
                if tm & (1 << v):
                    deg[v] += 1
        
        order = sorted(range(n), key=lambda v: -deg[v])
        
        best = [0]
        
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 5:
                return
            if current_size > best[0]:
                best[0] = current_size
            
            if idx == n:
                return
            
            v = order[idx]
            
            # Don't include v
            search(idx + 1, current_mask, current_size)
            
            # Include v
            new_mask = current_mask | (1 << v)
            valid = True
            for tm in triple_masks:
                if tm & new_mask == tm:
                    valid = False
                    break
            if valid:
                search(idx + 1, new_mask, current_size + 1)
        
        search(0, 0, 0)
        return best[0]
    
    # Even better: use a specialized approach
    # The independence number is the max k such that there exists a k-set with no triple contained in it
    # We can check for each k from high to low if an independent set of size k exists
    
    def has_independent_set(triples_list, k):
        """Check if there exists an independent set of size k."""
        if k <= 0:
            return True
        if k > 16:
            return False
        
        n = 16
        triple_masks = []
        for t in triples_list:
            mask = 0
            for v in t:
                mask |= (1 << v)
            triple_masks.append(mask)
        
        # For a set S to be independent, no triple can be a subset of S
        # Use backtracking
        
        # Precompute for each vertex, the triples containing it
        v2t_masks = [[] for _ in range(n)]
        for tm in triple_masks:
            for v in range(n):
                if tm & (1 << v):
                    v2t_masks[v].append(tm)
        
        found = [False]
        
        def backtrack(idx, current_mask, current_size):
            if found[0]:
                return
            if time.time() > deadline - 5:
                return
            if current_size == k:
                # Check if independent
                for tm in triple_masks:
                    if tm & current_mask == tm:
                        return
                found[0] = True
                return
            
            remaining = n - idx
            if current_size + remaining < k:
                return
            
            v = idx
            # Try including v
            new_mask = current_mask | (1 << v)
            # Prune: if adding v creates a triple subset, skip
            valid = True
            for tm in triple_masks:
                if tm & new_mask == tm:
                    valid = False
                    break
            if valid:
                backtrack(idx + 1, new_mask, current_size + 1)
                if found[0]:
                    return
            
            # Try excluding v
            backtrack(idx + 1, current_mask, current_size)
        
        backtrack(0, 0, 0)
        return found[0]
    
    # Use this to compute alpha
    def get_alpha(triples_list):
        if not triples_list:
            return 16
        for k in range(16, 0, -1):
            if has_independent_set(triples_list, k):
                return k
        return 0
    
    # Main construction
    # Start with a greedy construction
    used_pairs = set()
    current_triples = []
    
    # Greedy: add triples that don't violate linearity
    # To minimize alpha, we want to "hit" as many small sets as possible
    # A good heuristic: prefer triples that are "dense" in terms of covering pairs
    
    random.seed(42)
    triple_indices = list(range(n_triples))
    
    # Multiple restarts
    best_alpha = 17
    best_solution = None
    
    num_restarts = 5
    time_per_restart = (deadline - start_time - 10) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        used_pairs = set()
        current_triples = []
        
        # Greedy initialization with randomization
        triple_indices = list(range(n_triples))
        random.shuffle(triple_indices)
        
        for t_idx in triple_indices:
            t = all_triples[t_idx]
            pairs = triple_pairs[t_idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                for p in pairs:
                    used_pairs.add(p)
                current_triples.append(t_idx)
        
        # Local search: try to improve by swapping triples
        # Objective: minimize alpha
        # But alpha is expensive to compute, so use a proxy
        # Proxy: number of 5-sets covered (more covered = smaller alpha likely)
        
        # Actually, let's just do a few random swaps and check alpha occasionally
        
        alpha = get_alpha([all_triples[i] for i in current_triples])
        if alpha < best_alpha:
            best_alpha = alpha
            best_solution = list(current_triples)
        
        if best_alpha <= 4:
            break
        
        # Local search
        for step in range(200):
            if time.time() > deadline - 10:
                break
            
            # Random swap
            if not current_triples:
                break
            
            t_remove_idx = random.randint(0, len(current_triples) - 1)
            t_remove = current_triples[t_remove_idx]
            
            # Remove
            pairs_remove = triple_pairs[t_remove]
            for p in pairs_remove:
                used_pairs.discard(p)
            current_triples.pop(t_remove_idx)
            
            # Try to add a random triple
            t_add_idx = random.randint(0, n_triples - 1)
            t_add = all_triples[t_add_idx]
            pairs_add = triple_pairs[t_add_idx]
            valid = True
            for p in pairs_add:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                for p in pairs_add:
                    used_pairs.add(p)
                current_triples.append(t_add_idx)
                
                # Check alpha occasionally
                if step % 10 == 0:
                    alpha = get_alpha([all_triples[i] for i in current_triples])
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_solution = list(current_triples)
                        if best_alpha <= 4:
                            break
            else:
                # Restore
                for p in pairs_remove:
                    used_pairs.add(p)
                current_triples.insert(t_remove_idx, t_remove)
        
        if best_alpha <= 4:
            break
    
    if best_solution is None:
        # Fallback: return a simple linear system
        best_solution = [0]  # Just one triple
        best_solution = list(range(min(10, n_triples)))
    
    return [list(all_triples[i]) for i in best_solution]
