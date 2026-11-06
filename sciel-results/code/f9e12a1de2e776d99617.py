"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    """
    Find a large subset S of [1,256] such that for all a,b in S with a<=b,
    a+b is not a perfect square.
    
    Uses greedy construction with sophisticated local search (add/swap/perturbation) to maximize size.
    """
    deadline = time.time() + 85.0  # 5s safety margin before 90s limit
    
    N = 256
    
    # Precompute conflict graph: conflict[a] = set of b such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for k in range(2, 23):  # k^2 from 4 to 529
            b = k * k - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Nodes where 2*a is a perfect square (diagonal constraint) - cannot be in solution
    forbidden = set()
    for a in range(1, N + 1):
        if a in conflict[a]:
            forbidden.add(a)
    
    valid_nodes = [x for x in range(1, N + 1) if x not in forbidden]
    
    def can_add(elem, S):
        """Check if elem can be added to S without violating constraints."""
        for s in S:
            if elem in conflict[s]:
                return False
        return True
    
    def greedy_random(restart_seed):
        """Greedy construction with random ordering."""
        rng = random.Random(restart_seed)
        candidates = valid_nodes[:]
        rng.shuffle(candidates)
        
        S = set()
        for elem in candidates:
            if can_add(elem, S):
                S.add(elem)
        return S
    
    def greedy_min_degree():
        """Greedy construction preferring elements with fewest conflicts among remaining."""
        S = set()
        remaining = set(valid_nodes)
        
        while remaining:
            best_elem = None
            best_block = float('inf')
            
            for elem in remaining:
                if not can_add(elem, S):
                    continue
                # Count how many remaining elements this would block
                block_count = sum(1 for r in remaining if r != elem and r in conflict[elem])
                if block_count < best_block:
                    best_block = block_count
                    best_elem = elem
            
            if best_elem is None:
                break
            
            S.add(best_elem)
            remaining.discard(best_elem)
            # Remove elements that conflict with best_elem
            to_remove = {r for r in remaining if r in conflict[best_elem]}
            remaining -= to_remove
        
        return S
    
    def local_search(S, max_iter=500):
        """Local search with add, swap, and perturbation moves."""
        S = set(S)
        excluded = set(valid_nodes) - S
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            
            improved = False
            
            # Phase 1: Try to add excluded elements that have no blockers
            for elem in list(excluded):
                if time.time() >= deadline:
                    break
                if can_add(elem, S):
                    S.add(elem)
                    excluded.discard(elem)
                    improved = True
            
            # Phase 2: Try swaps - remove one element from S, add one from excluded
            # Only beneficial if the added element has fewer blockers than the removed one had
            for elem in list(excluded):
                if time.time() >= deadline:
                    break
                
                blockers = {s for s in S if elem in conflict[s]}
                if len(blockers) > 2:
                    continue
                
                # For each blocker, check if removing it allows adding elem
                # and doesn't expose too many new conflicts
                for blocker in blockers:
                    # Check: if we remove blocker and add elem, does it enable adding more?
                    new_S = S - {blocker} | {elem}
                    new_excluded = excluded - {elem} | {blocker}
                    
                    # Count how many new elements can be added
                    addable = 0
                    for e in new_excluded:
                        if can_add(e, new_S):
                            addable += 1
                            if addable >= len(blockers):
                                break
                    
                    # Accept swap if it enables adding at least as many as we removed
                    if addable >= len(blockers):
                        S = new_S
                        excluded = new_excluded
                        improved = True
                        break
            
            # Phase 3: Perturbation - randomly remove a few elements and try to re-add
            if not improved and len(excluded) > 0:
                if random.random() < 0.3:  # 30% chance to perturb
                    # Remove 1-3 random elements from S
                    to_remove = random.sample(list(S), min(random.randint(1, 3), len(S)))
                    for elem in to_remove:
                        S.discard(elem)
                        excluded.add(elem)
                    improved = True  # Force another iteration
            
            if not improved:
                break
        
        return S
    
    # Main search loop
    random.seed(42)
    
    # Start with best of several greedy constructions
    best_S = greedy_min_degree()
    best_size = len(best_S)
    
    # Try random greedy restarts
    for trial in range(200):
        if time.time() >= deadline:
            break
        
        S = greedy_random(trial * 7 + 1)
        
        # Apply local search
        S = local_search(S, max_iter=100)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Final intensive local search on best solution
    best_S = local_search(best_S, max_iter=2000)
    
    # Verify result
    S_list = sorted(best_S)
    for i, a in enumerate(S_list):
        for b in S_list[i:]:
            s = a + b
            r = int(math.isqrt(s))
            if r * r == s:
                best_S.discard(b)
                break
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
