import time
import itertools
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_iso(p, a, b):
    dp_a = _dist2(p, a)
    dp_b = _dist2(p, b)
    da_b = _dist2(a, b)
    return dp_a == dp_b or dp_a == da_b or dp_b == da_b

def construct():
    start_time = time.time()
    deadline = start_time + 320
    
    points = list(itertools.product(range(N), repeat=2))
    n_pts = N * N
    
    # Precompute hyperedges: each hyperedge is a triple (a,b,c) that forms an isosceles triangle
    # We want to find minimum set S such that:
    # 1. S is independent (no triple in S is an isosceles triangle)
    # 2. S is maximal (every point outside S, when added, creates an isosceles triangle with two points in S)
    
    # For maximality: for each point q not in S, there must exist a,b in S such that (q,a,b) is isosceles
    # This is equivalent to: q is "dominated" by S in the hypergraph sense
    
    # Precompute for each point q, the set of pairs (a,b) such that (q,a,b) is isosceles
    # q is dominated by S if there exists (a,b) in this set with a,b in S
    
    # Also precompute for each point q, the set of points r such that adding q to S would create
    # an isosceles triangle with some pair in S. This is for independence check.
    
    # Let's precompute:
    # 1. iso_pairs[q]: list of (a,b) such that (q,a,b) is isosceles
    # 2. For independence: when we add p to S, we need to check that no pair (a,b) in S has (p,a,b) isosceles
    
    # Precompute iso_pairs
    iso_pairs = [[] for _ in range(n_pts)]
    for q_idx in range(n_pts):
        q = points[q_idx]
        for a_idx in range(n_pts):
            if a_idx == q_idx:
                continue
            a = points[a_idx]
            for b_idx in range(a_idx + 1, n_pts):
                if b_idx == q_idx:
                    continue
                b = points[b_idx]
                if _is_iso(q, a, b):
                    iso_pairs[q_idx].append((a_idx, b_idx))
    
    # For independence: precompute for each point p, the set of pairs (a,b) that would form
    # an isosceles triangle with p. This is the same as iso_pairs[p].
    # When adding p to S, we need: for all (a,b) in iso_pairs[p], not (a in S and b in S)
    
    # For maximality: when checking if S is maximal, for each q not in S, we need:
    # exists (a,b) in iso_pairs[q] such that a in S and b in S
    
    def is_independent_add(p_idx, S_mask):
        """Check if p_idx can be added to S_mask while maintaining independence."""
        for a, b in iso_pairs[p_idx]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        return True
    
    def is_maximal(S_mask):
        """Check if S_mask is maximal (no point can be added without breaking independence)."""
        for q_idx in range(n_pts):
            if S_mask & (1 << q_idx):
                continue
            # Check if q_idx can be added
            if is_independent_add(q_idx, S_mask):
                return False
        return True
    
    def is_dominated(q_idx, S_mask):
        """Check if q_idx is dominated by S_mask (adding it would break independence)."""
        for a, b in iso_pairs[q_idx]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return True
        return False
    
    def greedy_build(seed):
        """Greedy construction with random ordering."""
        random.seed(seed)
        order = list(range(n_pts))
        random.shuffle(order)
        
        S_mask = 0
        S_list = []
        
        for p_idx in order:
            if is_independent_add(p_idx, S_mask):
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)
                
        # Make maximal by adding points that can be added
        changed = True
        while changed:
            changed = False
            for p_idx in range(n_pts):
                if S_mask & (1 << p_idx):
                    continue
                if is_independent_add(p_idx, S_mask):
                    S_mask |= (1 << p_idx)
                    S_list.append(p_idx)
                    changed = True
        
        return S_mask, S_list
    
    def local_search_remove(S_mask, S_list):
        """Try to remove points while maintaining maximality."""
        current_mask = S_mask
        current_list = S_list[:]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_list)
            
            for p_idx in current_list:
                new_mask = current_mask & ~(1 << p_idx)
                # Check if new_mask is still maximal
                # A set is maximal if no point can be added
                # After removing p_idx, we need to check if any point q not in new_mask
                # can be added to new_mask
                # But we also need to check that p_idx itself cannot be added back
                # (which would mean it was necessary for maximality)
                
                # Actually, after removing p_idx, the set is maximal iff:
                # For every q not in new_mask, q cannot be added to new_mask
                # This includes q = p_idx itself
                
                # Quick check: can p_idx be added back?
                if is_independent_add(p_idx, new_mask):
                    # p_idx can be added back, so removing it breaks maximality
                    continue
                
                # Check if any other point can be added
                # This is expensive, so do a quick check first
                # Sample a few points to see if any can be added
                can_add_other = False
                for q_idx in range(n_pts):
                    if q_idx == p_idx:
                        continue
                    if new_mask & (1 << q_idx):
                        continue
                    if is_independent_add(q_idx, new_mask):
                        can_add_other = True
                        break
                
                if not can_add_other:
                    # new_mask is maximal
                    current_mask = new_mask
                    current_list.remove(p_idx)
                    improved = True
                    break
        
        return current_mask, current_list
    
    def lower_bound(S_mask, uncovered_points):
        """Lower bound on additional points needed."""
        if not uncovered_points:
            return 0
        
        # Simple lower bound: each point can cover at most K uncovered points
        # Count how many uncovered points each candidate point can cover
        max_cover = 0
        for p_idx in uncovered_points:
            cover_count = 0
            for q_idx in uncovered_points:
                if q_idx == p_idx:
                    continue
                # Check if adding p_idx would dominate q_idx
                # This is complex, so use a simple bound
                pass
            
            # Use a heuristic: each point can cover at most 10 uncovered points
            # This is a very loose bound
            max_cover = max(max_cover, 10)
        
        if max_cover == 0:
            return len(uncovered_points)
        
        return (len(uncovered_points) + max_cover - 1) // max_cover
    
    best_size = n_pts + 1
    best_set = None
    
    num_trials = 2000
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        S_mask, S_list = greedy_build(trial)
        
        # Local search: try to remove points
        S_mask, S_list = local_search_remove(S_mask, S_list)
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
    
    if best_set is None:
        S_mask, S_list = greedy_build(0)
        best_set = S_list
    
    # Ensure maximality
    changed = True
    while changed:
        changed = False
        for p_idx in range(n_pts):
            if p_idx in best_set:
                continue
            S_mask = 0
            for idx in best_set:
                S_mask |= (1 << idx)
            if is_independent_add(p_idx, S_mask):
                best_set.append(p_idx)
                changed = True
    
    return [points[idx] for idx in best_set]

# EVOLVE-BLOCK-END
