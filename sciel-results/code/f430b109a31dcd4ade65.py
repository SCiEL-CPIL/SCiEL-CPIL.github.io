"""Combined approach: diverse greedy construction + enhanced local search with swaps.
Builds maximal no-isosceles sets via multiple heuristics, then refines via
removal, swap, and simulated-annealing local search to minimize set size."""

import time
import random
import itertools

N = 16
DEADLINE_SECONDS = 295  # safety margin before 330s kill

def _sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _is_iso_with_set(p, S):
    """Check if adding p to S creates any isosceles triangle with two points in S."""
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a = S[i]
            b = S[j]
            d_pa = _sqdist(p, a)
            d_pb = _sqdist(p, b)
            d_ab = _sqdist(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _can_add(p, S):
    """True if p can be added to S without creating an isosceles triangle."""
    return not _is_iso_with_set(p, S)

def _build_minimal_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    all_points = list(itertools.product(range(N), repeat=2))
    S = []
    addable = set(all_points)
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p in candidates:
            if not _can_add(p, S):
                continue
            
            new_S = S + [p]
            block_count = 0
            for q in addable:
                if q == p:
                    continue
                if not _can_add(q, new_S):
                    block_count += 1
            
            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        addable.discard(best_p)
        
        new_addable = set()
        for q in addable:
            if _can_add(q, S):
                new_addable.add(q)
        addable = new_addable
    
    return S

def _build_greedy_order(order):
    """Simple greedy in a given order."""
    S = []
    for p in order:
        if _can_add(p, S):
            S.append(p)
    return S

def _build_conflict_greedy(order_seed=0):
    """
    Greedy that at each step picks the addable point with minimum 'conflict score':
    how many pairs in S it would form isosceles with if we considered it as the apex.
    Actually, since we only add if no isosceles is formed, this is equivalent to
    picking points that are 'least constrained' by the current set.
    """
    rng = random.Random(order_seed)
    all_points = list(itertools.product(range(N), repeat=2))
    S = []
    addable = set(all_points)
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p in candidates:
            if not _can_add(p, S):
                continue
            
            # Score: how many other addable points would be blocked
            new_S = S + [p]
            block_count = 0
            for q in addable:
                if q == p:
                    continue
                if not _can_add(q, new_S):
                    block_count += 1
            
            # Also consider distance to existing points (prefer points far from S)
            min_dist = float('inf')
            for s in S:
                d = _sqdist(p, s)
                if d < min_dist:
                    min_dist = d
            
            score = block_count * 10 + min_dist * 0.01 + rng.random() * 0.001
            if score < best_score:
                best_score = score
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        addable.discard(best_p)
        
        new_addable = set()
        for q in addable:
            if _can_add(q, S):
                new_addable.add(q)
        addable = new_addable
    
    return S

def _compute_addable(S, all_points):
    """Compute set of points that can be added to S."""
    S_set = set(S)
    addable = set()
    for p in all_points:
        if p not in S_set and _can_add(p, S):
            addable.add(p)
    return addable

def _local_search_enhanced(S, all_points, max_iterations=300, time_limit=60, rng=None):
    """
    Enhanced local search with:
    1. Point removal (if set remains maximal)
    2. Point swap (remove one, add one that was blocked)
    3. Simulated annealing acceptance for non-improving moves
    """
    if rng is None:
        rng = random.Random()
    
    S = list(S)
    start_time = time.time()
    best_size = len(S)
    best_S = list(S)
    
    addable = _compute_addable(S, all_points)
    
    # Track temperature for simulated annealing
    temperature = 5.0
    cooling = 0.995
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Strategy 1: Try removing each point
        for idx in range(len(S)):
            if time.time() - start_time > time_limit:
                break
            
            removed = S[idx]
            S_trial = S[:idx] + S[idx+1:]
            S_trial_set = set(S_trial)
            
            # Check if S_trial is still maximal
            still_maximal = True
            for p in all_points:
                if p not in S_trial_set:
                    if _can_add(p, S_trial):
                        still_maximal = False
                        break
            
            if still_maximal:
                # We found a smaller maximal set!
                S = S_trial
                best_size = len(S)
                best_S = list(S)
                addable = _compute_addable(S, all_points)
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 2: Try swaps - remove one point, add a different one
        # This maintains size but may enable further reductions
        if len(S) > 0 and addable:
            swap_found = False
            for idx in range(len(S)):
                if time.time() - start_time > time_limit:
                    break
                
                removed = S[idx]
                S_trial = S[:idx] + S[idx+1:]
                
                # Try adding each point from addable (or all points)
                for p in list(addable):
                    if _can_add(p, S_trial):
                        # Swap: remove S[idx], add p
                        S_swap = S_trial + [p]
                        # Check if this enables a smaller maximal set
                        # For now, just accept if it doesn't increase size
                        # and hope subsequent removals help
                        S_trial_set = set(S_swap)
                        
                        # Quick check: is S_swap maximal?
                        is_maximal = True
                        for q in all_points:
                            if q not in S_trial_set:
                                if _can_add(q, S_swap):
                                    is_maximal = False
                                    break
                        
                        if is_maximal and len(S_swap) == len(S):
                            # Same size but different configuration
                            # Accept with some probability (simulated annealing)
                            if rng.random() < 0.1:
                                S = S_swap
                                addable = _compute_addable(S, all_points)
                                swap_found = True
                                break
                
                if swap_found:
                    break
        
        # Strategy 3: Random perturbation - remove a random point, rebuild
        if rng.random() < 0.2:
            if len(S) > 0:
                idx = rng.randrange(len(S))
                S_trial = S[:idx] + S[idx+1:]
                
                # Greedily rebuild to maximal
                S_rebuilt = list(S_trial)
                addable_rebuilt = _compute_addable(S_rebuilt, all_points)
                add_list = list(addable_rebuilt)
                rng.shuffle(add_list)
                
                for p in add_list:
                    if _can_add(p, S_rebuilt):
                        S_rebuilt.append(p)
                        addable_rebuilt = _compute_addable(S_rebuilt, all_points)
                        if not addable_rebuilt:
                            break
                
                if len(S_rebuilt) < best_size:
                    S = S_rebuilt
                    best_size = len(S)
                    best_S = list(S)
                    addable = _compute_addable(S, all_points)
                    improved = True
                elif len(S_rebuilt) == len(S):
                    # Accept with annealing probability
                    delta = 0  # same size
                    if rng.random() < 0.05:
                        S = S_rebuilt
                        addable = _compute_addable(S, all_points)
        
        temperature *= cooling
    
    return best_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    all_points = list(itertools.product(range(N), repeat=2))
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return DEADLINE_SECONDS - elapsed()
    
    # Phase 1: Quick baseline with row-major greedy
    row_major_order = list(itertools.product(range(N), repeat=2))
    S1 = _build_greedy_order(row_major_order)
    best_set = S1
    best_size = len(S1)
    
    if elapsed() > DEADLINE_SECONDS:
        return best_set
    
    # Phase 2: Multiple construction heuristics with local search
    # Priority: spend more time on local search for promising solutions
    
    constructions = []
    
    # Minimum-degree greedy with various seeds
    for seed in range(30):
        if time_remaining() < 30:
            break
        S = _build_minimal_greedy(order_seed=seed)
        constructions.append(S)
    
    # Conflict-based greedy
    for seed in range(15):
        if time_remaining() < 30:
            break
        S = _build_conflict_greedy(order_seed=seed)
        constructions.append(S)
    
    # Column-major order
    if time_remaining() > 30:
        col_major_order = list(itertools.product(range(N), repeat=2))
        col_major_order.sort(key=lambda p: (p[1], p[0]))
        S = _build_greedy_order(col_major_order)
        constructions.append(S)
    
    # Diagonal ordering
    if time_remaining() > 30:
        diag_order = sorted(itertools.product(range(N), repeat=2), 
                          key=lambda p: (abs(p[0]-p[1]), p[0], p[1]))
        S = _build_greedy_order(diag_order)
        constructions.append(S)
    
    # Random orderings
    rng_global = random.Random(42)
    for _ in range(15):
        if time_remaining() < 30:
            break
        random_order = list(itertools.product(range(N), repeat=2))
        rng_global.shuffle(random_order)
        S = _build_greedy_order(random_order)
        constructions.append(S)
    
    # Phase 3: Apply local search to each construction, prioritizing smaller ones
    # Sort by size to try smaller ones first
    constructions.sort(key=lambda x: len(x))
    
    # Deduplicate
    seen = set()
    unique_constructions = []
    for S in constructions:
        key = tuple(sorted(S))
        if key not in seen:
            seen.add(key)
            unique_constructions.append(S)
    
    # Apply local search with time allocation
    # Allocate more time to smaller initial solutions
    total_time_for_search = max(0, time_remaining() - 10)
    
    # Weight by inverse size (smaller solutions get more search time)
    weights = [1.0 / max(1, len(S)) for S in unique_constructions]
    total_weight = sum(weights)
    
    for i, S in enumerate(unique_constructions):
        if time_remaining() < 10:
            break
        
        # Allocate time proportional to weight
        allocated_time = (weights[i] / total_weight) * total_time_for_search
        allocated_time = max(5, min(allocated_time, 60))  # clamp between 5 and 60s
        
        if elapsed() + allocated_time > DEADLINE_SECONDS - 5:
            allocated_time = max(5, DEADLINE_SECONDS - 5 - elapsed())
        
        rng_local = random.Random(i * 1000 + 7)
        S_refined = _local_search_enhanced(
            S, all_points, 
            max_iterations=200, 
            time_limit=allocated_time,
            rng=rng_local
        )
        
        if len(S_refined) < best_size:
            best_size = len(S_refined)
            best_set = S_refined
    
    # Phase 4: If we have time, do a few more focused searches on the best solution
    if time_remaining() > 20 and best_set is not None:
        # Try to improve the best solution further with longer local search
        remaining = time_remaining() - 5
        if remaining > 10:
            rng_final = random.Random(999)
            S_final = _local_search_enhanced(
                best_set, all_points,
                max_iterations=500,
                time_limit=min(remaining, 120),
                rng=rng_final
            )
            if len(S_final) < best_size:
                best_size = len(S_final)
                best_set = S_final
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
