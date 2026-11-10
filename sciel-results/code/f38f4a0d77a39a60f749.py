"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import sys

DEADLINE = time.time() + 290  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict bitsets:
# conflict[p][a] is a bitmask of all points b such that (p, a, b) forms an isosceles triangle
conflict = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for p in range(NUM_PTS):
    for a in range(NUM_PTS):
        if p == a:
            continue
        mask = 0
        for b in range(NUM_PTS):
            if b == p or b == a:
                continue
            d_pa = dist[p][a]
            d_pb = dist[p][b]
            d_ab = dist[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                mask |= (1 << b)
        conflict[p][a] = mask


def can_add(p, chosen_mask):
    """Check if point p can be added given chosen points represented as bitmask."""
    cm = chosen_mask
    while cm:
        lb = cm & (-cm)
        a = lb.bit_length() - 1
        if conflict[p][a] & chosen_mask:
            return False
        cm ^= lb
    return True


def greedy_construct(order, chosen_mask=0, chosen_list=None):
    """Greedy construction following given order of points."""
    if chosen_list is None:
        chosen_list = []
    for p in order:
        if not (chosen_mask >> p) & 1:
            if can_add(p, chosen_mask):
                chosen_list.append(p)
                chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def complete_to_maximal(chosen_list, chosen_mask, rng):
    """Add points until the set is maximal (no more can be added)."""
    candidates = [p for p in range(NUM_PTS) if not (chosen_mask >> p) & 1]
    rng.shuffle(candidates)
    for p in candidates:
        if can_add(p, chosen_mask):
            chosen_list.append(p)
            chosen_mask |= (1 << p)
    return chosen_list, chosen_mask


def randomized_greedy(seed):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    chosen, mask = greedy_construct(order)
    # Ensure maximality
    chosen, mask = complete_to_maximal(chosen, mask, rng)
    return chosen, mask


def local_search(chosen_list, chosen_mask, max_iterations=100):
    """Local search: try removing a point and refilling to maximality."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point in the current best set
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Complete to maximality with a fresh random order
            rng = random.Random(iteration * 10000 + i * 7 + 42)
            new_list, new_mask = complete_to_maximal(new_list, new_mask, rng)
            
            if len(new_list) < best_len:
                best = new_list
                best_mask = new_mask
                best_len = len(best)
                improved = True
                break  # Restart with new best
        
        if not improved:
            break
    
    return best, best_mask


def shrink_search(chosen_list, chosen_mask):
    """Try to shrink the set by removing points one at a time and re-completing."""
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Try multiple random completions to find the smallest maximal superset
            best_completion = None
            best_completion_mask = 0
            best_completion_len = float('inf')
            
            for attempt in range(5):
                rng = random.Random(attempt * 1000 + i + 12345)
                cl, cm = complete_to_maximal(list(new_list), new_mask, rng)
                if len(cl) < best_completion_len:
                    best_completion = cl
                    best_completion_mask = cm
                    best_completion_len = len(cl)
            
            if best_completion is not None and best_completion_len < best_len:
                best = best_completion
                best_mask = best_completion_mask
                best_len = best_completion_len
                changed = True
                break
    
    return best, best_mask


def branch_and_bound_minimal(chosen_list, chosen_mask, deadline):
    """
    Branch-and-bound to find a minimum-size maximal no-isosceles set.
    We start with a maximal set and try to remove points while maintaining maximality.
    Uses bitset-based conflict propagation for fast pruning.
    """
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    # Precompute: for each point p, the set of points that would be "exposed" if p is removed
    # i.e., points q not in chosen such that q can be added iff p is present (or some other point)
    
    def is_maximal(mask):
        """Check if the set represented by mask is maximal."""
        for p in range(NUM_PTS):
            if not (mask >> p) & 1:
                if can_add(p, mask):
                    return False
        return True
    
    # Try to remove each point and check if the set remains maximal
    # If not maximal, we must add back some points - use greedy to complete
    # If the completed set is smaller, we improved
    
    # Sort points by "removability" - points that are in fewer isosceles triples with other chosen points
    point_conflict_counts = []
    for p in best:
        count = 0
        cm = chosen_mask
        while cm:
            lb = cm & (-cm)
            a = lb.bit_length() - 1
            if a != p and (conflict[p][a] & chosen_mask):
                count += 1
            cm ^= lb
        point_conflict_counts.append((count, p))
    
    # Try removing points with highest conflict count first (they're more "essential")
    # Actually, try removing points with LOWEST conflict count first (less essential)
    point_conflict_counts.sort(key=lambda x: x[0])
    
    for _, p_to_remove in point_conflict_counts:
        if time.time() > deadline:
            break
        if len(best) <= best_len:
            continue
        
        new_mask = best_mask & ~(1 << p_to_remove)
        new_list = [p for p in best if p != p_to_remove]
        
        # Check if still maximal
        if is_maximal(new_mask):
            # Great, we can remove this point
            best = new_list
            best_mask = new_mask
            best_len = len(best)
        else:
            # Need to add back some points to restore maximality
            # Use greedy completion
            rng = random.Random(hash(p_to_remove) % 10000)
            new_list, new_mask = complete_to_maximal(new_list, new_mask, rng)
            if len(new_list) < best_len:
                best = new_list
                best_mask = new_mask
                best_len = len(best)
    
    return best, best_mask


def advanced_local_search(chosen_list, chosen_mask, deadline):
    """
    More aggressive local search using branch-and-bound style pruning.
    """
    best = list(chosen_list)
    best_mask = chosen_mask
    best_len = len(best)
    
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try all single removals
        for i in range(len(best)):
            if time.time() > deadline:
                break
            
            removed = best[i]
            new_mask = best_mask & ~(1 << removed)
            new_list = [p for j, p in enumerate(best) if j != i]
            
            # Try to complete with minimal additions
            best_completion = None
            best_completion_mask = 0
            best_completion_len = float('inf')
            
            for attempt in range(10):
                rng = random.Random(iteration * 100000 + i * 13 + attempt * 7 + 999)
                cl, cm = complete_to_maximal(list(new_list), new_mask, rng)
                if len(cl) < best_completion_len:
                    best_completion = cl
                    best_completion_mask = cm
                    best_completion_len = len(cl)
            
            if best_completion is not None and best_completion_len < best_len:
                best = best_completion
                best_mask = best_completion_mask
                best_len = best_completion_len
                improved = True
                break
        
        if not improved:
            # Try double removals
            found_double = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline:
                        break
                    
                    new_mask = best_mask & ~(1 << best[i]) & ~(1 << best[j])
                    new_list = [p for k, p in enumerate(best) if k != i and k != j]
                    
                    best_completion = None
                    best_completion_mask = 0
                    best_completion_len = float('inf')
                    
                    for attempt in range(5):
                        rng = random.Random(iteration * 100000 + i * 100 + j * 7 + attempt * 3 + 555)
                        cl, cm = complete_to_maximal(list(new_list), new_mask, rng)
                        if len(cl) < best_completion_len:
                            best_completion = cl
                            best_completion_mask = cm
                            best_completion_len = len(cl)
                    
                    if best_completion is not None and best_completion_len < best_len:
                        best = best_completion
                        best_mask = best_completion_mask
                        best_len = best_completion_len
                        found_double = True
                        break
                if found_double:
                    break
            
            if not found_double:
                break
    
    return best, best_mask


def construct():
    """Construct a maximal no-isosceles set with minimum size."""
    best = None
    best_mask = 0
    best_len = float('inf')
    
    # Baseline: deterministic row-major greedy
    baseline_order = list(range(NUM_PTS))
    baseline, baseline_mask = greedy_construct(baseline_order)
    baseline, baseline_mask = complete_to_maximal(baseline, baseline_mask, random.Random(0))
    best = baseline
    best_mask = baseline_mask
    best_len = len(best)
    
    # Randomized restarts with local search
    seed = 42
    while time.time() < DEADLINE - 30:
        chosen, mask = randomized_greedy(seed)
        if len(chosen) < best_len:
            best = chosen
            best_mask = mask
            best_len = len(best)
        
        # Local search on this solution
        if len(chosen) <= best_len:
            improved, improved_mask = local_search(chosen, mask, max_iterations=20)
            if len(improved) < best_len:
                best = improved
                best_mask = improved_mask
                best_len = len(best)
        
        seed += 1
    
    # Advanced local search on the best solution
    if time.time() < DEADLINE - 15:
        polished, polished_mask = advanced_local_search(best, best_mask, DEADLINE - 10)
        if len(polished) < best_len:
            best = polished
            best_mask = polished_mask
            best_len = len(best)
    
    # Final branch-and-bound polish
    if time.time() < DEADLINE - 5:
        final, final_mask = branch_and_bound_minimal(best, best_mask, DEADLINE - 2)
        if len(final) < best_len:
            best = final
            best_mask = final_mask
            best_len = len(best)
    
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
