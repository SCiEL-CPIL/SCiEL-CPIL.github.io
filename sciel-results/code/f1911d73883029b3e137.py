"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START
def _is_isosceles(p1, p2, p3):
    """Check if three points form an isosceles triangle."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    d12 = (x1 - x2)**2 + (y1 - y2)**2
    d23 = (x2 - x3)**2 + (y2 - y3)**2
    d13 = (x1 - x3)**2 + (y1 - y3)**2
    return d12 == d23 or d12 == d13 or d23 == d13

def _count_violations_for_point(new_pt, existing_pts):
    """Count how many isosceles triangles new_pt forms with pairs in existing_pts."""
    count = 0
    n = len(existing_pts)
    for i in range(n):
        p1 = existing_pts[i]
        for j in range(i + 1, n):
            p2 = existing_pts[j]
            if _is_isosceles(new_pt, p1, p2):
                count += 1
    return count

def _greedy_init(seed=0):
    """Generate an initial set using greedy randomization."""
    rng = random.Random(seed)
    pts = []
    all_pts = [(x, y) for x in range(N) for y in range(N)]
    rng.shuffle(all_pts)
    
    for pt in all_pts:
        # Check if adding pt creates any isosceles triangle
        valid = True
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if _is_isosceles(pt, pts[i], pts[j]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            pts.append(pt)
    return pts

def _simulated_annealing(initial_pts, time_budget=60.0):
    """
    Improve the set using Simulated Annealing.
    Objective: Minimize E = violations - lambda * size
    We primarily want to maximize size while keeping violations at 0.
    """
    pts = list(initial_pts)
    pts_set = set(pts)
    start_time = time.time()
    
    # Current state
    current_size = len(pts)
    # Calculate current violations (should be 0 if from greedy, but let's be safe)
    current_violations = 0
    # We assume input is valid (0 violations).
    
    best_pts = list(pts)
    best_size = current_size
    
    # SA Parameters
    T = 5.0  # Initial temperature
    cooling_rate = 0.9995
    
    # Weight for size in the energy function
    # E = violations - LAMBDA * size
    # We want to prefer larger sets.
    LAMBDA = 10.0
    
    while (time.time() - start_time) < time_budget:
        if not pts:
            break
            
        # Pick a random move: add a random point or remove a random point
        if random.random() < 0.5:
            # Try to ADD a point
            candidate = (random.randint(0, N - 1), random.randint(0, N - 1))
            if candidate in pts_set:
                continue
            
            # Calculate delta violations
            delta_violations = _count_violations_for_point(candidate, pts)
            delta_size = 1
            delta_E = delta_violations - LAMBDA * delta_size
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                pts.append(candidate)
                pts_set.add(candidate)
                current_size += 1
                current_violations += delta_violations
                
                if current_violations == 0 and current_size > best_size:
                    best_size = current_size
                    best_pts = list(pts)
        else:
            # Try to REMOVE a point
            if not pts:
                continue
            idx = random.randint(0, len(pts) - 1)
            candidate = pts[idx]
            
            # Calculate how many violations this point is involved in
            # This is tricky to do exactly without recounting.
            # Approximation: Assume removing reduces violations by the number of pairs it was part of.
            # For simplicity and speed, we just remove it. 
            # Removing a point NEVER increases violations.
            # It might reduce size.
            # Delta E = delta_violations - LAMBDA * delta_size
            # delta_violations <= 0 (reduces or same)
            # delta_size = -1
            # delta_E = (neg or 0) + LAMBDA
            # Usually positive, so we only accept if T is high or if it breaks a deadlock.
            
            # To be precise, we can count violations involving candidate.
            # But that's O(k^2).
            # Let's just remove it.
            
            # Acceptance:
            # We accept removals more easily if we are stuck or if it allows future additions.
            # Simple rule: Always accept removal if it leads to 0 violations (it does) and we are cooling down?
            # Actually, standard SA:
            # We don't know exact delta_violations easily without recounting or precomputation.
            # Let's just accept removal with probability based on T.
            # Or, simpler: If we are stuck (no recent improvement), force removals.
            
            # Let's use a heuristic: Accept removal if random < 0.1 + T/10
            if random.random() < (0.05 + T / 20.0):
                pts.pop(idx)
                pts_set.remove(candidate)
                current_size -= 1
                # Violations likely decreased, but we don't track it exactly for E calculation 
                # because we don't have the exact count. 
                # However, since we only track best VALID sets, we can just re-validate occasionally 
                # or assume that if we started valid, and only added points that we checked, 
                # and removed points, we might have introduced "hidden" violations? 
                # No, removing points cannot introduce new isosceles triangles. 
                # It can only remove existing ones.
                # So if we started with 0 violations, and we only add points after checking them,
                # and we remove points, the set remains VALID (0 violations) always?
                # Wait.
                # If I add point P, I check if P forms isosceles with EXISTING pairs.
                # If it does, I reject (or accept with penalty).
                # If I accept a move that *increases* violations (delta_violations > 0), then my set is NO LONGER valid.
                # So `current_violations` can be > 0.
                # If I remove a point, violations decrease.
                
                # So we can track `current_violations` approximately?
                # No, we need exact count for the "best valid set" check.
                # If `current_violations > 0`, we cannot update `best_pts`.
                
                # To keep it simple and robust:
                # We only update best_pts if current_violations == 0.
                # But we don't know current_violations exactly if we accepted bad moves.
                # Solution: Only accept moves that keep violations at 0? That's just local search.
                # To escape local optima, we MUST allow violations.
                # Therefore, we MUST track violations accurately.
                
                # Tracking violations accurately is hard without precomputation.
                # Alternative: 
                # 1. Greedy/Local Search to get a valid set.
                # 2. SA that ONLY accepts moves that result in a VALID set? No, that's greedy.
                # 3. SA that accepts invalid sets, but we periodically "repair" or check validity.
                
                # Let's simplify: 
                # We will run a "Valid-Only" SA (which is effectively a sophisticated local search) 
                # BUT with a twist: We allow "Swaps" (remove one, add one) to escape local minima.
                # A swap is valid if the new set has 0 violations.
                
                # This is safer than tracking violations.
                
                pass # This branch was for removal. Let's implement Swap instead.
        
        # Cooling
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return best_pts

def _local_search_swap(pts, time_budget=60.0):
    """
    Local search using swaps.
    Try to swap a point in the set with a point outside the set.
    Accept if the new set is valid (0 isosceles triangles).
    """
    pts = list(pts)
    pts_set = set(pts)
    start_time = time.time()
    
    # Shuffle to randomize
    random.shuffle(pts)
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        for _ in range(50):
            if (time.time() - start_time) >= time_budget:
                break
            
            # Pick a random index to remove
            idx = random.randint(0, len(pts) - 1)
            old_pt = pts[idx]
            
            # Pick a random candidate to add
            candidate = (random.randint(0, N - 1), random.randint(0, N - 1))
            if candidate in pts_set:
                continue
            
            # Check if swap is valid
            # Remove old_pt, add candidate
            # We need to check if candidate forms isosceles with any pair in (pts - {old_pt})
            # AND we don't need to check if removing old_pt creates issues (it can't).
            
            remaining = [p for i, p in enumerate(pts) if i != idx]
            
            valid = True
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    if _is_isosceles(candidate, remaining[i], remaining[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                pts[idx] = candidate
                pts_set.remove(old_pt)
                pts_set.add(candidate)
                improved = True
                # Restart the loop to try more swaps from this new state
                break
                
    return pts

def construct():
    start = time.time()
    
    # 1. Generate multiple greedy initializations
    best_pts = []
    best_len = 0
    
    # Run greedy with different seeds
    for seed in range(10):
        if time.time() - start > TIME_LIMIT - 30:
            break
        pts = _greedy_init(seed)
        if len(pts) > best_len:
            best_pts = pts
            best_len = len(pts)
    
    # 2. Apply Local Search (Swap) to the best greedy result
    if time.time() - start < TIME_LIMIT - 10:
        time_remaining = TIME_LIMIT - 15 - (time.time() - start)
        if time_remaining > 0:
            improved_pts = _local_search_swap(best_pts, time_budget=time_remaining)
            if len(improved_pts) > best_len:
                best_pts = improved_pts
                best_len = len(improved_pts)
                
    # 3. Try a few more greedy starts if time permits, and improve them
    if time.time() - start < TIME_LIMIT - 20:
        for seed in range(10, 30):
            if time.time() - start > TIME_LIMIT - 10:
                break
            pts = _greedy_init(seed)
            if len(pts) > best_len:
                # Found a better start, improve it
                time_remaining = TIME_LIMIT - 15 - (time.time() - start)
                if time_remaining > 5:
                    improved_pts = _local_search_swap(pts, time_budget=min(time_remaining, 30.0))
                    if len(improved_pts) > best_len:
                        best_pts = improved_pts
                        best_len = len(improved_pts)
            else:
                # Even if not better initially, maybe local search can push it over?
                # Only if we have plenty of time.
                if time.time() - start < TIME_LIMIT - 30 and len(pts) > best_len - 2:
                     time_remaining = TIME_LIMIT - 20 - (time.time() - start)
                     if time_remaining > 5:
                        improved_pts = _local_search_swap(pts, time_budget=min(time_remaining, 10.0))
                        if len(improved_pts) > best_len:
                            best_pts = improved_pts
                            best_len = len(improved_pts)

    return best_pts
# EVOLVE-BLOCK-END
