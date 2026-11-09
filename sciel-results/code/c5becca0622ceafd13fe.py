import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_distances(all_points):
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix

def _can_add(x, chosen, dist_matrix):
    if not chosen:
        return True
        
    # 1. Distances from x to all p in chosen must be distinct
    distances_from_x = set()
    for p in chosen:
        d = dist_matrix[x][p]
        if d in distances_from_x:
            return False
        distances_from_x.add(d)
        
    # 2. For each p in chosen, d(x, p) must not be a distance already used by p to other points in chosen
    for p in chosen:
        d_xp = dist_matrix[x][p]
        for q in chosen:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    return True

def _is_maximal(chosen, all_indices, dist_matrix):
    chosen_set = set(chosen)
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen, dist_matrix):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, rng):
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen, dist_matrix):
            chosen.append(p)
    return chosen

def _local_search_reduce(chosen, all_indices, dist_matrix, rng, deadline):
    s = list(chosen)
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                p, q = s[i], s[j]
                base = [x for k, x in enumerate(s) if k != i and k != j]
                base_set = set(base)
                
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    if not _can_add(r, base, dist_matrix):
                        continue
                    
                    candidate = base + [r]
                    if _is_maximal(candidate, all_indices, dist_matrix):
                        s = candidate
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s

def _backtracking_search(target_size, all_indices, dist_matrix, rng, deadline):
    """
    Search for a maximal set of exactly target_size.
    Returns the set if found, None otherwise.
    """
    chosen = []
    
    # Sort candidates to improve pruning? 
    # Maybe points with fewer "conflicts" first? 
    # For now, shuffle to avoid bias.
    candidates = list(all_indices)
    rng.shuffle(candidates)
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        
        if len(chosen) == target_size:
            if _is_maximal(chosen, all_indices, dist_matrix):
                return list(chosen)
            return None
        
        # Heuristic: try to pick a point that is "compatible" with many others?
        # Or just iterate through remaining candidates.
        # To speed up, we can skip candidates that are already incompatible with chosen.
        
        for i in range(idx, len(candidates)):
            x = candidates[i]
            if x in chosen: # Should not happen if we manage indices, but safe
                continue
            
            if _can_add(x, chosen, dist_matrix):
                chosen.append(x)
                result = backtrack(i + 1)
                if result is not None:
                    return result
                chosen.pop()
                
        return None

    return backtrack(0)

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix = _precompute_distances(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Phase 1: Greedy + Local Search to get a good upper bound
    # Run a few restarts
    for _ in range(10):
        if time.time() > _DEADLINE - 50:
            break
        chosen = _construct_greedy(all_indices, dist_matrix, rng)
        pruned = _local_search_reduce(chosen, all_indices, dist_matrix, rng, _DEADLINE - 20)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
            
    # Phase 2: Backtracking to find smaller sets
    # We know best_size is an upper bound. Try to find smaller.
    # Start from small sizes and go up? Or start from best_size-1 and go down?
    # If we start from 1, it's very fast to fail.
    # If we start from best_size-1, we might find a solution quickly if it exists.
    
    # Let's try sizes from 1 to best_size - 1.
    # If we find a maximal set of size k, we can stop (since we want minimum).
    # But we should check smaller ones first?
    # If size 1 is maximal, return it.
    # If size 2 is maximal, return it.
    
    # Actually, if we find size k, we don't need to check k+1, k+2...
    # But we DO need to check 1, 2, ..., k-1.
    # So iterate k from 1 to best_size - 1.
    
    # Optimization: If best_size is small (e.g., 5), check 1..4.
    # If best_size is large (e.g., 20), checking 1..19 might take too long if 1..18 don't exist.
    # However, small sets are easier to verify non-existence? No, backtracking is exponential.
    
    # Let's try a few small k first, then if time permits, try larger k.
    # Or, use the fact that if k is not found, we increment k.
    
    # To be safe, we'll try k = best_size - 1, best_size - 2, etc., down to 1?
    # No, if we find size 10, we should check if size 9 exists.
    # If we find size 9, check 8.
    # So, we can do a loop:
    # current_limit = best_size
    # while current_limit > 1:
    #   try to find set of size current_limit - 1
    #   if found: best_size = current_limit - 1; best = found
    #   else: break? (If size k-1 doesn't exist, does k-2 exist? Yes, it's possible. 
    #           Wait, if no maximal set of size k-1 exists, can a maximal set of size k-2 exist?
    #           Yes. Maximal sets don't have to be "dense" in size.
    #           Example: No maximal set of size 5, but a maximal set of size 4 exists.
    #           So we cannot break early on failure.)
    
    # So we must check all k < best_size.
    # But checking k=1 is fast. k=2 is fast. k=5 might be slow.
    
    # Let's iterate k from 1 to best_size - 1.
    # If we find one, update best.
    # If we run out of time, return best.
    
    # To improve performance of backtracking, we can use a better ordering.
    # Points that are "isolated" (fewer distances to others?) might be better candidates?
    
    # Let's just run the backtracking for k = best_size - 1 down to 1?
    # Or 1 up to best_size - 1?
    # If we start from 1, we might spend too much time on k=1,2,3 if they don't exist?
    # Actually, non-existence of small k is usually fast to prove?
    # No, proving non-existence is harder than finding existence.
    # But for k=1, we just check if any single point is maximal. (Is any point maximal? No, because you can always add another point unless the grid is 1x1? 
    # In 16x16, a single point is NEVER maximal because you can always add a second point at a different distance?
    # Wait. If S={p}, can we add q?
    # Condition 1: Distances from q to p must be distinct. (Only one distance, so yes).
    # Condition 2: For p, d(q,p) must not be in {d(p,r) for r in S\{p}}. S\{p} is empty. So yes.
    # So any pair {p,q} is a valid no-isosceles set.
    # Is {p,q} maximal? No, we can add r.
    # So k=1 is never maximal for N>1.
    # k=2? Is any pair maximal? 
    # If we have {p,q}, can we add r?
    # We need d(r,p) != d(r,q) and d(r,p) != d(p,q) and d(r,q) != d(p,q).
    # Usually yes. So k=2 is likely never maximal.
    # The minimum maximal size is likely around 4-6.
    
    # So we can start checking from k=3 or 4 to save time?
    # Let's be safe and start from 1, but it will be very fast to reject 1 and 2.
    
    # Let's modify backtracking to be faster.
    # We can pre-filter candidates that are incompatible with the first chosen point?
    
    # Let's try the loop.
    
    for k in range(1, best_size):
        if time.time() > _DEADLINE - 10:
            break
        
        # Use a fresh RNG or deterministic order?
        # Deterministic order might be faster for finding solutions if they exist?
        # Or random to avoid getting stuck?
        # Let's use a fixed order for k=1,2,3 to be fast, then random for larger?
        
        # Actually, let's just use the backtracking function.
        # To speed it up, we can limit the depth or use a heuristic.
        
        found = _backtracking_search(k, all_indices, dist_matrix, rng, _DEADLINE - 5)
        
        if found is not None:
            best_size = k
            best = found
            # Since we are iterating k upwards, if we find a solution for k, 
            # we should check k-1? No, we started from 1.
            # So if we find k=4, we have already checked 1,2,3 and they failed.
            # So 4 is the minimum. We can break?
            # Yes! If we find a maximal set of size k, and we have checked all 1..k-1 and failed,
            # then k is the minimum.
            break
            
    if best is None:
        best = _construct_greedy(all_indices, dist_matrix, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
