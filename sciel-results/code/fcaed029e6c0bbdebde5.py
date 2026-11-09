import time
import random

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all 4-APs and for each element, which APs it participates in
    # ap_list[i] = list of (j, k, l) such that (i, j, k, l) is a 4-AP
    # For element x, we need to check all APs containing x
    
    # Build AP incidence: for each x, list of (a, b, c) where {x, a, b, c} forms a 4-AP
    # and a, b, c are the other three elements
    ap_incidence = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > N:
                break
            # (a, b, c, e) is a 4-AP
            ap_incidence[a].append((b, c, e))
            ap_incidence[b].append((a, c, e))
            ap_incidence[c].append((a, b, e))
            ap_incidence[e].append((a, b, c))
    
    def can_add(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        for (a, b, c) in ap_incidence[x]:
            # Check if all three other elements are in mask
            if (mask >> a & 1) and (mask >> b & 1) and (mask >> c & 1):
                return False
        return True

    def fast_greedy(rng=None):
        """Fast greedy construction with optional randomization."""
        mask = 0
        if rng is None:
            order = list(range(1, N + 1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        
        return mask
    
    def greedy_with_priority(rng=None):
        """Greedy that tries to add elements in an order that maximizes future flexibility."""
        mask = 0
        if rng is None:
            order = list(range(1, N + 1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        
        return mask

    def polish_pass(mask, deadline):
        """Final polish: try adding elements in multiple orders."""
        S_size = bin(mask).count('1')
        blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
        
        # Try natural order
        for x in blocked:
            if time.time() >= deadline:
                break
            if can_add(mask, x):
                mask |= (1 << x)
                S_size += 1
        
        # Try reverse order
        for x in reversed(blocked):
            if time.time() >= deadline:
                break
            if not (mask >> x & 1) and can_add(mask, x):
                mask |= (1 << x)
                S_size += 1
        
        # Try a few more random passes
        rng = random.Random(int(time.time() * 1000) % 10000)
        for _ in range(5):
            blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
            if not blocked:
                break
            rng.shuffle(blocked)
            for x in blocked:
                if time.time() >= deadline:
                    break
                if can_add(mask, x):
                    mask |= (1 << x)
                    S_size += 1
        
        return mask, S_size

    def local_search(mask, rng, deadline):
        """Local search with add and swap moves."""
        S_size = bin(mask).count('1')
        blocked = set(x for x in range(1, N + 1) if not (mask >> x & 1))
        
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try adding any blocked element
            if blocked:
                blocked_list = list(blocked)
                rng.shuffle(blocked_list)
                for x in blocked_list:
                    if time.time() >= deadline:
                        return mask, S_size
                    if can_add(mask, x):
                        mask |= (1 << x)
                        S_size += 1
                        blocked.discard(x)
                        improved = True
                        break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try swaps: remove r, add a
            if not blocked:
                break
                
            # Try multiple random blocked elements
            blocked_list = list(blocked)
            rng.shuffle(blocked_list)
            
            swap_found = False
            # Try more swap candidates
            for a in blocked_list[:50]:  
                if time.time() >= deadline:
                    return mask, S_size
                
                S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                rng.shuffle(S_list)
                
                # Try up to 150 random removals
                for r in S_list[:150]:
                    if time.time() >= deadline:
                        return mask, S_size
                    
                    mask_without_r = mask & ~(1 << r)
                    
                    if can_add(mask_without_r, a):
                        # Perform swap
                        mask = mask_without_r | (1 << a)
                        blocked.discard(a)
                        blocked.add(r)
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return mask, S_size

    # Main
    best_mask = 0
    best_size = 0
    
    # Use ILS: Start with greedy, then iterate perturb+search
    # This is often better than independent restarts for structured problems
    
    rng = random.Random(42)
    
    # Try multiple initial greedy starts with different random seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        if time.time() >= deadline - 15:
            break
        r = random.Random(seed)
        current_mask = fast_greedy(r)
        current_mask, current_size = local_search(current_mask, r, deadline - 15)
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
    
    # ILS Loop
    # We will perturb the current best (or current state) and search again.
    # To avoid getting stuck in one basin, we occasionally restart from scratch if stuck.
    
    iterations = 0
    while time.time() < deadline - 10:
        iterations += 1
        
        # Perturbation: Remove 3-6 random elements to break the local optimum structure
        # This creates "holes" that might allow different elements to be added
        perturb_mask = best_mask
        S_list = [x for x in range(1, N + 1) if (perturb_mask >> x & 1)]
        
        # Randomly remove k elements
        k = 3 + rng.randint(0, 3) # 3 to 6 elements
        if len(S_list) > k:
            to_remove = rng.sample(S_list, k)
            for r in to_remove:
                perturb_mask &= ~(1 << r)
        
        # Run local search from this perturbed state
        # Use a tighter deadline for this iteration to allow more iterations
        iter_deadline = min(deadline - 10, time.time() + 4.0) # 4s per iteration max
        new_mask, new_size = local_search(perturb_mask, rng, iter_deadline)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
        else:
            # If we didn't improve, we might want to restart from scratch occasionally
            # to avoid exploring a bad basin for too long.
            if iterations % 7 == 0:
                # Full restart with a fresh seed
                fresh_seed = int(time.time() * 10000) % 100000
                r = random.Random(fresh_seed)
                current_mask = fast_greedy(r)
                current_mask, current_size = local_search(current_mask, r, deadline - 10)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

    # Final polish on the best solution
    if best_size > 0 and time.time() < deadline - 2:
        best_mask, best_size = polish_pass(best_mask, deadline - 1)
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
