import time
import random
import numpy as np

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all 4-APs: for each x, store the set of (x1, x2, x3) such that
    # adding x would complete a 4-AP. We store for each x, a list of bitmasks where
    # if any of these 3 bits are all set in the current mask, x cannot be added.
    # Actually, let's precompute for each x, the list of triples (a,b,c) such that
    # {a,b,c,x} forms a 4-AP. Then can_add checks if any triple is fully in mask.
    
    # For efficiency, precompute for each x, a list of (bit1, bit2, bit3) positions
    ap_triples = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # 4-AP: a, b, c, e
            ap_triples[a].append((b, c, e))
            ap_triples[b].append((a, c, e))
            ap_triples[c].append((a, b, e))
            ap_triples[e].append((a, b, c))
    
    def can_add(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        for (a, b, c) in ap_triples[x]:
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

    def fast_greedy_numpy(rng=None):
        """Greedy using numpy for faster checks."""
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
        max_no_improve = 120
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try adding any blocked element
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
            for a in blocked_list[:30]:  # Try up to 30 random blocked elements
                if time.time() >= deadline:
                    return mask, S_size
                
                S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                rng.shuffle(S_list)
                
                # Try up to 100 random removals
                for r in S_list[:100]:
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
    
    # Multiple restarts with different random seeds
    for trial in range(30):
        if time.time() >= deadline - 15:
            break
        
        rng = random.Random(42 + trial)
        
        # Initial seed with randomization
        mask = fast_greedy(rng)
        
        # Run local search
        mask, size = local_search(mask, rng, deadline - 10)
        
        if size > best_size:
            best_size = size
            best_mask = mask
    
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
