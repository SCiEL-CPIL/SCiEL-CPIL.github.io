import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safe margin for 60s limit

    # 1. Precompute Conflict Masks
    # masks[x] is a list of integers. Each integer is a bitmask of 3 elements.
    # If (S & mask) == mask, then x CANNOT be added to S.
    masks = [[] for _ in range(N)]
    
    # Generate all 4-APs
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            # 1-indexed: a, a+d, a+2d, a+3d
            # 0-indexed:
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            # For i0, the other 3 are i1, i2, i3
            masks[i0].append((1 << i1) | (1 << i2) | (1 << i3))
            # For i1, the other 3 are i0, i2, i3
            masks[i1].append((1 << i0) | (1 << i2) | (1 << i3))
            # For i2, the other 3 are i0, i1, i3
            masks[i2].append((1 << i0) | (1 << i1) | (1 << i3))
            # For i3, the other 3 are i0, i1, i2
            masks[i3].append((1 << i0) | (1 << i1) | (1 << i2))

    def can_add(x, S):
        """
        Check if x (0-indexed) can be added to S (integer bitmask).
        """
        for m in masks[x]:
            if (S & m) == m:
                return False
        return True

    def get_size(S):
        return bin(S).count("1")

    def greedy_construct(seed):
        rng = random.Random(seed)
        S = 0
        elements = list(range(N))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S):
                S |= (1 << x)
        return S

    def local_search(S, time_budget):
        """
        Systematic swap local search.
        Tries to swap one element out and one in.
        """
        current_size = get_size(S)
        best_size = current_size
        best_S = S
        no_improve_count = 0
        max_no_improve = 5 # Reset or stop if stuck? 
                           # Actually, if we do a full pass with no improvement, we are at a local optimum.
                           # We should break the outer loop if a full pass yields no improvement.
        
        while time.time() - start_time < time_budget:
            improved = False
            # To avoid getting stuck in a cycle or slow convergence, 
            # we can randomize the order of 'u' candidates or 'v' candidates slightly?
            # Or just do a deterministic sweep.
            # Deterministic sweep is faster and reproducible.
            
            # Iterate over all 'u' not in S
            for u in range(N):
                if S & (1 << u):
                    continue # u is already in S
                
                # Check if u can be added to S (without removing anything)
                # This is a "free" improvement if possible.
                if can_add(u, S):
                    S |= (1 << u)
                    current_size += 1
                    best_size = current_size
                    best_S = S
                    improved = True
                    continue # Move to next u, as S changed
                
                # If u cannot be added, try swapping.
                # We need to find a v in S such that u can be added to S \ {v}.
                # This is equivalent to: Is there a v in S such that
                # for all masks m of u, (S \ {v} & m) != m?
                # i.e., for all m, it is NOT the case that all 3 elements of m are in S \ {v}.
                # This means: For every mask m of u that is "blocked" in S, v must be one of the elements in m.
                
                # Let blocked_masks = [m for m in masks[u] if (S & m) == m]
                # If blocked_masks is empty, we could have added u (handled above).
                # So blocked_masks is not empty.
                # We need to pick a v that is in the intersection of all blocked_masks?
                # No, we need to pick a v such that for EACH blocked mask m, v is in m.
                # Because if v is NOT in m, then m is still fully present in S \ {v}, so u is still blocked.
                # So v must be in the intersection of all blocked masks.
                
                blocked_masks = []
                for m in masks[u]:
                    if (S & m) == m:
                        blocked_masks.append(m)
                
                if not blocked_masks:
                    # Should not happen if can_add(u, S) was False
                    continue

                # Find intersection of all blocked masks
                # Start with the first mask
                intersection = blocked_masks[0]
                for m in blocked_masks[1:]:
                    intersection &= m
                    if intersection == 0:
                        break
                
                # The candidates for v are the bits set in 'intersection' that are also in S.
                # Note: intersection contains bits that are in ALL blocked masks.
                # If we remove any v in intersection, ALL blocked masks are broken.
                # So u can be added.
                
                candidates_v = intersection & S
                if candidates_v:
                    # Pick one v. 
                    # Which one? 
                    # Heuristic: Pick the v that is "least useful"? 
                    # Or just the first one?
                    # Let's pick the one with the lowest index for determinism, 
                    # or randomize to escape symmetry?
                    # Let's pick the one that appears in the fewest total masks in S?
                    # For now, just pick the first bit.
                    v = (candidates_v & -candidates_v).bit_length() - 1
                    
                    # Perform swap
                    S &= ~(1 << v)
                    S |= (1 << u)
                    # Size remains same
                    improved = True
                    # We don't update best_size here because size is same, 
                    # but we update best_S? 
                    # Actually, we only care about maximizing size.
                    # If size is same, we might still want to keep the new configuration 
                    # if it allows future improvements.
                    # So we continue the search from this new S.
                    # We don't need to update best_S unless size increases.
                    # But if we get stuck later, we might want to revert?
                    # Standard hill climbing: accept move if size increases OR size same (sideways move).
                    # Here, size is same. We accept it to explore.
                    
                    # If we do many sideways moves, we might cycle.
                    # To prevent cycles, we could keep a history, but that's complex.
                    # Or, we can just do a fixed number of passes.
                    # Since N=512, the state space is large, cycles are unlikely in short time.
                    
                    # Optimization: After a swap, we should restart the inner loop for u?
                    # Or continue?
                    # Continuing is fine.
            
            if not improved:
                break # Local optimum reached

            # If we improved, we might want to restart the sweep?
            # The current loop structure does a full sweep of u.
            # If we swapped, the set S changed, so previous u's might now be addable?
            # No, removing an element only makes it *easier* to add others.
            # So if u was not addable before, it might be addable now.
            # But we are iterating u from 0 to N.
            # If we are at u=100 and we swap something, u=50 (already processed) might now be addable.
            # So a single pass might not be optimal.
            # However, the `while` loop will run another pass if `improved` was True.
            # So it will catch it in the next iteration.

        return best_S

    # Main
    best_S = 0
    best_size = 0
    rng = random.Random(42)
    
    # Try a few different greedy seeds
    seeds = [0, 1, 42, 123, 999]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 5:
            break
            
        S = greedy_construct(seed)
        size = get_size(S)
        
        # Run local search
        S = local_search(S, time_limit - 2)
        size = get_size(S)
        
        if size > best_size:
            best_size = size
            best_S = S

    # Final greedy sweep on the best set found
    for x in range(N):
        if not (best_S & (1 << x)) and can_add(x, best_S):
            best_S |= (1 << x)
            best_size += 1

    # Convert to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_S & (1 << i)]
    return result

# EVOLVE-BLOCK-END
