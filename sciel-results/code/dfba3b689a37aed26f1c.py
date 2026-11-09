import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safe margin for 60s limit

    # 1. Precompute Conflicts
    # conflicts[x] is a list of 3-tuples (indices 0..N-1)
    # If S contains all 3 indices in a tuple, x CANNOT be added.
    # This is because x + those 3 form a 4-AP.
    
    # First, generate all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We store them as 0-indexed tuples for numpy compatibility
    all_aps = []
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            # 1-indexed: a, a+d, a+2d, a+3d
            # 0-indexed: a-1, a+d-1, a+2d-1, a+3d-1
            ap = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            all_aps.append(ap)
            
    # Build conflict lists for each element
    # conflicts[i] = list of arrays of shape (3,)
    conflicts = [[] for _ in range(N)]
    
    for ap in all_aps:
        a, b, c, d = ap
        # For a, the others are b, c, d
        conflicts[a].append((b, c, d))
        # For b, the others are a, c, d
        conflicts[b].append((a, c, d))
        # For c, the others are a, b, d
        conflicts[c].append((a, b, d))
        # For d, the others are a, b, c
        conflicts[d].append((a, b, c))
        
    # Convert to numpy arrays for vectorized checking
    # conflicts_np[i] will be a 2D array of shape (num_conflicts, 3)
    conflicts_np = [None] * N
    for i in range(N):
        if conflicts[i]:
            conflicts_np[i] = np.array(conflicts[i], dtype=np.int32)
        else:
            conflicts_np[i] = np.empty((0, 3), dtype=np.int32)

    def can_add(x, S):
        """
        Check if x (0-indexed) can be added to S (numpy bool array).
        Returns True if valid.
        """
        if S[x]:
            return False
        confs = conflicts_np[x]
        if len(confs) == 0:
            return True
        # Check if any row in confs is all True in S
        # S[confs] has shape (num_conflicts, 3)
        # np.all(S[confs], axis=1) has shape (num_conflicts,)
        # We want to know if ANY is True
        return not np.any(np.all(S[confs], axis=1))

    def get_violated_aps_if_add(x, S):
        """
        If x were added, which APs would be violated?
        Returns a list of AP tuples (0-indexed) that are fully in S union {x}.
        """
        confs = conflicts_np[x]
        if len(confs) == 0:
            return []
        
        # S[confs] shape (K, 3)
        # We need to find rows where all 3 are True
        # But wait, confs are the *other* 3 elements.
        # If all 3 are in S, then {x} + {3 elements} is a 4-AP in S union {x}.
        
        # We need to map back to the full AP.
        # conflicts_np[x] only stores the other 3.
        # We lost the AP id. 
        # For the "Kick" strategy, we just need to know *which elements* to remove.
        # If a conflict triple (b, c, d) is fully in S, we can remove b, c, or d.
        # We don't strictly need the AP id, just the elements.
        
        # However, if we want to be smart, we might want to remove the element 
        # that is involved in the most other conflicts? 
        # For now, random selection from the blocked triples is sufficient.
        
        # Find indices of triples that are fully present in S
        # all_present = np.all(S[confs], axis=1)
        # blocked_indices = np.where(all_present)[0]
        # blocked_triples = confs[blocked_indices]
        
        # Return the blocked triples (each is a tuple of 3 indices)
        all_present = np.all(S[confs], axis=1)
        if not np.any(all_present):
            return []
        return confs[all_present]

    # Greedy Construction
    def greedy_construct(seed):
        rng = random.Random(seed)
        S = np.zeros(N, dtype=bool)
        elements = list(range(N))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(x, S):
                S[x] = True
        return S

    def local_search(S, rng, time_budget):
        S = S.copy()
        current_size = np.sum(S)
        best_size = current_size
        best_S = S.copy()
        
        # List of indices not in S
        # We'll pick random candidates from 0..N-1
        
        while time.time() - start_time < time_budget:
            # 1. Try to add a random element
            x = rng.randint(0, N - 1)
            
            if S[x]:
                continue # Already in set
                
            if can_add(x, S):
                S[x] = True
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # Cannot add x. Find blocking triples.
                blocked_triples = get_violated_aps_if_add(x, S)
                
                if len(blocked_triples) > 0:
                    # Pick a random blocking triple
                    # blocked_triples is a numpy array of shape (K, 3)
                    k = rng.randint(0, len(blocked_triples) - 1)
                    triple = blocked_triples[k]
                    
                    # Pick a random element from the triple to remove
                    # This "kicks" the set, potentially allowing x (or others) to be added
                    # We remove one element from S
                    y = triple[rng.randint(0, 2)]
                    
                    if S[y]:
                        S[y] = False
                        current_size -= 1
                        
                        # Now try to add x again?
                        # It might still be blocked by other APs.
                        # But we broke one barrier.
                        # Let's try to add x.
                        if can_add(x, S):
                            S[x] = True
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
                        else:
                            # Still blocked. 
                            # We are now at size -1 (relative to before kick) or same (if x added).
                            # If x not added, we are at size -1.
                            # This is risky. 
                            # Strategy: If we don't improve size after a kick, 
                            # we might want to revert? 
                            # Or just continue. The random walk will eventually find a way.
                            # To prevent drift to smaller sets, we can track a "patience" counter.
                            # If current_size < best_size - 5, revert to best_S.
                            if current_size < best_size - 5:
                                S = best_S.copy()
                                current_size = best_size
                else:
                    # This case should not happen if can_add returned False
                    # can_add returns False iff there is at least one blocked triple.
                    pass
                    
            # 2. Occasionally try a pure swap or removal to escape
            # If we are stuck (no size increase for a while), we can force a swap.
            # But the kick mechanism above effectively does swaps (remove y, add x).
            # If x fails, we just removed y. 
            # To avoid shrinking indefinitely, we rely on the "revert to best" logic.
            
            # Additional heuristic: Try to add *any* available element if possible?
            # The random pick might miss easy additions.
            # Let's occasionally do a "sweep" of a few random elements.
            if rng.random() < 0.05:
                for _ in range(10):
                    y = rng.randint(0, N - 1)
                    if not S[y] and can_add(y, S):
                        S[y] = True
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S.copy()

        return best_S

    # Main
    best_S = np.zeros(N, dtype=bool)
    best_size = 0
    rng = random.Random(42)
    
    # Multiple restarts with different greedy seeds
    # Since N=512 is small, greedy is fast.
    # We can afford many restarts.
    
    seeds = [0, 1, 2, 42, 123, 999, 1024, 2048]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 10:
            break
            
        S = greedy_construct(seed)
        size = np.sum(S)
        
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
        # Run local search on this S
        # Allocate remaining time / remaining seeds roughly
        # Simpler: just run local search for a fixed chunk of time or until improvement stops?
        # Let's run local search for a portion of the remaining time.
        
        remaining_time = time_limit - (time.time() - start_time)
        # Allocate 20% of remaining time to this local search?
        # Or just run it and let the outer loop handle time?
        # The local_search function has its own time check.
        
        S = local_search(S, rng, time_limit - 2)
        size = np.sum(S)
        
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # Final check: Try to add any remaining elements greedily
    # This catches any easy wins missed by the stochastic search
    for x in range(N):
        if not best_S[x] and can_add(x, best_S):
            best_S[x] = True
            best_size += 1

    # Convert to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_S[i]]
    return result

# EVOLVE-BLOCK-END
