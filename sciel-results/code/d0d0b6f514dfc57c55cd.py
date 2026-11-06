"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)
    
    # Precompute a "forbidden mask" for each pair of subsets.
    # If we have two sets A and B in our family, a new set M forms a sunflower with them
    # if and only if:
    # 1. M contains K = A ∩ B
    # 2. M is disjoint from F = (A \ K) ∪ (B \ K)
    # This is equivalent to: (M & K) == K AND (M & F) == 0
    # We can precompute K and F for every pair (i, j) in the universe of all_subsets.
    
    # However, storing K and F for ~1.6M pairs might be memory heavy if we store as tuples of ints.
    # But 1.6M * 2 * 8 bytes (assuming 64-bit ints) is ~25MB, which is fine.
    # But we only need to check pairs *within the current family*.
    # The current program checks pairs within the family on the fly.
    # Let's stick to checking on the fly but optimize the inner loop structure.
    
    # Actually, a smarter approach for the greedy step:
    # Instead of checking all pairs in the family for a new candidate M,
    # we can group the family members by their intersection with M?
    # No, that's what the current program does in can_add.
    
    # Let's try a different optimization:
    # Maintain a list of "conflict signatures" for the current family.
    # A conflict signature is (K, F) for a pair (A, B) in the family.
    # When adding a new set M, we check if M violates any (K, F).
    # If we add M, we generate |F| new signatures.
    # Total signatures is O(|F|^2).
    # Checking all signatures is O(|F|^2).
    # This is the same complexity as the current approach, but the constant might be better
    # because we avoid the nested loop overhead in Python and can potentially use numpy or just simpler loops.
    
    # Let's implement the "signature" approach.
    
    def can_add_sig(mask_new, sigs):
        # sigs is a list of (K, F)
        for K, F in sigs:
            if (mask_new & K) == K:
                if (mask_new & F) == 0:
                    return False
        return True

    def generate_sigs(mask_new, family_masks):
        # Generate new signatures involving mask_new and each member of family_masks
        new_sigs = []
        for m in family_masks:
            K = mask_new & m
            # F = (mask_new \ K) | (m \ K)
            # mask_new \ K = mask_new & ~K
            # m \ K = m & ~K
            # F = (mask_new ^ K) | (m ^ K)  -- Note: XOR with K removes K bits
            F = (mask_new ^ K) | (m ^ K)
            new_sigs.append((K, F))
        return new_sigs

    best_family = []
    best_size = 0
    
    # Strategy 1: Deterministic Greedy (Lexicographic)
    # Using the signature approach for speed
    
    chosen = []
    sigs = []
    for mask in all_subsets:
        if time.time() > deadline:
            break
        if can_add_sig(mask, sigs):
            # Add mask
            chosen.append(mask)
            # Update signatures
            sigs.extend(generate_sigs(mask, chosen[:-1]))
            
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Strategy 2: Randomized Greedy Restarts
    # We will run many restarts.
    # The signature approach is faster, so we can do more restarts.
    
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:  # 2s safety margin
            break
        
        # Use a different seed for each restart
        random.seed(12345 + restart)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        
        chosen = []
        sigs = []
        
        for mask in shuffled:
            if time.time() > deadline - 2.0:
                break
            if can_add_sig(mask, sigs):
                chosen.append(mask)
                sigs.extend(generate_sigs(mask, chosen[:-1]))
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            if best_size >= 250:
                break

    # Strategy 3: Local Search / Hill Climbing on the best family found
    # If we have time, try to swap elements or add/remove to improve the size?
    # Since the constraint is "no three sunflowers", removing a set never creates a new sunflower.
    # Adding a set is the only way to increase size.
    # If we are stuck at a local maximum, we might need to remove some sets to allow others in.
    # This is complex. Let's stick to greedy restarts which are simple and effective.
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
