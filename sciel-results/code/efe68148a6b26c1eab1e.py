import time
import random
from itertools import combinations

# Global variables for speed
ALL_MASKS = []
MASK_TO_IDX = {}
CONFLICTS = []
N_MASKS = 0

def precompute():
    global ALL_MASKS, MASK_TO_IDX, CONFLICTS, N_MASKS
    
    # Generate all 4-subsets of 16 elements
    # Represent as bitmasks (16 bits)
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        ALL_MASKS.append(mask)
    
    N_MASKS = len(ALL_MASKS)
    for i, m in enumerate(ALL_MASKS):
        MASK_TO_IDX[m] = i
        
    # Precompute conflicts
    # conflicts[i][j] is a bitmask of indices k such that {i, j, k} is a sunflower
    CONFLICTS = [[0] * N_MASKS for _ in range(N_MASKS)]
    
    # To speed up, we can group by (union, intersection)
    # But let's just iterate pairs.
    # 1820 * 1819 / 2 ~ 1.65 million pairs.
    
    for i in range(N_MASKS):
        A = ALL_MASKS[i]
        for j in range(i + 1, N_MASKS):
            B = ALL_MASKS[j]
            
            I = A & B
            S = A | B
            
            I_size = bin(I).count('1')
            P_size = 4 - I_size
            
            # C must contain I and be disjoint from S \ I
            # C = I U P, where P subset of [16] \ S
            # |P| = P_size
            
            if P_size < 0:
                continue
                
            U = ((1 << 16) - 1) ^ S # Universe minus S
            
            # Generate all subsets P of U of size P_size
            # And check if I U P is in our set (it always is, as we have all 4-subsets)
            # But we need the index.
            
            # Iterate combinations of bits in U
            # We can extract bits from U
            bits_in_U = []
            temp_U = U
            while temp_U:
                lsb = temp_U & -temp_U
                bits_in_U.append(lsb)
                temp_U ^= lsb
                
            if len(bits_in_U) < P_size:
                continue
                
            conflict_mask = 0
            for P_combo in combinations(bits_in_U, P_size):
                P_mask = 0
                for b in P_combo:
                    P_mask |= b
                C_mask = I | P_mask
                idx_C = MASK_TO_IDX[C_mask]
                conflict_mask |= (1 << idx_C)
                
            CONFLICTS[i][j] = conflict_mask
            CONFLICTS[j][i] = conflict_mask

def construct():
    global ALL_MASKS, N_MASKS, CONFLICTS
    
    # Ensure precomputation is done
    if not ALL_MASKS:
        start_time = time.time()
        precompute()
        elapsed = time.time() - start_time
        # If precomputation took too long, we might have issues, but 90s is generous.
        
    best_size = 0
    best_set = []
    
    # Deadline guard
    deadline = time.time() + 85.0 # 5s margin
    
    # Randomized Greedy with Restarts
    # We can try different strategies for ordering
    # 1. Random shuffle
    # 2. Degree based (not precomputed, maybe too slow)
    
    indices = list(range(N_MASKS))
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        selected_mask = 0
        blocked_mask = 0
        selected_indices = []
        
        for u in indices:
            # Check if u is blocked or already selected
            bit_u = 1 << u
            if (selected_mask & bit_u) or (blocked_mask & bit_u):
                continue
                
            # Add u
            selected_mask |= bit_u
            selected_indices.append(u)
            
            # Update blocked mask
            # For all v in selected_indices (except u), block conflicts[u][v]
            # We need to iterate over set bits of selected_mask excluding u
            m = selected_mask & ~bit_u
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                blocked_mask |= CONFLICTS[u][v]
                m ^= lsb
                
            # Optimization: if we have picked many, we can break early if we know we can't beat best?
            # Hard to know.
            
        current_size = len(selected_indices)
        if current_size > best_size:
            best_size = current_size
            best_set = selected_indices[:]
            
    # Convert indices back to sorted lists of elements
    result = []
    for idx in best_set:
        mask = ALL_MASKS[idx]
        # Convert mask to list of 1-based indices
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

# Run precomputation on import? No, better to do it in construct or lazily.
# But precomputation takes time. Let's do it in a way that it happens once.
# The prompt says "Each evaluation must finish within 60s" (actually 90s kill).
# Precomputation ~ 5-10s. Search ~ 70s.
# We should probably precompute in a background thread or just accept the delay in construct.
# Since construct is called once per evaluation, doing it inside is fine.

# To be safe, let's make precompute efficient.
# The inner loop `for P_combo in combinations(...)` is the bottleneck.
# We can precompute a list of indices for each (I_mask, U_mask) pair?
# There are not many distinct (I, U) pairs.
# I is a subset of S. S is union of two 4-sets.
# Let's stick to the current method but ensure it's as fast as possible.

# One optimization: 
# Instead of `combinations(bits_in_U, P_size)`, we can iterate through all 4-subsets C 
# and check if they satisfy the condition for (A, B).
# But that's O(N) per pair, which is worse.
# The current method is O(2^|U|) which is small.

# Let's refine the bit iteration in the search phase.
# `m = selected_mask & ~bit_u`
# `while m: lsb = m & -m; ...`
# This is efficient.

# We can also add a "local search" improvement step?
# If we get stuck, we can try to swap a selected element with a blocked one?
# Given the time constraint, simple greedy with many restarts is robust.

# Let's verify the sunflower condition again.
# "No three distinct sets may have all three pairwise intersections equal"
# A, B, C distinct.
# A cap B = A cap C = B cap C = K.
# My precomputation finds C such that C cap A = K and C cap B = K where K = A cap B.
# This is exactly the condition.
# And we block C if it exists.
# So if A, B are selected, C is blocked.
# If we try to select C later, it will be skipped.
# This correctly prevents sunflowers.

# What if A, B, C are selected in different order?
# If A, B selected, C is blocked.
# If A, C selected, B is blocked.
# If B, C selected, A is blocked.
# So regardless of order, the third one will be blocked when the first two are in.
# Correct.

# One edge case: "including an empty common core".
# This is covered by K = 0.
# If K=0, A and B are disjoint.
# C must be disjoint from A and B.
# C subset of U = [16] \ (A U B).
# |C| = 4.
# This is handled.

# Final check on performance.
# Precomputation: 1.6M pairs.
# Inside loop:
#   I = A & B
#   S = A | B
#   I_size = bin(I).count('1')
#   P_size = 4 - I_size
#   U = ~S & 0xFFFF
#   bits_in_U extraction: O(16)
#   combinations loop: avg ~20-30 iterations.
#   Total ~ 1.6M * 30 = 50M operations.
#   In Python, 50M simple ops might take 5-10 seconds.
#   This is acceptable.

# Search:
#   1820 candidates.
#   Inner loop over selected bits.
#   If size ~ 200, inner loop ~ 200 iterations.
#   Total ~ 1820 * 200 = 360k bit ops per restart.
#   With 1000 restarts, 360M bit ops.
#   Bit ops on Python ints (arbitrary precision) are fast (C level).
#   1000 restarts should finish in < 10 seconds.
#   So we can do many more restarts.

# Let's increase the number of restarts by running until deadline.

# One potential issue: `random.shuffle(indices)` takes time.
# 1820 elements shuffle is fast.

# Let's write the final code.

