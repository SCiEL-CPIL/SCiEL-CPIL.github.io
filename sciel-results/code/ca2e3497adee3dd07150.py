import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    rng = random.Random(42)
    
    # 1. Precomputation
    # All triples on 16 points
    all_triples = []
    triple_masks = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
                triple_masks.append((1 << a) | (1 << b) | (1 << c))
    
    n_triples = len(all_triples) # 560
    
    # Precompute supersets for each triple
    # supersets[i] is a list of masks that contain triple i
    supersets = [[] for _ in range(n_triples)]
    full_mask = (1 << 16) - 1
    for i, tm in enumerate(triple_masks):
        remaining = full_mask ^ tm
        sub = remaining
        while True:
            supersets[i].append(tm | sub)
            if sub == 0:
                break
            sub = (sub - 1) & remaining
            
    # Precompute popcount
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Precompute compatibility: compat[i] is bitmask of triples compatible with i
    # We need this to quickly find candidates for swap
    compat = [0] * n_triples
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            inter = mi & mj
            # Count bits
            cnt = 0
            t = inter
            while t:
                cnt += 1
                t &= t - 1
            if cnt <= 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    # Helper to compute alpha
    def compute_alpha(selected_indices):
        if not selected_indices:
            return 16
        bad = bytearray(1 << 16)
        for idx in selected_indices:
            # Mark supersets
            for m in supersets[idx]:
                bad[m] = 1
        # Find max popcount of non-bad
        best = 0
        # Iterate only over masks? 65536 is small.
        for m in range(1 << 16):
            if not bad[m]:
                c = popcount[m]
                if c > best:
                    best = c
        return best

    # Helper to build a random maximal system
    def generate_maximal():
        available = (1 << n_triples) - 1
        selected = set()
        while available:
            # Pick random available triple
            bits = bin(available).count('1')
            if bits == 0:
                break
            idx_in_bits = rng.randrange(bits)
            tmp = available
            for _ in range(idx_in_bits):
                tmp &= tmp - 1
            bit = tmp & (-tmp)
            i = bit.bit_length() - 1
            
            selected.add(i)
            # Update available
            available &= compat[i]
            available &= ~(1 << i)
        return selected

    # Initial best
    best_system = generate_maximal()
    best_alpha = compute_alpha(best_system)
    current_system = set(best_system)
    current_alpha = best_alpha
    
    # SA Parameters
    T = 2.0
    T_min = 0.05
    cooling = 0.999
    
    # Precompute list of triples for faster iteration in swap generation
    # We need to find j compatible with all k in current_system - {i}
    # Let S = current_system - {i}
    # j must be in intersection of compat[k] for all k in S
    # AND j not in current_system
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Move: Swap
        # Pick i to remove
        if not current_system:
            # Should not happen for maximal
            break
        i = rng.choice(list(current_system))
        S = current_system - {i}
        
        # Find candidates j
        # Start with all triples
        candidates = (1 << n_triples) - 1
        # Remove current system triples
        for k in current_system:
            candidates &= ~(1 << k)
            
        # Intersect with compat of all in S
        # If S is large, this is fast.
        for k in S:
            candidates &= compat[k]
            
        if candidates == 0:
            # No swap possible. Try a remove? 
            # Or just skip.
            continue
            
        # Pick random j from candidates
        bits = bin(candidates).count('1')
        idx_in_bits = rng.randrange(bits)
        tmp = candidates
        for _ in range(idx_in_bits):
            tmp &= tmp - 1
        bit = tmp & (-tmp)
        j = bit.bit_length() - 1
        
        new_system = S | {j}
        new_alpha = compute_alpha(new_system)
        
        delta = new_alpha - current_alpha
        
        # Acceptance
        if delta <= 0:
            # Improvement or equal
            # If equal, accept with prob 0.5 to allow random walk
            if delta < 0 or rng.random() < 0.5:
                current_system = new_system
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_system = set(current_system)
        else:
            # Worsening
            # Boltzmann factor
            if rng.random() < (2.71828 ** (-delta / T)):
                current_system = new_system
                current_alpha = new_alpha
                
        T *= cooling
        
        # Periodic restart if stuck?
        # Maybe not needed if T cools down.
        
    return [list(all_triples[i]) for i in sorted(best_system)]

# EVOLVE-BLOCK-END
