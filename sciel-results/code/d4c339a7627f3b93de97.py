import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute incidence structure
    # For each AP, we need to know how the objective changes if we flip any subset of its bits.
    # Objective term for an AP: 1 if all 4 bits are equal, else 0.
    # Let the bits be b0, b1, b2, b3.
    # mono = 1 if (b0==b1==b2==b3) else 0.
    
    # We can precompute for each AP and each pair of indices (i, j) within that AP,
    # the change in the mono-status if we flip bits i and j.
    
    # However, flipping a bit affects all APs containing it.
    # Let's use a global coordinate descent approach.
    # We want to find the best pair of bits (i, j) in the whole system to flip.
    
    # The total objective is F(c) = sum_{k} mono_k(c).
    # Delta(i, j) = F(c with i,j flipped) - F(c).
    
    # Only APs containing i or j are affected.
    # Let S_i be the set of APs containing i.
    # Let S_j be the set of APs containing j.
    # Affected APs = S_i union S_j.
    
    # For an AP k in S_i ∩ S_j: both bits in this AP are flipped.
    # For an AP k in S_i \ S_j: only bit i in this AP is flipped.
    # For an AP k in S_j \ S_i: only bit j in this AP is flipped.
    
    # We can precompute the "contribution change" for single flips and double flips within an AP.
    
    # Let's build a sparse representation.
    # index_to_aps[i] = list of ap_indices containing i.
    index_to_aps = [[] for _ in range(N)]
    for k in range(num_aps):
        for idx in aps[k]:
            index_to_aps[idx].append(k)
            
    # To make the double-flip search efficient, we need to quickly compute Delta(i, j).
    # Delta(i, j) = sum_{k in S_i} delta_single(k, pos_i) 
    #             + sum_{k in S_j} delta_single(k, pos_j)
    #             - sum_{k in S_i ∩ S_j} (delta_single(k, pos_i) + delta_single(k, pos_j) - delta_double(k, pos_i, pos_j))
    # Wait, this logic is tricky.
    
    # Let's just compute Delta(i, j) directly for candidate pairs.
    # But checking all pairs is O(N^2 * deg). N=256, N^2 ~ 65k. deg ~ 40. 2.6M ops.
    # This is fast enough for a few iterations?
    # 2.6M ops per "sweep". If we do 100 sweeps, 260M ops. Might be too slow in Python loops.
    # We need to vectorize.
    
    # Alternative: Single-bit greedy is fast. Let's try to improve it by allowing "double flips" 
    # only when stuck, or using a smarter initialization.
    
    # Actually, the prompt suggests a "quadratic-form-based objective".
    # F(c) = sum_k I(c_k0==c_k1==c_k2==c_k3).
    # This is a degree-4 polynomial.
    
    # Let's stick to the efficient single-bit greedy but optimize the inner loop heavily 
    # and add a "kick" mechanism that uses precomputed good pairs.
    
    # Better Idea: Use the structure of 4-APs.
    # Many 4-APs are short (small d).
    # Let's precompute the "delta" for flipping each bit for the current state.
    # We do this by iterating over all APs and updating the deltas of their 4 bits.
    
    # This is O(num_aps * 4) per evaluation of "best flip".
    # num_aps is ~ 10,000. 40,000 ops per evaluation.
    # If we do 1000 evaluations (steps), that's 40M ops. Very fast in NumPy/Python if optimized.
    
    # Let's implement a highly optimized single-bit greedy with "random restarts" 
    # and "pair-flip kicks".
    
    # Precompute for each AP the 4 indices.
    # aps is (num_aps, 4).
    
    # Helper: Calculate deltas for all N bits given current coloring c.
    # deltas[i] = change in score if we flip bit i.
    
    # We can compute this by:
    # 1. Initialize deltas to 0.
    # 2. For each AP k:
    #      Let bits be c[aps[k,0]], ..., c[aps[k,3]].
    #      Determine if the AP is currently mono.
    #      If mono: flipping any of the 4 bits breaks mono (delta -= 1 for each).
    #      If not mono:
    #          Check if flipping a specific bit makes it mono.
    #          It makes it mono if the other 3 bits are equal AND the flipped bit becomes equal to them.
    
    # Let's vectorize the AP loop.
    
    def compute_deltas(c):
        deltas = np.zeros(N, dtype=np.int32)
        
        # Get all bits for all APs: shape (num_aps, 4)
        bits = c[aps]
        
        # Check if each AP is mono
        # mono_k = 1 if bits[k,0]==bits[k,1]==bits[k,2]==bits[k,3]
        b0, b1, b2, b3 = bits[:, 0], bits[:, 1], bits[:, 2], bits[:, 3]
        
        is_mono = (b0 == b1) & (b1 == b2) & (b2 == b3)
        
        # Case 1: AP is mono.
        # Flipping any of the 4 bits will make it non-mono (since 3 bits remain same, 1 changes).
        # So delta -= 1 for all 4 indices.
        # We can use np.add.at
        if np.any(is_mono):
            mono_aps = aps[is_mono] # shape (num_mono, 4)
            # For each AP in mono_aps, subtract 1 from deltas at its 4 indices
            # np.add.at(deltas, mono_aps.ravel(), -1)
            # But we need to be careful with duplicates? No, indices are distinct within an AP.
            # However, an index might appear in multiple mono APs.
            # np.add.at handles accumulation.
            np.add.at(deltas, mono_aps.ravel(), -1)
            
        # Case 2: AP is NOT mono.
        # Flipping bit j makes it mono iff the other 3 bits are equal to each other AND 
        # different from bit j (so that flipping j makes it equal to them).
        # Actually, if other 3 are equal to val, and bit j is 1-val, then flipping j makes it val.
        # If bit j is already val, then the AP would have been mono (4 equal), which is Case 1.
        # So in Case 2, if other 3 are equal, bit j MUST be different.
        
        not_mono_mask = ~is_mono
        if not np.any(not_mono_mask):
            return deltas
            
        nm_bits = bits[not_mono_mask]
        nm_aps = aps[not_mono_mask]
        
        # For each non-mono AP, check each of the 4 positions.
        # pos 0: if b1==b2==b3, then flipping b0 makes it mono.
        # pos 1: if b0==b2==b3, then flipping b1 makes it mono.
        # pos 2: if b0==b1==b3, then flipping b2 makes it mono.
        # pos 3: if b0==b1==b2, then flipping b3 makes it mono.
        
        b0_nm, b1_nm, b2_nm, b3_nm = nm_bits[:, 0], nm_bits[:, 1], nm_bits[:, 2], nm_bits[:, 3]
        
        # Check conditions
        cond0 = (b1_nm == b2_nm) & (b2_nm == b3_nm)
        cond1 = (b0_nm == b2_nm) & (b2_nm == b3_nm)
        cond2 = (b0_nm == b1_nm) & (b1_nm == b3_nm)
        cond3 = (b0_nm == b1_nm) & (b1_nm == b2_nm)
        
        # For each condition, if true, delta += 1 for the corresponding index in that AP.
        # We need to map the AP index back to the global AP index to get the correct row in aps.
        # nm_aps is already the global indices.
        
        # Indices to update:
        # For cond0: index is nm_aps[:, 0]
        # For cond1: index is nm_aps[:, 1]
        # For cond2: index is nm_aps[:, 2]
        # For cond3: index is nm_aps[:, 3]
        
        if np.any(cond0):
            np.add.at(deltas, nm_aps[cond0, 0], 1)
        if np.any(cond1):
            np.add.at(deltas, nm_aps[cond1, 1], 1)
        if np.any(cond2):
            np.add.at(deltas, nm_aps[cond2, 2], 1)
        if np.any(cond3):
            np.add.at(deltas, nm_aps[cond3, 3], 1)
            
        return deltas

    def calc_score(c):
        bits = c[aps]
        b0, b1, b2, b3 = bits[:, 0], bits[:, 1], bits[:, 2], bits[:, 3]
        is_mono = (b0 == b1) & (b1 == b2) & (b2 == b3)
        return np.sum(is_mono)

    rng = np.random.default_rng(42)
    
    best_score = np.inf
    best_coloring = None
    
    # Strategy:
    # 1. Start with a few heuristic initializations (e.g., random, alternating, Thue-Morse-like).
    # 2. Run Greedy Hill Climbing (Single Bit).
    # 3. If stuck, try "Double Bit" flips.
    #    To find best double bit flip:
    #    We can approximate or calculate exactly.
    #    Exact calculation of best double flip is expensive if done naively.
    #    However, we can use the single-bit deltas to filter candidates.
    #    If delta(i) is very negative and delta(j) is very negative, flipping both might be good.
    #    Or if delta(i) is positive but delta(j) is negative, etc.
    
    # Let's implement a "Greedy with Double-Step Kicks".
    
    def greedy_optimize(c, max_steps=100):
        current_score = calc_score(c)
        for _ in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            deltas = compute_deltas(c)
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                break
                
            # Find best index
            candidates = np.where(deltas == min_delta)[0]
            i_star = candidates[rng.integers(0, len(candidates))]
            
            c[i_star] = 1 - c[i_star]
            current_score += min_delta
            
        return c, current_score

    # Double flip search:
    # We want to find i, j such that Delta(i, j) < 0.
    # Delta(i, j) = F(c with i,j flipped) - F(c).
    
    def find_best_double_flip(c):
        # This is O(N^2 * cost). Too slow for full N^2.
        # We only check pairs where single-bit deltas are "promising".
        # Or just check a random subset of pairs.
        
        # Let's compute single deltas first.
        deltas = compute_deltas(c)
        
        # Candidate pairs:
        # 1. i, j where delta(i) < 0 and delta(j) < 0. (Both improve individually)
        # 2. i, j where delta(i) < 0 and delta(j) > 0 (but combined effect is negative).
        
        # For simplicity and speed, let's just try random pairs from the set of indices with delta < 0.
        # If no such indices, try random pairs from all indices.
        
        neg_indices = np.where(deltas < 0)[0]
        if len(neg_indices) == 0:
            # No single improvement. Try random pairs.
            # Sample 100 random pairs.
            best_pair = None
            best_delta = 0 # We want negative
            for _ in range(100):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j: continue
                # Calculate Delta(i, j)
                # This requires checking affected APs.
                # For now, skip complex double flip logic if no single improvement, 
                # as it's complex to vectorize efficiently on the fly.
                # Instead, we rely on random kicks.
                pass
            return None, 0
        
        # If there are negative deltas, the greedy step would have taken one.
        # So we only call this if we are stuck (min_delta >= 0).
        # Thus neg_indices should be empty if we call this from the stuck state.
        # So we need a way to find double flips that help even if single flips don't.
        
        # Let's just do a random kick of 2 bits.
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        while j == i:
            j = rng.integers(0, N)
            
        # Check if this kick improves.
        c_orig = c.copy()
        c[i] = 1 - c[i]
        c[j] = 1 - c[j]
        
        new_score = calc_score(c)
        old_score = calc_score(c_orig)
        
        if new_score < old_score:
            return (i, j), new_score - old_score
        else:
            c[:] = c_orig
            return None, 0

    # Multi-start
    num_restarts = 50
    
    # Initializations
    # 1. Random
    # 2. Alternating
    # 3. Thue-Morse (0,1,1,0,1,0,0,1...)
    
    init_strategies = []
    
    # Random
    init_strategies.append(lambda: rng.integers(0, 2, size=N, dtype=np.int8))
    
    # Alternating
    init_strategies.append(lambda: np.arange(N) % 2)
    
    # Thue-Morse
    def thue_morse(n):
        t = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            # t[i] = t[i//2] ^ (i%2)
            # Vectorized:
            pass
        # Simple loop for N=256 is fast
        t = [0]
        for i in range(1, n):
            # Count bits in i
            # t[i] = popcount(i) % 2
            t.append(bin(i).count('1') % 2)
        return np.array(t, dtype=np.int8)
        
    init_strategies.append(lambda: thue_morse(N))
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Pick an initialization
        if r == 0:
            c = init_strategies[1]() # Alternating
        elif r == 1:
            c = init_strategies[2]() # Thue-Morse
        else:
            c = init_strategies[0]() # Random
            
        # Greedy Optimize
        c, current_score = greedy_optimize(c, max_steps=100)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        # If stuck, try a few double-flip kicks
        for kick_iter in range(10):
            if time.time() > deadline - 5:
                break
                
            # Try a random 2-bit flip
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            while j == i:
                j = rng.integers(0, N)
                
            c[i] = 1 - c[i]
            c[j] = 1 - c[j]
            
            new_score = calc_score(c)
            
            if new_score < current_score:
                current_score = new_score
                # Continue greedy
                c, current_score = greedy_optimize(c, max_steps=50)
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = c.copy()
            else:
                # Revert
                c[i] = 1 - c[i]
                c[j] = 1 - c[j]
                
    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()
