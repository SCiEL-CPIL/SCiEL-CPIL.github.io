import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute incidence structure for fast updates
    # We need to efficiently compute the score and gradients/updates.
    
    # For the "Spin System" approach with vectorized updates:
    # Let s_i in {-1, 1}. 
    # A 4-AP (i1, i2, i3, i4) is monochromatic iff s_i1 = s_i2 = s_i3 = s_i4.
    # This is equivalent to: (s_i1*s_i2 + 1)/2 * (s_i2*s_i3 + 1)/2 * ... no, that's not right.
    # Let's use the indicator: I_mono = 1 if all equal, 0 otherwise.
    # I_mono = (1 + s1*s2)/2 * (1 + s2*s3)/2 * (1 + s3*s4)/2 ? No.
    # If all are 1: (1+1)/2=1. If all -1: (1+1)/2=1.
    # If 1,1,1,-1: (1+1)/2=1, (1+1)/2=1, (1-1)/2=0 -> 0.
    # If 1,1,-1,1: (1+1)/2=1, (1-1)/2=0, ... -> 0.
    # So yes, I_mono = prod_{j=0 to 2} (1 + s_{i_j} s_{i_{j+1}}) / 2.
    
    # Wait, let's verify.
    # s1=1, s2=1, s3=1, s4=1.
    # t1 = (1+1*1)/2 = 1
    # t2 = (1+1*1)/2 = 1
    # t3 = (1+1*1)/2 = 1
    # Prod = 1. Correct.
    
    # s1=1, s2=1, s3=1, s4=-1.
    # t1 = 1, t2 = 1, t3 = (1+1*(-1))/2 = 0.
    # Prod = 0. Correct.
    
    # s1=1, s2=-1, s3=1, s4=1.
    # t1 = 0. Prod = 0. Correct.
    
    # So Score = sum_{APs} 0.125 * (1 + s_a s_b) (1 + s_b s_c) (1 + s_c s_d).
    
    # This is a polynomial in spins.
    # We can use a local search that minimizes this.
    # The derivative with respect to flipping s_k (which changes s_k to -s_k) is:
    # Delta = New_Score - Old_Score.
    
    # Let's stick to the previous efficient delta calculation but perhaps use a different search strategy.
    # The prompt suggests: "vectorized, stochastic subgradient descent ... updates the entire color vector simultaneously".
    
    # This suggests a method like:
    # 1. Compute "pressure" on each spin from all APs it belongs to.
    # 2. Flip spins that have high "pressure" (i.e., flipping them would reduce the score significantly).
    # 3. Do this in parallel (vectorized).
    
    # Let's precompute the contributions.
    # For each AP j = (a, b, c, d):
    # Term = 0.125 * (1 + s_a s_b) (1 + s_b s_c) (1 + s_c s_d)
    # Let P_ab = 1 + s_a s_b, P_bc = 1 + s_b s_c, P_cd = 1 + s_c s_d.
    # Term = 0.125 * P_ab * P_bc * P_cd.
    
    # If we flip s_k, which terms change?
    # Only APs containing k.
    # For an AP containing k, we need to recalculate the term.
    
    # Vectorized update:
    # We can maintain the current score S.
    # We can compute for each index i, the "gain" of flipping i.
    # Gain(i) = S - S_if_flipped_i.
    # If Gain(i) > 0, flipping i improves the score.
    
    # To compute Gain(i) for all i simultaneously:
    # This is hard to do in one shot without iterating over APs.
    # However, we can iterate over APs, and for each AP, update the "potential gain" of its 4 members.
    
    # Let's try this:
    # gain = np.zeros(N)
    # For each AP (a,b,c,d):
    #   current_val = 1 if mono else 0
    #   If we flip a: is it mono?
    #   If we flip b: is it mono?
    #   ...
    #   gain[a] += current_val - val_if_flip_a
    #   gain[b] += current_val - val_if_flip_b
    #   ...
    
    # This loop is over num_aps (~10k). Inside, we do constant work.
    # 10k iterations in Python is fast (~1ms).
    # Then we select the top K gains and flip them.
    # This is a "parallel" or "batch" local search step.
    
    # Let's implement this "Batch Local Search" with random restarts and simulated annealing logic.
    
    # Precompute arrays for APs
    a_idx = aps[:, 0]
    b_idx = aps[:, 1]
    c_idx = aps[:, 2]
    d_idx = aps[:, 3]
    
    # We need to check mono status quickly.
    # s is an array of 0/1 or -1/1. Let's use 0/1 for easier indexing?
    # No, -1/1 is better for the product formula.
    # But 0/1 is easier for "is_mono" check: (s[a]==s[b]) & ...
    
    # Let's use 0/1.
    # Mono check: (s[a] == s[b]) & (s[b] == s[c]) & (s[c] == s[d])
    
    # Let's stick to the previous precomputation of neighbors for speed, but use a batch flip strategy.
    
    # Re-precompute neighbors as in the original, but we will use them to compute gains.
    # Actually, iterating over APs and updating gains is cleaner and likely fast enough.
    
    rng = np.random.default_rng()
    
    best_score = np.inf
    best_coloring = None
    
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    batch_size = 50  # Number of flips per batch step
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        
        # Calculate initial score
        ca = c[a_idx]
        cb = c[b_idx]
        cc = c[c_idx]
        cd = c[d_idx]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        current_score = np.sum(is_mono)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        T = T_start
        steps = 0
        max_steps = 2000 # Cap steps per restart
        
        while T > T_end and steps < max_steps:
            if time.time() > deadline - 2:
                break
                
            # Compute gains for all indices
            # gain[i] = sum over APs containing i of (current_mono - mono_if_flipped)
            # Note: current_mono is 1 or 0.
            # mono_if_flipped is 1 or 0.
            # gain[i] can be negative (flipping makes it worse).
            
            gains = np.zeros(N, dtype=np.int32)
            
            # Vectorized-ish loop over APs
            # We can't easily vectorize the "if flip a, is it mono" part without 4 separate vectors.
            # Let's do it with numpy operations on the AP arrays.
            
            # Current mono status
            mono = is_mono # 1 if mono, 0 else
            
            # If we flip a:
            # new_a = 1 - ca
            # new_mono_a = (new_a == cb) & (cb == cc) & (cc == cd)
            # This is equivalent to: (cb == cc) & (cc == cd) & (new_a == cb)
            # Which is: (cb == cc) & (cc == cd) & (1 - ca == cb)
            
            # Let's compute the "others equal" mask for each position.
            # For AP (a,b,c,d):
            # others_bcd_equal = (cb == cc) & (cc == cd)
            # others_acd_equal = (ca == cc) & (cc == cd)
            # others_abd_equal = (ca == cb) & (cb == cd)
            # others_abc_equal = (ca == cb) & (cb == cc)
            
            # If we flip a, the AP is mono iff:
            # 1. b,c,d are all equal (others_bcd_equal)
            # 2. new_a == b (1 - ca == cb)
            
            # So:
            # gain_a = mono - (others_bcd_equal & (1 - ca == cb))
            
            # Similarly for b, c, d.
            
            # Let's compute these masks.
            
            # Note: mono is already computed.
            
            # Masks for "others equal"
            # For a: others are b,c,d
            m_a = (cb == cc) & (cc == cd)
            # For b: others are a,c,d
            m_b = (ca == cc) & (cc == cd)
            # For c: others are a,b,d
            m_c = (ca == cb) & (cb == cd)
            # For d: others are a,b,c
            m_d = (ca == cb) & (cb == cc)
            
            # New mono status if flipped
            new_mono_a = m_a & (1 - ca == cb)
            new_mono_b = m_b & (1 - cb == ca)
            new_mono_c = m_c & (1 - cc == cb)
            new_mono_d = m_d & (1 - cd == cc)
            
            # Gains
            # gain[i] += mono - new_mono_i
            # We need to scatter-add these to the gains array.
            
            # np.add.at is slow but correct.
            # Given num_aps ~ 10k, and 4 updates per AP, 40k updates.
            # np.add.at(gains, a_idx, mono - new_mono_a)
            
            # Let's try it.
            np.add.at(gains, a_idx, mono.astype(np.int32) - new_mono_a.astype(np.int32))
            np.add.at(gains, b_idx, mono.astype(np.int32) - new_mono_b.astype(np.int32))
            np.add.at(gains, c_idx, mono.astype(np.int32) - new_mono_c.astype(np.int32))
            np.add.at(gains, d_idx, mono.astype(np.int32) - new_mono_d.astype(np.int32))
            
            # Select indices to flip
            # We want to flip indices with positive gain.
            # If T is high, we might also accept some negative gains.
            
            # Let's select the top K positive gains.
            # If fewer than K positive gains, fill with random ones (with some probability based on T)?
            # Or just flip the positive ones.
            
            pos_mask = gains > 0
            pos_indices = np.where(pos_mask)[0]
            
            if len(pos_indices) > 0:
                # Shuffle and take up to batch_size
                if len(pos_indices) > batch_size:
                    rng.shuffle(pos_indices)
                    pos_indices = pos_indices[:batch_size]
                
                # Flip these
                c[pos_indices] = 1 - c[pos_indices]
                # Update score
                current_score += np.sum(gains[pos_indices])
                
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = c.copy()
                    
                # Update is_mono?
                # We need to recalculate is_mono for the next step anyway.
                # But we can update it incrementally?
                # No, just recalculate. It's fast.
                
                ca = c[a_idx]
                cb = c[b_idx]
                cc = c[c_idx]
                cd = c[d_idx]
                is_mono = (ca == cb) & (cb == cc) & (cc == cd)
                
            else:
                # No positive gains. Stagnation?
                # Try random flips with probability exp(-delta/T)
                # Or just break if T is low.
                if T < 1.0:
                    break
                else:
                    # Pick a random index
                    i = rng.integers(0, N)
                    # Calculate its gain (we have it in gains[i])
                    delta = -gains[i] # gain is improvement, delta is cost
                    # Actually, if gain is negative, delta = -gain is positive cost.
                    # If gain is 0, delta is 0.
                    # If gain > 0, we would have flipped it. So here gain <= 0.
                    # So delta >= 0.
                    
                    if delta < 0:
                        # Should not happen if we picked from non-positive gains?
                        # Wait, gains[i] could be 0.
                        pass
                    
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        c[i] = 1 - c[i]
                        current_score += gains[i] # which is negative or 0
                        # Update score
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = c.copy()
                        
                        ca = c[a_idx]
                        cb = c[b_idx]
                        cc = c[c_idx]
                        cd = c[d_idx]
                        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
            
            T *= cooling_rate
            steps += 1

    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()
