import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # 1. Precompute all 4-APs in 0..255 (representing 1..256)
    # AP is (a, a+d, a+2d, a+3d) with a+3d < n
    aps = []
    # d goes up to (n-1)//3
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Precompute which APs contain each index
    # bit_aps[i] = list of indices in 'aps' that contain index i
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for bit in ap:
            bit_aps[bit].append(idx)

    # Helper to get the other 3 indices in an AP given one index
    # This is slightly inefficient if done naively, but we can just check the tuple.
    # For speed, we can precompute for each (ap_idx, pos_in_ap) the other 3 indices.
    # But since we just need to check if all 4 are same, we can do:
    # For an AP (i0, i1, i2, i3), if we flip i_k, we check if c[i0]==c[i1]==c[i2]==c[i3].
    
    # 2. Initialize with Thue-Morse sequence
    # Thue-Morse: t(i) = popcount(i) % 2
    def popcount(x):
        return bin(x).count('1')
        
    color = [popcount(i) % 2 for i in range(n)]
    
    # Calculate initial cost
    def calc_cost(c):
        cost = 0
        for ap in aps:
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                cost += 1
        return cost

    current_cost = calc_cost(color)
    best_color = color[:]
    best_cost = current_cost
    
    # 3. Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We want to allow many moves. 
    # Each move evaluation is O(len(bit_aps[k])) which is small.
    # Let's run until deadline.
    
    while time.time() < deadline:
        if T < min_T:
            # If temperature is very low, we can switch to pure hill climbing 
            # or just break if no improvements for a while. 
            # For simplicity, keep going but acceptance is rare for bad moves.
            pass
            
        # Pick a random index to flip
        k = random.randint(0, n - 1)
        
        # Calculate delta cost
        # We need to check all APs containing k
        delta = 0
        ap_indices = bit_aps[k]
        
        # For each AP containing k, check if it was mono before and after
        for ap_idx in ap_indices:
            ap = aps[ap_idx]
            # Check current state (before flip)
            # All 4 elements must be equal to color[k] for it to be mono
            # Since we are about to flip k, the "after" state will have color[k] inverted.
            
            # Current colors of the other 3
            other_indices = [x for x in ap if x != k]
            
            # Before flip:
            # Mono if color[k] == color[o1] == color[o2] == color[o3]
            is_mono_before = (color[k] == color[other_indices[0]] == color[other_indices[1]] == color[other_indices[2]])
            
            # After flip:
            # New color is 1 - color[k]
            # Mono if (1-color[k]) == color[o1] == color[o2] == color[o3]
            # This is equivalent to: color[o1] == color[o2] == color[o3] == (1-color[k])
            is_mono_after = (color[other_indices[0]] == color[other_indices[1]] == color[other_indices[2]] == (1 - color[k]))
            
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
            # else no change
            
        # Acceptance criterion
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            # Accept
            color[k] = 1 - color[k]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_color = color[:]
        
        # Cool down
        T *= cooling_rate
        
    return best_color

# EVOLVE-BLOCK-END
