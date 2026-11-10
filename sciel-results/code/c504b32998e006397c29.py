import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a <= 256
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            others = tuple(j for j in ap if j != idx)
            ap_list[idx].append((ap_id, others))

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta_single(colors, idx):
        """Calculate change in mono count if we flip colors[idx]."""
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def get_delta_segment(colors, start, length):
        """Calculate change in mono count if we flip a contiguous segment [start, start+length)."""
        if length == 1:
            return get_delta_single(colors, start)
        
        # We need to evaluate the change for each AP affected by the segment flip.
        # An AP is affected if any of its 4 indices fall within [start, start+length).
        # For each such AP, we check if it was mono before and if it is mono after.
        
        delta = 0
        affected_aps = set()
        
        # Collect all APs that intersect the segment
        for i in range(start, start + length):
            for ap_id, _ in ap_list[i]:
                affected_aps.add(ap_id)
        
        for ap_id in affected_aps:
            ap = aps[ap_id]
            # Check if this AP was monochromatic before
            was_mono = False
            c0 = colors[ap[0]]
            if c0 == colors[ap[1]] and c0 == colors[ap[2]] and c0 == colors[ap[3]]:
                was_mono = True
            
            # Check if this AP is monochromatic after flipping the segment
            # We don't actually flip the colors, we just simulate.
            # If an index is in the segment, its color is inverted.
            is_mono_after = False
            # Determine the color of the first element after flip
            # We can just check consistency of (color ^ flip_mask)
            
            # Optimization: Just check the 4 indices
            # Let's create a helper to get flipped color
            def get_flipped_color(idx):
                if start <= idx < start + length:
                    return 1 - colors[idx]
                else:
                    return colors[idx]
            
            c0_f = get_flipped_color(ap[0])
            if c0_f == get_flipped_color(ap[1]) and c0_f == get_flipped_color(ap[2]) and c0_f == get_flipped_color(ap[3]):
                is_mono_after = True
                
            if was_mono and not is_mono_after:
                delta -= 1
            elif not was_mono and is_mono_after:
                delta += 1
                
        return delta

    def flip_segment(colors, start, length):
        """Flip a contiguous segment of colors."""
        for i in range(start, start + length):
            colors[i] = 1 - colors[i]

    # Simulated Annealing
    best_count = float('inf')
    best_colors = None
    
    # Initial random state
    current_colors = [random.randint(0, 1) for _ in range(N)]
    current_count = count_mono(current_colors)
    
    if current_count < best_count:
        best_count = current_count
        best_colors = current_colors[:]

    T_start = 1.0
    T_end = 0.001
    # Number of steps to determine cooling schedule
    # We estimate steps per second. get_delta_segment is O(K) where K is number of affected APs.
    # Average K is small (~10-20). So we can do many steps.
    # Let's just run until deadline.
    
    steps = 0
    while time.time() < deadline:
        # Cooling schedule: exponential
        # We don't know total steps in advance, so use a fixed number of steps for T decay
        # Or just use a simple linear decay based on time?
        # Better: use a fixed number of iterations for the "annealing" part, then restart?
        # Or: standard SA with a large number of steps.
        
        # Let's use a fixed temperature schedule over a large number of steps, 
        # and restart if we get stuck or if we want to explore more.
        # Actually, a single long run is often better.
        
        # To manage T, we can calculate T based on the fraction of time elapsed?
        # No, standard SA is step-based. Let's assume we can do ~5000 steps/sec.
        # Total time ~145s. ~700,000 steps.
        # T = T_start * (T_end/T_start)^step
        # This is tricky without knowing total steps.
        
        # Alternative: Use a "restart" strategy.
        # Run SA for a fixed number of steps (e.g., 5000), then restart with new T or new init?
        # Or just keep T constant and rely on the segment flips to escape?
        # Let's use a dynamic T that decreases slowly.
        
        # Simple heuristic: T = max(T_end, T_start * exp(-steps / 10000))
        # This cools down over ~20k steps.
        # If we do 700k steps, we'll be at T_end for most of the time.
        # That's fine. It becomes local search eventually.
        
        T = max(T_end, T_start * math.exp(-steps / 10000.0))
        
        # Choose a move
        # 1. Single flip (probability 0.2)
        # 2. Short segment flip (length 2-5) (probability 0.3)
        # 3. Medium segment flip (length 6-20) (probability 0.3)
        # 4. Long segment flip (length 21-50) (probability 0.2)
        
        r = random.random()
        if r < 0.2:
            start = random.randint(0, N - 1)
            length = 1
        elif r < 0.5:
            length = random.randint(2, 5)
            start = random.randint(0, N - length)
        elif r < 0.8:
            length = random.randint(6, 20)
            start = random.randint(0, N - length)
        else:
            length = random.randint(21, 50)
            start = random.randint(0, N - length)
            
        delta = get_delta_segment(current_colors, start, length)
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            flip_segment(current_colors, start, length)
            current_count += delta
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current_colors[:]
                
        steps += 1
        
    if best_colors is None:
        best_colors = [random.randint(0, 1) for _ in range(N)]
        
    return best_colors
