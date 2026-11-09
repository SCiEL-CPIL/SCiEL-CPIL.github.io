import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    # Precompute all 4-APs: (i1, i2, i3, i4) 0-based
    # i1 = a-1, i2 = a+d-1, i3 = a+2d-1, i4 = a+3d-1
    # a >= 1, a+3d <= 128
    aps = []
    # For each a, d
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
            
    # For each index, list of AP indices it belongs to
    # ap_index -> (i1, i2, i3, i4)
    # We need to map index -> list of ap indices
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            index_to_aps[i].append(idx)
            
    def count_bad_aps(colors):
        bad_count = 0
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c1, c2, c3, c4}) < 3:
                bad_count += 1
        return bad_count
        
    def is_bad_ap(colors, ap):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4}) < 3
        
    def search(max_colors, time_limit):
        # Initialize randomly
        colors = [random.randint(0, max_colors - 1) for _ in range(N)]
        current_energy = count_bad_aps(colors)
        
        if current_energy == 0:
            return colors
            
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        start = time.time()
        
        while time.time() - start < time_limit and T > min_T:
            # Pick a random index
            i = random.randint(0, N - 1)
            old_color = colors[i]
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
                
            # Calculate delta energy
            delta = 0
            for ap_idx in index_to_aps[i]:
                ap = aps[ap_idx]
                was_bad = is_bad_ap(colors, ap)
                # Temporarily assign
                colors[i] = new_color
                is_bad_now = is_bad_ap(colors, ap)
                # Revert
                colors[i] = old_color
                
                if was_bad and not is_bad_now:
                    delta -= 1
                elif not was_bad and is_bad_now:
                    delta += 1
                    
            # Accept
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                colors[i] = new_color
                current_energy += delta
                if current_energy == 0:
                    return colors
                    
            T *= cooling
            
        return None

    # Try 3 colors first
    # Use multiple restarts if time permits
    best_res = None
    # We have ~85s. Let's use 40s for 3-color search.
    res_3 = search(3, 40.0)
    if res_3 is not None:
        return res_3
        
    # If 3 fails, try 4 colors
    # 4 colors should be easier.
    res_4 = search(4, 40.0)
    if res_4 is not None:
        return res_4
        
    # Fallback: Greedy 4 colors (seed logic)
    # Just return a valid 4-coloring if search fails
    # A simple valid 4-coloring:
    # We can just use the seed's greedy fallback
    colors = []
    # Re-use the seed's slow but robust fallback if needed
    # But we should have found one.
    # If not, return a default 4-coloring that is likely valid or just the seed's output.
    # Let's just run the seed's final fallback logic here to be safe.
    colors = []
    for x in range(1, 129):
        for color in range(4):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

