import time
import random

def construct():
    N = 128
    # Precompute all 4-APs (0-indexed)
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    # Precompute the indices of the 4 positions in each AP relative to the AP tuple
    # aps[i] is a tuple (p0, p1, p2, p3)
    
    def search(K, time_limit):
        start_time = time.time()
        # Initial random coloring
        coloring = [random.randint(0, K - 1) for _ in range(N)]
        
        # Precompute bad status for each AP
        is_bad = [False] * num_aps
        bad_count = 0
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                is_bad[idx] = True
                bad_count += 1
                
        if bad_count == 0:
            return coloring
            
        best_count = bad_count
        best_coloring = coloring[:]
        T = 10.0
        
        while time.time() - start_time < time_limit:
            # Pick a random position
            i = random.randint(0, N - 1)
            old_c = coloring[i]
            
            # Pick a new color
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            delta = 0
            # Calculate delta by checking only affected APs
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                was_bad = is_bad[ap_idx]
                
                # Get current colors
                c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                
                # Apply new color to the specific position in the AP
                if ap[0] == i:
                    c0 = new_c
                elif ap[1] == i:
                    c1 = new_c
                elif ap[2] == i:
                    c2 = new_c
                elif ap[3] == i:
                    c3 = new_c
                    
                new_distinct = len({c0, c1, c2, c3})
                is_now_bad = new_distinct < 3
                
                if was_bad and not is_now_bad:
                    delta -= 1
                elif not was_bad and is_now_bad:
                    delta += 1
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Simulated Annealing probability
                if random.random() < (1.0 / (1.0 + delta * T)):
                    accept = True
            
            if accept:
                coloring[i] = new_c
                # Update is_bad and bad_count
                for ap_idx in pos_to_aps[i]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    is_bad[ap_idx] = len({c0, c1, c2, c3}) < 3
                bad_count += delta
                
                if bad_count < best_count:
                    best_count = bad_count
                    best_coloring = coloring[:]
                    if best_count == 0:
                        return best_coloring
            
            # Cool down
            T *= 0.99
            if T < 0.01:
                T = 0.01

        return best_coloring if best_count < N else None # Return best found, even if not perfect? 
        # The task requires valid coloring. If we don't find one, we must return a valid one.
        # So we should not return an invalid one.
        
    # Try 3 colors
    # It is known that 3 colors are not enough for N=128? 
    # Ruzsa-Szemeredi-Trotter theorem bounds? 
    # Actually, for 4-APs, the minimum number of colors is related to the independence number.
    # Let's try 3, then 4.
    
    # Use a generous time limit for 3 colors? 
    # If 3 colors is impossible, we waste time. 
    # Let's try 3 colors for 5s.
    result = search(3, 5.0)
    if result is not None:
        # Verify it's valid
        valid = True
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                valid = False
                break
        if valid:
            return result
        else:
            # It wasn't valid, discard
            pass
            
    # Try 4 colors
    result = search(4, 30.0)
    if result is not None:
        valid = True
        for ap in aps:
            if len({result[ap[0]], result[ap[1]], result[ap[2]], result[ap[3]]}) < 3:
                valid = False
                break
        if valid:
            return result

    # Fallback: Greedy with backtracking is complex. 
    # Simple fallback: Use 128 colors (identity coloring). 
    # Identity coloring: color[i] = i.
    # Any 4-AP has 4 distinct positions, so 4 distinct colors.
    # This is always valid.
    # Score will be 128, which is bad, but it's valid.
    # We can try to improve this quickly with a simple greedy that picks smallest available color.
    
    colors = [-1] * N
    for i in range(N):
        used_colors = set()
        # Check all APs containing i
        for ap_idx in pos_to_aps[i]:
            ap = aps[ap_idx]
            for pos in ap:
                if pos != i and colors[pos] != -1:
                    used_colors.add(colors[pos])
        
        # We need to pick a color c such that for every AP containing i, 
        # the set of colors in that AP has size >= 3.
        # This is a constraint satisfaction problem.
        # For a simple fallback, let's just try colors 0..127 and check validity locally.
        
        found = False
        for c in range(128):
            valid = True
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                # Get colors of other 3 positions
                cs = []
                for pos in ap:
                    if pos == i:
                        cs.append(c)
                    else:
                        if colors[pos] == -1:
                            # Future position, unknown. 
                            # We can't check validity now. 
                            # This greedy approach fails if it depends on future choices.
                            # However, if we pick distinct colors for all positions, it works.
                            # So let's just assign i to i.
                            pass
                        else:
                            cs.append(colors[pos])
                # If all 4 are known, check
                # If some are unknown, we can't guarantee.
                # So this greedy is risky.
                
            # Let's stick to Identity Coloring as a guaranteed valid fallback.
            # Or, a better fallback: Randomize and run search for a very short time, 
            # and if it fails, return Identity.
            
        # Let's just return Identity Coloring. It's valid.
        # score = 128.
        # The search for 4 colors should have succeeded.
        
    return list(range(N))
