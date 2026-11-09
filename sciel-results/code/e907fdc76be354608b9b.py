import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for safety
    
    n = 128
    # Precompute all non-constant four-term progressions
    # (a, a+d, a+2d, a+3d) with 0-based indices
    aps = []
    ap_indices = [[] for _ in range(n)]
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            idx = len(aps)
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            aps.append((a, b, c, dd))
            ap_indices[a].append(idx)
            ap_indices[b].append(idx)
            ap_indices[c].append(idx)
            ap_indices[dd].append(idx)
    
    # Helper to check if an AP is bad (uses < 3 colors)
    def is_bad(c, ap_idx):
        a, b, c_idx, d_idx = aps[ap_idx]
        colors = {c[a], c[b], c[c_idx], c[d_idx]}
        return len(colors) < 3

    # Try to find a valid coloring with K colors
    for K in range(4, 10):
        if time.time() > deadline:
            break
        
        found = None
        # Multiple restarts for local search
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Initialize random coloring
            c = [random.randint(0, K - 1) for _ in range(n)]
            
            # Initialize state
            ap_state = [False] * len(aps)
            bad_set = set()
            for i in range(len(aps)):
                if is_bad(c, i):
                    ap_state[i] = True
                    bad_set.add(i)
            
            if not bad_set:
                found = c
                break
            
            # Local search: Hill Climbing with random restarts
            # We allow some "downhill" moves if stuck? 
            # For now, simple greedy: always fix a bad AP.
            for step in range(20000):
                if time.time() > deadline:
                    break
                
                if not bad_set:
                    found = c
                    break
                
                # Pick a random bad AP
                ap_idx = random.choice(list(bad_set))
                a, b, c_idx, d_idx = aps[ap_idx]
                
                # Pick a random position in this AP to change
                pos_idx = random.randint(0, 3)
                if pos_idx == 0:
                    pos = a
                elif pos_idx == 1:
                    pos = b
                elif pos_idx == 2:
                    pos = c_idx
                else:
                    pos = d_idx
                
                old_color = c[pos]
                # Try a new color
                # To avoid infinite loops, ensure we change to a different color
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                c[pos] = new_color
                
                # Update state for all APs involving pos
                for idx in ap_indices[pos]:
                    was_bad = ap_state[idx]
                    is_bad_now = is_bad(c, idx)
                    ap_state[idx] = is_bad_now
                    
                    if was_bad and not is_bad_now:
                        bad_set.discard(idx)
                    elif not was_bad and is_bad_now:
                        bad_set.add(idx)
                    
                    # Early exit if we fixed the one we picked?
                    # Not strictly necessary, but good for speed
                    if idx == ap_idx and not is_bad_now:
                        # We fixed the target AP, but might have broken others
                        # We don't break the inner loop, we just continue to next step
                        pass

            if found:
                break
        
        if found:
            return found

    # Fallback: Seed program logic (greedy)
    # This guarantees a valid solution, though potentially with more colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            # Check constraints with previous elements
            # 1. No 4-AP ending at x with < 3 colors
            # 2. No 3-AP ending at x where the 4th would be forced? 
            # The seed logic checks:
            # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
            # This ensures that if we pick 'color', no 4-AP ending at x has <3 colors.
            # It also checks:
            # all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This ensures we don't create a pattern A,B,A,B or A,A,B,B etc. that *would* be bad if extended?
            # Actually, the second condition is to prevent "pre-bad" states for future extensions?
            # Let's just use the first condition which is sufficient for validity of the prefix.
            # Wait, the seed's second condition:
            # any(colors[x-1-d] != color for j in (1,2)) -> colors[x-1-d] != color OR colors[x-1-2d] != color
            # This prevents c[x] == c[x-d] == c[x-2d]? No.
            # If c[x] == color, and c[x-d] == color and c[x-2d] == color, then any(...) is False.
            # So it prevents 3 consecutive terms in an AP from having the same color?
            # If c[x]=A, c[x-d]=A, c[x-2d]=A, then the 4-AP c[x-3d], A, A, A uses at most 2 colors if c[x-3d] is A or B.
            # If c[x-3d] is C, then colors are {C, A}. Size 2. Bad.
            # So preventing 3 identical in a row helps.
            
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    continue
                c0 = colors[x - 1 - 3 * d]
                c1 = colors[x - 1 - 2 * d]
                c2 = colors[x - 1 - d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check the "no 3 identical" heuristic from seed?
            # The seed had: all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This means: for all d, NOT (colors[x-1-d]==color AND colors[x-1-2*d]==color)
            # i.e., c[x] != c[x-d] != c[x-2d] is not required, but c[x] != c[x-d] OR c[x] != c[x-2d] is required.
            # Which is equivalent to: NOT (c[x]==c[x-d] AND c[x]==c[x-2d]).
            # So it prevents c[x], c[x-d], c[x-2d] being all equal.
            for d in range(1, (x - 1) // 2 + 1):
                if x - 1 - 2 * d < 0:
                    continue
                if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                    valid = False
                    break
            
            if valid:
                colors.append(color)
                break
        else:
            # Should not happen if 128 colors are available
            colors.append(0)
            
    return colors
