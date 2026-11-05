import time
import random

def construct():
    n = 256
    deadline = time.time() + 110  # Leave 10s buffer for 120s limit
    
    # Precompute all 4-APs
    # a, a+d, a+2d, a+3d < n
    # d >= 1
    # a >= 0
    # a + 3d <= n - 1 => a <= n - 1 - 3d
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute bit_aps: for each bit, which AP indices contain it
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for bit in ap:
            bit_aps[bit].append(idx)
    
    # Helper to count mono APs
    def count_mono(colors):
        cnt = 0
        for ap in ap_list:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                cnt += 1
        return cnt

    def local_search(colors):
        # colors is a list of 0/1
        deltas = [0] * n
        
        # Initialize deltas
        for idx, ap in enumerate(ap_list):
            b0, b1, b2, b3 = ap
            c0, c1, c2, c3 = colors[b0], colors[b1], colors[b2], colors[b3]
            current_mono = 1 if (c0 == c1 == c2 == c3) else 0
            
            # Gain for flipping b0
            # If b0 flips, new color is 1-c0
            # New mono is 1 if 1-c0 == c1 == c2 == c3
            new_mono = 1 if ((1 - c0) == c1 and c1 == c2 and c2 == c3) else 0
            deltas[b0] += new_mono - current_mono
            
            # Gain for flipping b1
            new_mono = 1 if (c0 == (1 - c1) and (1 - c1) == c2 and c2 == c3) else 0
            deltas[b1] += new_mono - current_mono
            
            # Gain for flipping b2
            new_mono = 1 if (c0 == c1 and c1 == (1 - c2) and (1 - c2) == c3) else 0
            deltas[b2] += new_mono - current_mono
            
            # Gain for flipping b3
            new_mono = 1 if (c0 == c1 and c1 == c2 and c2 == (1 - c3)) else 0
            deltas[b3] += new_mono - current_mono
        
        # Local search loop
        while True:
            # Find best flip
            min_delta = min(deltas)
            if min_delta >= 0:
                break
            
            # Find index of min_delta
            # If multiple, pick random among them? Or first?
            # Let's pick first for determinism, or random to escape symmetry
            candidates = [i for i, d in enumerate(deltas) if d == min_delta]
            k = random.choice(candidates)
            
            # Flip k
            colors[k] = 1 - colors[k]
            
            # Update deltas for neighbors
            # For each AP containing k, update deltas of other bits in that AP
            for p_idx in bit_aps[k]:
                ap = ap_list[p_idx]
                # Identify the other 3 bits
                others = [b for b in ap if b != k]
                
                # Get current colors (after flip)
                c_k = colors[k]
                c_others = [colors[b] for b in others]
                
                # Calculate current_mono (with new k)
                all_c = [c_k] + c_others
                current_mono = 1 if all(x == all_c[0] for x in all_c) else 0
                
                # Calculate T_old for each other bit
                # T_old depends on state BEFORE flip of k
                # State before: k was 1 - c_k
                old_c_k = 1 - c_k
                old_all_c = [old_c_k] + c_others
                old_current_mono = 1 if all(x == old_all_c[0] for x in old_all_c) else 0
                
                for j_idx, j in enumerate(others):
                    # T_new: gain if we flip j NOW
                    # New color of j would be 1 - colors[j]
                    new_c_j = 1 - colors[j]
                    # Check if AP becomes mono if j flips
                    # Colors: c_k, c_others[0], c_others[1], c_others[2]
                    # But j is one of the others.
                    # Let's construct the hypothetical colors
                    hyp_c = [c_k]
                    for idx_o, c_o in enumerate(c_others):
                        if idx_o == j_idx:
                            hyp_c.append(new_c_j)
                        else:
                            hyp_c.append(c_o)
                    
                    new_mono = 1 if all(x == hyp_c[0] for x in hyp_c) else 0
                    T_new = new_mono - current_mono
                    
                    # T_old: gain if we flipped j BEFORE k flipped
                    # State before: k was old_c_k
                    # Hypothetical: j flips, k is old_c_k
                    hyp_c_old = [old_c_k]
                    for idx_o, c_o in enumerate(c_others):
                        if idx_o == j_idx:
                            hyp_c_old.append(new_c_j) # Wait, new_c_j is 1 - colors[j]. colors[j] hasn't changed.
                                                      # So this is correct: the color j WOULD take.
                        else:
                            hyp_c_old.append(c_o)
                    
                    old_new_mono = 1 if all(x == hyp_c_old[0] for x in hyp_c_old) else 0
                    T_old = old_new_mono - old_current_mono
                    
                    deltas[j] += T_new - T_old
            
            # Update delta for k itself?
            # The delta for k is the gain of flipping k.
            # We just flipped k. So the "gain of flipping k" is now the gain of flipping it back.
            # We need to update deltas[k].
            # deltas[k] was min_delta.
            # After flip, the gain of flipping k back is -min_delta?
            # Not exactly, because the environment changed? No, only k changed.
            # The gain of flipping k back is exactly the negative of the gain of flipping k forward?
            # Let's check.
            # Gain(k) = Score(after) - Score(before).
            # Gain(k back) = Score(before) - Score(after) = -Gain(k).
            # So deltas[k] should become -min_delta.
            # But wait, deltas[k] also depends on other bits? No, deltas[k] is the sum of changes in APs containing k.
            # Since only k changed, the change in APs containing k is the only thing that matters for the "gain of flipping k".
            # The "gain of flipping k" is calculated based on the CURRENT state of others.
            # So yes, deltas[k] = -min_delta.
            deltas[k] = -min_delta

        return colors

    best_score = float('inf')
    best_colors = None
    
    # Initializations
    # 1. Random
    # 2. Thue-Morse
    # 3. Alternating
    # 4. Blocks
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def alternating(n):
        return [i % 2 for i in range(n)]
        
    def blocks(n, size=2):
        return [(i // size) % 2 for i in range(n)]

    init_strategies = [
        lambda: [random.randint(0, 1) for _ in range(n)],
        lambda: thue_morse(n),
        lambda: alternating(n),
        lambda: blocks(n, 2),
        lambda: blocks(n, 4),
        lambda: blocks(n, 8)
    ]
    
    while time.time() < deadline:
        # Pick a random strategy
        init_fn = random.choice(init_strategies)
        colors = init_fn()
        
        # Run local search
        colors = local_search(colors)
        
        score = count_mono(colors)
        if score < best_score:
            best_score = score
            best_colors = colors[:]
            
        # If we found 0, we can stop
        if best_score == 0:
            break

    if best_colors is None:
        return [0] * n
        
    return best_colors
