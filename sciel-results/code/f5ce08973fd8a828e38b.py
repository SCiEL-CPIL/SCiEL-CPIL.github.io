"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

N = 256

def count_mono_4ap_fast(coloring):
    """Faster count using numpy vectorization per d."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N):
        max_a = N - 3*d
        if max_a < 1:
            break
        # Extract the four sequences
        seq1 = c[:max_a]           # a-1 for a=1..max_a
        seq2 = c[d:d+max_a]
        seq3 = c[2*d:2*d+max_a]
        seq4 = c[3*d:3*d+max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return int(count)

def flip_delta(coloring, pos):
    """
    Calculate the change in the number of monochromatic 4-APs when flipping coloring[pos].
    Returns (delta, affected_aps_count) where delta is the change in total count.
    
    A 4-AP (a, a+d, a+2d, a+3d) is affected if pos is one of its elements.
    Before flip: all 4 same -> contributes 1 to count.
    After flip: 3 same, 1 different -> contributes 0.
    So if AP was mono, delta -= 1.
    
    Before flip: not all same -> contributes 0.
    After flip: 
      - If the 3 others were same color as the flipped one (before flip), then after flip 
        the 3 others are same color as each other, but different from flipped. 
        Wait, let's trace carefully.
        
    Let c[i] be the color before flip. We flip at pos p.
    Consider an AP containing p. Let the other 3 positions be q1, q2, q3.
    Let their colors be c[q1], c[q2], c[q3].
    Let c[p] be the color of p before flip.
    
    Case 1: c[q1] == c[q2] == c[q3] == c[p].
      Before: Mono (count 1).
      After: c[p] becomes 1-c[p]. The others are still c[p]. So not all same.
      Delta: -1.
      
    Case 2: c[q1] == c[q2] == c[q3] != c[p].
      Before: Not mono (count 0).
      After: c[p] becomes 1-c[p] = c[q1]. So all 4 are now same.
      Delta: +1.
      
    Case 3: The other 3 are not all same.
      Before: Not mono (count 0).
      After: c[p] flips. Can the 4 become all same?
      For the 4 to be all same, q1,q2,q3 must all be same, and equal to new c[p].
      But we assumed q1,q2,q3 are not all same. So they cannot all equal new c[p].
      Thus, after flip, they are still not all same.
      Delta: 0.

    So we only need to check APs where the other 3 elements are all the same color.
    """
    c = coloring
    delta = 0
    p = pos
    cp = c[p]
    
    # For each possible role of p in a 4-AP
    # Roles:
    # 1. p is a (1st term): p, p+d, p+2d, p+3d. Need p+3d <= N.
    # 2. p is a+d (2nd term): p-d, p, p+d, p+2d. Need p-d >= 1 and p+2d <= N.
    # 3. p is a+2d (3rd term): p-2d, p-d, p, p+d. Need p-2d >= 1 and p+d <= N.
    # 4. p is a+3d (4th term): p-3d, p-2d, p-d, p. Need p-3d >= 1.
    
    # Note: coloring is 0-indexed in our array, but problem is 1..256.
    # Let's use 1-indexed logic for clarity in bounds, then map to 0-indexed array.
    # Array index i corresponds to integer i+1.
    # pos is 0-indexed. Integer value is pos+1.
    
    # Role 1: p is 1st term (index i, value i+1)
    # AP: i, i+d, i+2d, i+3d
    # Constraints: i+3d < N
    # Other 3: i+d, i+2d, i+3d
    for d in range(1, (N - p - 1) // 3 + 1):
        i1 = p + d
        i2 = p + 2*d
        i3 = p + 3*d
        if i3 >= N: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    # Role 2: p is 2nd term (index i+d, value i+d+1)
    # AP: i, i+d, i+2d, i+3d where i+d = p => i = p-d
    # Other 3: p-d, p+d, p+2d
    for d in range(1, min(p, (N - p - 1) // 2) + 1):
        i1 = p - d
        i2 = p + d
        i3 = p + 2*d
        if i1 < 0 or i3 >= N: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    # Role 3: p is 3rd term (index i+2d, value i+2d+1)
    # AP: i, i+d, i+2d, i+3d where i+2d = p => i = p-2d
    # Other 3: p-2d, p-d, p+d
    for d in range(1, min(p // 2, N - p - 1) + 1):
        i1 = p - 2*d
        i2 = p - d
        i3 = p + d
        if i1 < 0 or i3 >= N: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    # Role 4: p is 4th term (index i+3d, value i+3d+1)
    # AP: i, i+d, i+2d, i+3d where i+3d = p => i = p-3d
    # Other 3: p-3d, p-2d, p-d
    for d in range(1, p // 3 + 1):
        i1 = p - 3*d
        i2 = p - 2*d
        i3 = p - d
        if i1 < 0: break
        if c[i1] == c[i2] == c[i3]:
            if c[i1] == cp:
                delta -= 1
            else:
                delta += 1

    return delta

def local_search_incremental(coloring, deadline, max_passes=500):
    """
    Perform local search using incremental delta updates.
    coloring is a list of 0s and 1s.
    """
    c = coloring[:]
    current_score = count_mono_4ap_fast(c)
    best_score = current_score
    best_c = c[:]
    
    # Precompute a list of all positions to shuffle? Or just iterate?
    # Iterating sequentially might have bias. Shuffling is better.
    
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline - 2:
        improved = False
        passes += 1
        # Randomize order to avoid bias
        order = list(range(N))
        random.shuffle(order)
        
        for pos in order:
            if time.time() > deadline - 2:
                return best_c, best_score
            
            delta = flip_delta(c, pos)
            if delta < 0:
                c[pos] = 1 - c[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c[:]
                improved = True
    
    return best_c, best_score

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Generate diverse starting points
    starting_points = []
    
    # 1. Thue-Morse sequence (optimal for some AP problems, good baseline)
    # t(n) = popcount(n) % 2
    tm = [bin(i).count('1') % 2 for i in range(N)]
    starting_points.append(tm)
    
    # 2. Random
    for _ in range(20):
        starting_points.append([random.randint(0, 1) for _ in range(N)])
        
    # 3. Block patterns
    for block_size in [2, 4, 8, 16, 32, 64]:
        starting_points.append([(i // block_size) % 2 for i in range(N)])
        
    # 4. Alternating
    starting_points.append([i % 2 for i in range(N)])
    
    # 5. Greedy construction (fast version)
    # Greedy: pick color that minimizes new mono APs
    greedy_c = [-1] * N
    for j in range(N):
        # Count potential APs ending at j
        count0 = 0
        count1 = 0
        # Check APs where j is the last term
        for d in range(1, (j // 3) + 1):
            a = j - 3*d
            if a < 0: break
            # AP: a, a+d, a+2d, a+3d=j
            # Other 3: a, a+d, a+2d
            c1 = greedy_c[a]
            c2 = greedy_c[a+d]
            c3 = greedy_c[a+2*d]
            if c1 != -1 and c2 != -1 and c3 != -1:
                if c1 == c2 == c3:
                    if c1 == 0:
                        count0 += 1
                    else:
                        count1 += 1
        # Also check APs where j is not the last term?
        # Greedy usually only considers completed APs or those that will be completed later.
        # To be safe, we can just count all APs containing j that are fully within [0, j].
        # But APs extending beyond j are not yet "mono" because future colors are unknown.
        # So minimizing completed APs is a reasonable heuristic.
        
        if count0 < count1:
            greedy_c[j] = 0
        elif count1 < count0:
            greedy_c[j] = 1
        else:
            greedy_c[j] = random.randint(0, 1)
    starting_points.append(greedy_c)

    # Run local search on each starting point, tracking the best
    best_overall_c = None
    best_overall_score = float('inf')
    
    # We can run these in parallel? No, Python GIL.
    # But we can interleave them or just run sequentially.
    # Since local search is fast (O(N) per flip, ~1000 flips per pass, ~10 passes per start),
    # and we have ~30 starts, it should be fast enough.
    
    # To maximize diversity and avoid getting stuck in the same local minima,
    # we will run a limited number of passes for each, then pick the best.
    
    for idx, sp in enumerate(starting_points):
        if time.time() > deadline - 10:
            break
            
        # Run local search
        # Allocate some time for each
        # If we have 30 starts, and 140s total, ~4.6s each.
        # But local search is very fast.
        c_res, score_res = local_search_incremental(sp, deadline - 5, max_passes=100)
        
        if score_res < best_overall_score:
            best_overall_score = score_res
            best_overall_c = c_res
            
        # If we have time, do a second round of deeper search on the current best
        # But first, let's finish the first round for all starts.
        
    # Second phase: Deep dive on the best found
    if best_overall_c is not None and time.time() < deadline - 10:
        # Run more passes on the best
        final_c, final_score = local_search_incremental(best_overall_c, deadline - 2, max_passes=1000)
        if final_score < best_overall_score:
            best_overall_c = final_c
            best_overall_score = final_score
            
    # Third phase: If still time, try simulated annealing on the best
    if time.time() < deadline - 20:
        current = best_overall_c[:]
        current_score = best_overall_score
        T = 5.0
        alpha = 0.999
        while time.time() < deadline - 2 and T > 0.01:
            pos = random.randint(0, N - 1)
            delta = flip_delta(current, pos)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current[pos] = 1 - current[pos]
                current_score += delta
                if current_score < best_overall_score:
                    best_overall_score = current_score
                    best_overall_c = current[:]
            T *= alpha

    return best_overall_c
# EVOLVE-BLOCK-END
