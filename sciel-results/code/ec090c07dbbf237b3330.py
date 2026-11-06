"""
Combined approach: fast bitmask greedy construction (inspiration 1's structural idea)
plus a dedicated color-elimination local search (refinement from current program).
"""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    N = 128

    # Precompute all non-constant 4-term APs
    ap_list = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                ap_list.append((a, b, c, e))
    num_aps = len(ap_list)

    # Precompute for each position, which APs it participates in and its role
    # position_aps[i] = list of (ap_index, role) where role is 0,1,2,3 for a,b,c,e
    position_aps = [[] for _ in range(N)]
    for idx, (a, b, c, e) in enumerate(ap_list):
        position_aps[a].append((idx, 0))
        position_aps[b].append((idx, 1))
        position_aps[c].append((idx, 2))
        position_aps[e].append((idx, 3))

    def validate(colors):
        """Check every 4-AP has >= 3 distinct colors."""
        for a, b, c, e in ap_list:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def greedy_construct(K, rng=None):
        """
        Construct a valid K-coloring using forward greedy with bitmask.
        Returns None if stuck.
        """
        if rng is None:
            rng = random.Random(0)
        
        colors = [-1] * N
        # colors[i] = -1 means unassigned
        
        for i in range(N):
            # Determine forbidden colors for position i
            # A color c is forbidden if assigning colors[i]=c would make any AP containing i violate the constraint
            # given the already-assigned colors of the other 3 positions in that AP.
            
            # For each AP containing i, check the other 3 positions.
            # If all 3 others are assigned:
            #   Let S = set of their colors.
            #   We need |S ∪ {c}| >= 3.
            #   If |S| >= 3: no restriction on c.
            #   If |S| == 2: c must not be in S. (Because if c in S, |S ∪ {c}| = 2 < 3)
            #   If |S| == 1: c must not be in S? No. If S={x}, and c=x, |S ∪ {c}|=1. If c!=x, |S ∪ {c}|=2. 
            #   Wait. If S has size 1, say {x}. Then {x, c} has size 1 if c=x, and 2 if c!=x.
            #   In both cases size < 3. So if |S|==1, NO color c is valid?
            #   This implies that if 3 positions in an AP are all the same color, the 4th position cannot be assigned any color.
            #   This means the first 3 assignments were bad. 
            #   However, in a forward greedy, when we assign position i, the "other 3" might include future positions (unassigned).
            #   Let's re-evaluate.
            
            # The constraint is on the FULL AP. 
            # When assigning position i, some other positions in the AP might be unassigned (future).
            # We only know the colors of the *past* positions in the AP.
            # This makes simple forward greedy tricky because we can't fully validate until all 4 are set.
            
            # Alternative: Backtracking or Constraint Programming is safer but slower.
            # However, the inspiration says "preventing monochromatic 3-term APs".
            # Actually, if we ensure that no 3 positions in any 4-AP are the same color *at the time they are set*, 
            # does that help?
            
            # Let's try a different greedy strategy:
            # Assign colors 0..K-1.
            # For position i, try each color c in 0..K-1.
            # Check if assigning c to i creates any *immediate* violation with *already assigned* neighbors.
            # An immediate violation occurs if there is an AP (a,b,c,e) containing i, where the other 3 are assigned,
            # and the set of their colors plus c has size < 3.
            # This only checks APs where i is the LAST position to be assigned (in index order? No, in assignment order).
            # Since we assign in order 0..N-1, an AP (a,b,c,e) with a<b<c<e is fully determined when we assign e.
            # But i could be a, b, or c. If i=a, and b,c,e are unassigned, we can't check.
            # So forward greedy only checks constraints where i is the largest index in the AP.
            # This is insufficient.
            
            # Better approach: Use the precomputed AP list.
            # For position i, look at all APs containing i.
            # For each AP, let Others be the other 3 indices.
            # If all Others are assigned (< i), we can check.
            # If some Others are unassigned (> i), we cannot check yet.
            # So we only restrict based on "backward" APs (where i is the max index).
            # This might lead to dead ends later.
            
            # To make this robust, we use backtracking with limited depth or restarts.
            # Given the 60s limit, let's try a randomized greedy with restarts.
            
            forbidden = set()
            for idx, role in position_aps[i]:
                a, b, c, e = ap_list[idx]
                others = []
                if role == 0: others = [b, c, e]
                elif role == 1: others = [a, c, e]
                elif role == 2: others = [a, b, e]
                else: others = [a, b, c]
                
                # Only check if all others are assigned
                if all(colors[other] != -1 for other in others):
                    s = {colors[other] for other in others}
                    if len(s) == 2:
                        forbidden |= s
                    elif len(s) == 1:
                        # If all 3 others are same color X, then {X, c} is always size 1 or 2.
                        # So no color is valid. This is a dead end.
                        return None
                    # if len(s) >= 3, no restriction
                
            # Pick a color not in forbidden
            available = [c for c in range(K) if c not in forbidden]
            if not available:
                return None
            
            if rng.random() < 0.8:
                colors[i] = available[0]
            else:
                colors[i] = rng.choice(available)
        
        # Final validation
        if validate(colors):
            return colors
        return None

    def local_search_reduce(colors, K, max_iters=5000):
        """
        Try to reduce the number of colors used by recoloring.
        """
        colors = colors[:]
        n_colors = count_colors(colors)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to eliminate the least used color
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            if not color_counts:
                break
                
            # Pick color to eliminate (least frequent)
            target_color = min(color_counts, key=color_counts.get)
            if color_counts[target_color] == 0:
                continue
                
            positions = [i for i in range(N) if colors[i] == target_color]
            
            # Try to recolor each position to a different color
            success = True
            for pos in positions:
                found = False
                for nc in range(K):
                    if nc == target_color:
                        continue
                    old = colors[pos]
                    colors[pos] = nc
                    if validate(colors):
                        found = True
                        break
                    else:
                        colors[pos] = old
                if not found:
                    success = False
                    break
            
            if success:
                # We eliminated target_color
                pass
            else:
                # Try random single recoloring to escape local minima
                pos = random.randint(0, N-1)
                old = colors[pos]
                nc = random.randint(0, K-1)
                if nc != old:
                    colors[pos] = nc
                    if validate(colors):
                        pass
                    else:
                        colors[pos] = old
        
        return colors

    # --- Main Loop ---
    best_colors = None
    best_count = N + 1
    
    # Try K=3 first
    for trial in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(trial * 13 + 7)
        res = greedy_construct(3, rng)
        if res is not None:
            res, k = relabel(res)
            if k < best_count:
                best_colors = res
                best_count = k
                if best_count == 3:
                    # Can't do better than 3 (since 2 colors always allow a 4-AP with <3 colors? 
                    # Actually, with 2 colors, can we avoid 4-APs with <3 colors?
                    # No, by van der Waerden or similar, or simple pigeonhole.
                    # A 4-AP with 2 colors must have pattern AABA, ABAA, ABBA, AABB, etc.
                    # It's very hard to avoid. Let's assume 3 is the minimum.)
                    break
    
    if best_colors is None:
        # Try K=4
        for trial in range(200):
            if time.time() > deadline:
                break
            rng = random.Random(trial * 31 + 11)
            res = greedy_construct(4, rng)
            if res is not None:
                res, k = relabel(res)
                if k < best_count:
                    best_colors = res
                    best_count = k
                    if best_count == 3:
                        break

    if best_colors is None:
        # Fallback: Known valid 4-coloring
        # Try a simple pattern
        best_colors = [i % 4 for i in range(N)]
        if not validate(best_colors):
            # Random search fallback
            rng = random.Random(999)
            while time.time() < deadline - 5:
                c = [rng.randint(0, 3) for _ in range(N)]
                if validate(c):
                    best_colors = c
                    break
            else:
                # Last resort: all distinct colors (valid but high count)
                best_colors = list(range(N))
        
        best_colors, best_count = relabel(best_colors)
    else:
        # Try to reduce further if K>3
        if best_count > 3:
            reduced = local_search_reduce(best_colors, best_count)
            reduced, k = relabel(reduced)
            if k < best_count:
                best_colors = reduced
                best_count = k

    return best_colors
# EVOLVE-BLOCK-END
