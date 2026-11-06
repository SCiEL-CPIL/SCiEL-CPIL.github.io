"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    n = 128

    # Precompute all 4-term APs (a, a+d, a+2d, a+3d) with 0-indexed positions
    all_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Precompute for each position, the list of AP indices that contain it
    # and the index of that position within the AP (0,1,2,3)
    pos_ap_index = [[] for _ in range(n)]
    for ap_idx, (a, a1, a2, a3) in enumerate(all_aps):
        pos_ap_index[a].append((ap_idx, 0))
        pos_ap_index[a1].append((ap_idx, 1))
        pos_ap_index[a2].append((ap_idx, 2))
        pos_ap_index[a3].append((ap_idx, 3))

    def count_violations(colors):
        """Count the number of 4-term APs with fewer than 3 distinct colors."""
        violations = 0
        for (a, a1, a2, a3) in all_aps:
            if len({colors[a], colors[a1], colors[a2], colors[a3]}) < 3:
                violations += 1
        return violations

    def get_violating_aps(colors):
        """Return list of violating AP indices."""
        violating = []
        for ap_idx, (a, a1, a2, a3) in enumerate(all_aps):
            if len({colors[a], colors[a1], colors[a2], colors[a3]}) < 3:
                violating.append(ap_idx)
        return violating

    def greedy_init(num_colors, seed=0):
        """Greedy construction with bitmask for O(1) color selection.
        Prevents monochromatic 3-term APs to avoid dead-ends."""
        random.seed(seed)
        colors = [-1] * n
        colors[0] = 0
        colors[1] = 0
        colors[2] = 1 if num_colors > 1 else 0

        # For each position, we need to check constraints from:
        # 1. 4-term APs where this position is the last element
        # 2. 4-term APs where this position creates a monochromatic 3-term sub-AP
        
        # Precompute for each position the APs ending there
        ap_endings = [[] for _ in range(n)]
        for (a, a1, a2, a3) in all_aps:
            ap_endings[a3].append((a, a1, a2))

        # Precompute 3-term APs for monochromatic check
        # A 3-term AP (x, x+d, x+2d) is problematic if all same color
        # We want to avoid creating monochromatic 3-term APs

        for pos in range(3, n):
            # Build forbidden mask
            forbidden = 0

            # Check 4-term APs ending at pos
            for (a, a1, a2) in ap_endings[pos]:
                c1, c2, c3 = colors[a], colors[a1], colors[a2]
                # We need at least 3 distinct colors in {color, c1, c2, c3}
                # If c1, c2, c3 use only 1 distinct color, then color must be different
                # If c1, c2, c3 use only 2 distinct colors, color can be any of those or new
                distinct = {c1, c2, c3}
                if len(distinct) == 1:
                    # All same, color must be different
                    forbidden |= (1 << c1)
                elif len(distinct) == 2:
                    # Two colors used, color can be any (including the two or new)
                    # But we need at least 3 total, so color can be anything
                    pass
                # If len(distinct) == 3, any color works

            # Check 3-term APs to avoid monochromatic
            # For each d, check (pos-2d, pos-d, pos) and (pos-d, pos, pos+d) etc.
            # Actually, we should check if assigning color creates a mono 3-AP
            for d in range(1, pos // 2 + 1):
                # Check (pos-2d, pos-d, pos)
                if pos - 2 * d >= 0:
                    if colors[pos - 2 * d] == colors[pos - d]:
                        forbidden |= (1 << colors[pos - 2 * d])
                # Check (pos-d, pos, pos+d) - but pos+d not assigned yet, skip
                # Check (pos, pos+d, pos+2d) - future, skip

            # Also check 3-term APs where pos is first: (pos, pos+d, pos+2d)
            # These don't constrain pos directly since future positions are unknown
            # But if we're being cautious, we might want to avoid setting up traps
            
            # Find lowest available color
            color = 0
            while color < num_colors and (forbidden >> color) & 1:
                color += 1
            if color >= num_colors:
                # Try to find any color that minimizes future problems
                # For now, just pick the one with fewest conflicts
                color = 0
                best_conflicts = float('inf')
                for c in range(num_colors):
                    if (forbidden >> c) & 1:
                        continue
                    # Count how many 3-term APs this would make monochromatic
                    conflicts = 0
                    for d in range(1, (n - 1 - pos) // 2 + 1):
                        if pos + 2 * d < n:
                            if colors[pos + d] != -1 and colors[pos + d] == c:
                                conflicts += 1
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        color = c
            colors[pos] = color

        return colors

    def local_search(colors, num_colors, max_iters=5000):
        """Local search to minimize number of colors used while maintaining validity."""
        # First, ensure we have a valid coloring
        violations = count_violations(colors)
        if violations > 0:
            # Try to fix violations
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                violating = get_violating_aps(colors)
                if not violating:
                    break
                ap_idx = random.choice(violating)
                a, a1, a2, a3 = all_aps[ap_idx]
                # Try changing one of the positions
                pos = random.choice([a, a1, a2, a3])
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                if count_violations(colors) < violations:
                    violations = count_violations(colors)
                else:
                    colors[pos] = old_color

        # Now minimize colors
        current_colors = set(colors)
        while len(current_colors) > 3:
            if time.time() > deadline:
                break
            
            # Try to eliminate one color
            target_color = random.choice(list(current_colors))
            improved = False
            
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                # Find positions with target_color
                positions = [i for i in range(n) if colors[i] == target_color]
                if not positions:
                    break
                
                pos = random.choice(positions)
                # Try changing to a different color
                other_colors = [c for c in current_colors if c != target_color]
                new_color = random.choice(other_colors)
                
                old_color = colors[pos]
                colors[pos] = new_color
                violations = count_violations(colors)
                
                if violations == 0:
                    # Check if we can remove target_color
                    if target_color not in colors:
                        current_colors.discard(target_color)
                        improved = True
                        break
                else:
                    colors[pos] = old_color
            
            if not improved:
                # Try a different target color
                break

        return colors

    # Try different numbers of colors, starting from 3
    best_result = None
    best_color_count = 128

    for num_colors in range(3, 8):
        if time.time() > deadline:
            break

        # Try multiple seeds for greedy init
        for seed in range(20):
            if time.time() > deadline:
                break

            colors = greedy_init(num_colors, seed)
            violations = count_violations(colors)

            if violations == 0:
                # Valid coloring, try to minimize
                colors = local_search(colors, num_colors, max_iters=2000)
                actual_colors = len(set(colors))
                if actual_colors < best_color_count:
                    best_color_count = actual_colors
                    best_result = colors[:]

                if best_color_count == 3:
                    return best_result

    # If no valid coloring found, use greedy fallback
    if best_result is None:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3+1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2+1))):
                    colors.append(color)
                    break
        return colors

    return best_result
# EVOLVE-BLOCK-END
