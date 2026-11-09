"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 80  # safety margin within 90s wall-clock

    n = 128

    # We want to minimize the number of distinct colors used.
    # Try with 3 colors first, then 4, etc.
    # The constraint: every nonconstant 4-AP (a, a+d, a+2d, a+3d) with a, a+3d in [1,n]
    # must use at least 3 distinct colors.

    def count_aps():
        """Count number of 4-term APs in [1, n]."""
        cnt = 0
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                cnt += 1
        return cnt

    def check_validity(colors):
        """Check if coloring satisfies the constraint."""
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a + 3 * d - 1]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    def count_colors(colors):
        return len(set(colors))

    def greedy_construct(num_colors, random_seed=0):
        """
        Greedy: assign colors one by one, picking the color that keeps the coloring valid
        for all APs ending at the current position, preferring colors already used.
        If no valid color, try to restart with different random choices.
        """
        rng = random.Random(random_seed)
        colors = [0] * n

        for x in range(1, n + 1):
            # Find all APs ending at x: (x-3d, x-2d, x-d, x) for d where x-3d >= 1
            # Also APs where x is the first, second, or third term:
            # x, x+d, x+2d, x+3d (x is first) - but those positions aren't colored yet
            # x, x+d, x+2d, x+3d where x is at position j (j=1,2,3)
            # Actually, when assigning color to position x, the only constraints we need to check
            # are APs where all 4 positions are already assigned or will be assigned.
            # Since we go left to right, positions > x are not assigned yet.
            # So we only need to check APs where x is the LAST position (x = a+3d).
            # For APs where x is not the last, those will be checked later.
            # Wait, that's not right. We need the final coloring to be valid.
            # Let's use a different approach: assign greedily, checking all APs that include x
            # and have all other positions already assigned.

            valid_colors = set(range(num_colors))

            # APs where x is the 4th term: a = x-3d, a+d, a+2d, x
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                # We need {c1, c2, c3, c4} to have >= 3 distinct colors
                # c4 is what we're choosing
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    # Any color works for the 4th position
                    pass
                elif len(existing) == 2:
                    # Need c4 to be different from at least one of the two
                    # i.e., c4 != c1 OR c4 != c2
                    # Actually we need {c1,c2,c3,c4} to have >=3 distinct
                    # Since existing has 2 distinct, we need c4 to add a new one
                    # OR c4 can be one of the two as long as the set still has >=3
                    # Wait: if existing = {A, B}, then {A,B,c4} needs >=3, so c4 must be != A and != B
                    # No wait: {A,B,c4} has >=3 elements means c4 is not in {A,B}
                    # But we only have num_colors colors. If num_colors=3, c4 must be the third color.
                    for c in list(valid_colors):
                        if c in existing:
                            valid_colors.discard(c)
                else:  # len(existing) == 1
                    # All three same color, c4 must be different
                    for c in list(valid_colors):
                        if c == list(existing)[0]:
                            valid_colors.discard(c)

            # APs where x is the 3rd term: a, a+d, x, x+d => a+d = x-d, so a = x-2d, positions: x-2d, x-d, x, x+d
            # x+d must be <= n, but x+d > x so not assigned yet. Skip for now? 
            # No - we need the final coloring to be valid. Let's just check 4th position for now
            # and do a final validation.

            if not valid_colors:
                return None

            # Prefer already-used colors to minimize distinct colors
            used_colors = set(colors[:x-1])
            preferred = valid_colors & used_colors
            if preferred:
                colors[x - 1] = min(preferred)  # deterministic choice
            else:
                colors[x - 1] = min(valid_colors)

        # Final validation
        if check_validity(colors):
            return colors
        return None

    # Try with 3 colors
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        result = greedy_construct(3, random_seed=attempt)
        if result is not None:
            return result

    # Try with 4 colors
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        result = greedy_construct(4, random_seed=100 + attempt)
        if result is not None:
            return result

    # Fallback: use the seed program's approach (should always work)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
