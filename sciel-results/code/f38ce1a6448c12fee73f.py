"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The minimum number of terms for sum=1 with denominators >= 5 is 9.
    # We use a DP to find a minimal-length set of denominators.
    # We use LCM(5..15) = 360360 or similar, but 2520 is a good base for small denominators.
    # However, denominators can be up to 10^7.
    # The DP with target=2520 finds denominators in [5, 2520].
    # This is sufficient to find a 9-term solution.
    
    target = 2520
    # reached[sum] = list of denominators
    # We want to minimize the length of the list.
    reached = {0: []}
    
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        # Iterate over a copy to avoid updating dict while iterating
        for total, values in list(reached.items()):
            new_total = total + w
            if new_total <= target:
                # Update if new list is shorter
                if new_total not in reached or len(values) + 1 < len(reached[new_total]):
                    reached[new_total] = values + [d]
                    
    if target in reached:
        return reached[target]
    
    # If no solution found in [5, 2520], try a larger range or hardcoded
    # But 9 terms exist in this range.
    
    # Fallback: Return a known valid 9-term solution if one is available
    # Example: 5, 6, 7, 8, 9, 10, 14, 15, 20 is NOT valid.
    # Let's just trust the DP or raise.
    
    # To be absolutely safe against any edge cases or if DP misses a specific minimal combo
    # due to integer division issues (none here), we can add a small search.
    
    # Actually, the DP is robust.
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
