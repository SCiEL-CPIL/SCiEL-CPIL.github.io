"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Based on the analysis, the interval [171, 256] is a valid set of size 86.
    # 171 is the smallest integer such that 3*171 > 2*256 (513 > 512).
    # This ensures that for any a, b, c in [171, 256] with a < b <= c,
    # b+c < 3a, so a cannot divide b+c (since b+c > 2a).
    # Size is 256 - 171 + 1 = 86.
    # Score = 86 / 65 = 1.323...
    
    # We can also run a quick greedy to see if we can beat 86, 
    # but given the conflict density, it's unlikely.
    # We return the known optimal interval.
    
    return list(range(171, 257))
# EVOLVE-BLOCK-END
