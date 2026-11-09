import zlib
import time

# EVOLVE-BLOCK-START
_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill

def priority(vector, n):
    # Deadline guard: if we are too close to the wall-clock limit,
    # return a neutral value to avoid any further work in the evaluator.
    if time.monotonic() > _DEADLINE:
        return 0.0

    # Probabilistic collision-density model:
    # Estimate the local 3-AP collision probability for this vector based on
    # its coordinate-wise entropy and Hamming weight distribution.
    # Vectors with lower estimated collision density get higher priority.
    
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Coordinate-wise entropy: measures how "spread out" the vector is across
    # the three values. Higher entropy = more uniform = lower collision risk.
    # H = -sum(p_i * log2(p_i)) for non-zero p_i
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * (p ** 0.5)  # Using sqrt as a cheap concave proxy for log
    
    # Hamming weight distribution: count how many coordinates are "central" (value 1)
    # vs "extreme" (values 0 or 2). Central coordinates participate in fewer
    # potential APs with other central coordinates.
    central = c1
    extreme = c0 + c2
    
    # Collision density proxy:
    # 1. Low entropy (concentrated on few values) -> high collision risk
    # 2. Many extreme coordinates -> high collision risk (0-1-2 APs)
    # 3. Balance between extreme values matters: if c0 >> c2, vectors with many 0s
    #    and few 2s can form APs with vectors having intermediate values
    
    # Effective collision probability estimate:
    # Base risk from low entropy
    entropy_risk = 1.0 - entropy / (n / 3.0)  # Normalize: max entropy ~ n/3
    
    # Risk from extreme coordinate concentration
    extreme_ratio = extreme / n
    extreme_risk = extreme_ratio * (1.0 - 2.0 * min(c0, c2) / max(extreme, 1))
    
    # Combined collision density
    collision_density = entropy_risk * 0.6 + extreme_risk * 0.4
    
    # Priority is inversely proportional to collision density
    # Add a small CRC-based tiebreaker for determinism
    h = zlib.crc32(bytes(vector))
    tiebreak = float(h) / (2**32)
    
    # Higher priority = lower collision density
    score = (1.0 - collision_density) * 1e6 + tiebreak
    return score
# EVOLVE-BLOCK-END
