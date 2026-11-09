import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Structural change: Use pairwise coordinate "discordance"
    # A 3-term AP in F_3^n requires x_i, x_j, x_k such that x_i + x_k = 2*x_j mod 3.
    # This implies that for each coordinate, the values must not be "too uniform".
    # Specifically, if we look at pairs of coordinates, we want to avoid patterns
    # that are prone to forming APs.
    
    # A simple structural metric: Count the number of coordinate pairs (i, j) 
    # where vector[i] != vector[j]. This is the "Hamming weight" of the vector
    # relative to its own mean or just a measure of diversity.
    # Actually, a better metric for cap sets is the "energy" or "spread".
    
    # Let's try a metric based on the "distance" from the all-zeros and all-twos vectors,
    # which are the "corners" of the cube. Vectors in the middle (many 1s) are
    # often part of large cap sets.
    
    # New Structural Metric: 
    # 1. Count symbols
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
            
    # 2. Calculate a "centrality" score. 
    # Vectors with c1 close to n are central.
    # Vectors with c0 and c2 balanced are also good.
    # Let's use the distance from the "center" of the simplex.
    # Center is (n/3, n/3, n/3).
    # Distance squared = (c0 - n/3)^2 + (c1 - n/3)^2 + (c2 - n/3)^2
    # We want to MINIMIZE this distance? No, cap sets often include vectors
    # that are not at the center. 
    # Actually, the seed program (score 1.0) minimized "bad pairs" which is
    # equivalent to maximizing c0*c2 + c1^2. This favors vectors with many 1s
    # or many 0s/2s.
    
    # Let's try a different structural family: 
    # "Pairwise Coordinate Interaction"
    # For each pair of coordinates (i, j), look at the pair (v_i, v_j).
    # There are 9 possible pairs. 
    # Some pairs are "bad" for cap sets. 
    # Specifically, if we have two vectors x and y in the set, x+y is in the set
    # if it forms an AP. 
    # This is hard to compute per vector.
    
    # Let's stick to a simpler structural change: 
    # Use the "Hamming weight" of the vector (number of non-zero entries) 
    # combined with the "Hamming weight" of the vector mod 2 (or something similar).
    
    # Actually, let's look at inspiration 2: "pairwise coordinate relationships".
    # Let's compute the number of "discordant" pairs.
    # A pair (i, j) is discordant if v_i != v_j.
    # discordant = n*(n-1)/2 - (c0*(c0-1)/2 + c1*(c1-1)/2 + c2*(c2-1)/2)
    # This is just n^2 - c0^2 - c1^2 - c2^2 over 2.
    
    # This metric favors vectors with balanced counts (maximizes discordance).
    # Balanced vectors are often good for cap sets.
    
    # Let's combine:
    # Primary: Maximize discordance (balance)
    # Secondary: Tie-break with hash
    
    # Max discordance: n^2/3 (approx)
    # Min discordance: 0 (all same)
    
    # Let's try: Score = discordance * 10 + (n - abs(c1 - n/3) - abs(c0 - n/3) - abs(c2 - n/3))
    
    # Simpler: 
    # Score = (c0*c2 + c1*c1) * 10 + (n - abs(c0 - c2))
    # This is similar to the seed but with a twist.
    
    # Let's try a completely different representation: 
    # Map the vector to a number in base 3, but then apply a non-linear transform
    # that emphasizes certain patterns.
    
    # Let's try: 
    # 1. Compute the "energy" E = sum(v_i^2)
    # 2. Compute the "linear" L = sum(v_i)
    # 3. Score = E * 10 - L
    
    # For v in {0,1,2}:
    # v=0: E+=0, L+=0
    # v=1: E+=1, L+=1
    # v=2: E+=4, L+=2
    
    # Vector of all 0s: E=0, L=0, Score=0
    # Vector of all 1s: E=n, L=n, Score=9n
    # Vector of all 2s: E=4n, L=2n, Score=38n
    # Vector with n/2 1s and n/2 2s: E=n/2+2n=2.5n, L=n/2+n=1.5n, Score=25n-1.5n=23.5n
    
    # This seems to favor 2s. Let's invert?
    
    # Let's try the "Discordance" metric from above.
    # It's a structural measure of how "mixed" the vector is.
    
    # discordance = (n*n - c0*c0 - c1*c1 - c2*c2) / 2
    
    # We want to maximize this? Or minimize?
    # Cap sets often contain vectors that are "spread out".
    # Let's maximize discordance.
    
    discordance = (n*n - c0*c0 - c1*c1 - c2*c2)
    
    # Secondary: prefer c1 to be large?
    # The seed program favored c1.
    # Let's add c1 as a secondary term.
    
    score = discordance * 10 + c1
    
    # Normalize
    max_discordance = n*n
    max_score = max_discordance * 10 + n
    normalized_score = score / max_score
    
    # Tie-break with hash
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    hash_fraction = mixed / (2**64)
    
    return float(normalized_score + hash_fraction)
