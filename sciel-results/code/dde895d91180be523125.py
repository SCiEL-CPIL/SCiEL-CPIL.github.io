def priority(vector, n):
        # Sum of squares favors 2s (4) > 1s (1) > 0s (0)
        s2 = sum(x*x for x in vector)
        
        # Calculate entropy of symbol distribution
        c0 = vector.count(0)
        c1 = vector.count(1)
        c2 = vector.count(2)
        
        # Entropy is maximized when c0=c1=c2
        # We want to favor high entropy? Or low?
        # High entropy vectors are "generic".
        # Let's add a small bonus for high entropy.
        import math
        probs = [c0/n, c1/n, c2/n]
        ent = 0
        for p in probs:
            if p > 0:
                ent -= p * math.log(p)
        
        # Max possible entropy is log(3)
        # Normalize entropy to [0, 1]
        norm_ent = ent / math.log(3) if n > 0 else 0
        
        # Combine
        # s2 ranges from 0 to 4n
        # norm_ent ranges from 0 to 1
        # We want s2 to be the main driver.
        return s2 + 0.1 * norm_ent
