"""Construct largest set with cardinality-separated subset sums."""

def construct():
    """
    Find the largest set of distinct integers in 1..256 such that
    whenever two subsets have different cardinalities, their sums differ.
    
    Uses greedy construction with exact cardinality-indexed subset-sum DP.
    """
    import time
    
    deadline = time.time() + 50
    
    result = []
    sums_by_card = [set() for _ in range(61)]
    sums_by_card[0].add(0)
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        
        n = len(result)
        new_sums = [set() for _ in range(n + 2)]
        
        for k in range(n + 1):
            new_sums[k] = set(sums_by_card[k])
        
        for k in range(1, n + 2):
            for s in sums_by_card[k - 1]:
                new_sums[k].add(s + x)
        
        valid = True
        seen = {}
        for k in range(n + 2):
            for s in new_sums[k]:
                if s in seen:
                    valid = False
                    break
                seen[s] = k
            if not valid:
                break
        
        if valid:
            sums_by_card = new_sums
            result.append(x)
    
    return result

