def construct():
    limit = 256
    candidates = list(range(2, limit + 1))
    
    # Pre-compute a heuristic score for each candidate
    # Score: number of pairs (b, c) in [2, 256] such that x divides b*c
    # Lower score is better (safer to include)
    scores = {}
    for x in candidates:
        count = 0
        for b in range(2, limit + 1):
            for c in range(b + 1, limit + 1):
                if (b * c) % x == 0:
                    count += 1
        scores[x] = count
        
    # Sort candidates by score (ascending)
    # Tie-breaker: larger numbers first
    sorted_candidates = sorted(candidates, key=lambda x: (scores[x], -x))
    
    result = []
    
    for x in sorted_candidates:
        valid = True
        
        # Check 1: x does not divide a*b for any distinct a,b in result
        for i in range(len(result)):
            a = result[i]
            for j in range(i + 1, len(result)):
                b = result[j]
                if (a * b) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
            
        # Check 2: a does not divide x*b for any distinct a,b in result
        for i in range(len(result)):
            a = result[i]
            for j in range(i + 1, len(result)):
                b = result[j]
                if (x * b) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
            
        result.append(x)
        
    return result
