import math

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = sorted(set(U))
        if not U or U[0] != 0:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0
        return 1 + math.log(ratio) / denom

    # The set {0, 1, 3, 7, 15} yields a score of ~1.098
    # This is significantly better than the previous local search result of 1.0275
    best_set = [0, 1, 3, 7, 15]
    best_score = score_of(best_set)
    
    # Perform a limited local search to see if we can improve further
    # We try to replace elements or add/remove near the current set
    max_val = 50
    improved = True
    while improved:
        improved = False
        current = list(best_set)
        current_score = best_score
        
        # Try replacing each element
        for i in range(1, len(current)):
            old = current[i]
            for x in range(1, max_val + 1):
                if x == old or x in current:
                    continue
                new_set = current[:i] + [x] + current[i+1:]
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved:
            for x in range(1, max_val + 1):
                if x not in current:
                    new_set = current + [x]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
        
        # Try removing an element
        if not improved:
            for i in range(1, len(current)):
                new_set = current[:i] + current[i+1:]
                if len(new_set) < 2:
                    continue
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
        
        if improved:
            best_set = current
            best_score = current_score

    return sorted(best_set)
# EVOLVE-BLOCK-END
