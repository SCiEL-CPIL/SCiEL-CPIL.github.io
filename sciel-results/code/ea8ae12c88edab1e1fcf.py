"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85
    
    def lcm_val(a, b):
        return a * b // gcd(a, b)
    
    nums = list(range(2, 257))
    n = 255
    
    # Precompute incompatibility
    incompat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if lcm_val(nums[i], nums[j]) <= 256:
                incompat[i].add(j)
                incompat[j].add(i)
    
    weights = [1.0 / nums[i] for i in range(n)]
    
    # Start with [129,256]
    current = set(range(127, 255))
    
    # Greedy add
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(n):
            if i in current:
                continue
            if not (incompat[i] & current):
                current.add(i)
                changed = True
    
    best_set = set(current)
    best_score = sum(weights[i] for i in best_set)
    
    # Single-swap local search
    while time.time() < deadline:
        improved = False
        for i in list(best_set):
            if time.time() > deadline:
                break
            cand = best_set - {i}
            base = best_score - weights[i]
            best_j = -1
            best_w = 0.0
            for j in range(n):
                if j in cand:
                    continue
                if not (incompat[j] & cand):
                    if weights[j] > best_w:
                        best_w = weights[j]
                        best_j = j
            if best_j >= 0 and base + best_w > best_score + 1e-12:
                best_set = cand | {best_j}
                best_score = base + best_w
                improved = True
                break
        if not improved:
            break
    
    result = sorted(nums[i] for i in best_set)
    
    # Validate
    for a in range(len(result)):
        for b in range(a + 1, len(result)):
            if lcm_val(result[a], result[b]) <= 256:
                return list(range(129, 257))
    
    return result
# EVOLVE-BLOCK-END
