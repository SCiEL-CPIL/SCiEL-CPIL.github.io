"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _elapsed():
    return time.time() - _START

def _is_valid(colors, n=128):
    """Check every nonconstant 4-term AP uses at least 3 distinct colors."""
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if d == 0:
                continue
            b, c, e = a + d, a + 2 * d, a + 3 * d
            if e >= n:
                break
            vals = {colors[a], colors[b], colors[c], colors[e]}
            if len(vals) < 3:
                return False
    return True

def _num_colors(colors):
    return len(set(colors))

def _count_violations(colors, n=128):
    """Count number of 4-term APs that use fewer than 3 distinct colors."""
    count = 0
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            b, c, e = a + d, a + 2 * d, a + 3 * d
            if e >= n:
                break
            vals = {colors[a], colors[b], colors[c], colors[e]}
            if len(vals) < 3:
                count += 1
    return count

def _greedy_construct(K, seed=0, n=128):
    """Greedy construction with K colors, trying to minimize violations."""
    rng = random.Random(seed)
    colors = [0] * n
    # Try multiple random orderings and color assignments
    best_colors = None
    best_violations = float('inf')
    
    for attempt in range(50):
        colors = [rng.randint(0, K - 1) for _ in range(n)]
        v = _count_violations(colors, n)
        if v < best_violations:
            best_violations = v
            best_colors = colors[:]
        if best_violations == 0:
            break
        if _elapsed() > 70:
            break
    return best_colors, best_violations

def _local_search(colors, K, n=128, max_iter=200000):
    """Local search: flip colors to reduce violations."""
    colors = colors[:]
    current_violations = _count_violations(colors, n)
    if current_violations == 0:
        return colors, 0
    
    best_colors = colors[:]
    best_violations = current_violations
    
    for i in range(max_iter):
        if _elapsed() > 75:
            break
        if best_violations == 0:
            break
        
        # Pick a random position to try changing
        pos = random.randint(0, n - 1)
        old_color = colors[pos]
        new_color = random.randint(0, K - 1)
        if new_color == old_color:
            continue
        
        colors[pos] = new_color
        new_violations = _count_violations(colors, n)
        
        if new_violations <= best_violations:
            best_violations = new_violations
            best_colors = colors[:]
        else:
            colors[pos] = old_color
    
    return best_colors, best_violations

def _try_k_colors(K, n=128, max_attempts=100):
    """Try to find a valid coloring with exactly K colors."""
    for attempt in range(max_attempts):
        if _elapsed() > 70:
            break
        seed = attempt * 1000 + 42
        colors, violations = _greedy_construct(K, seed=seed, n=n)
        if violations == 0:
            return colors
        # Try local search to fix violations
        colors, violations = _local_search(colors, K, n=n, max_iter=50000)
        if violations == 0:
            return colors
    return None

def construct():
    global _START
    _START = time.time()
    n = 128
    
    # First, try small K values
    for K in range(2, 13):
        if _elapsed() > 60:
            break
        result = _try_k_colors(K, n=n, max_attempts=50 if K <= 5 else 20)
        if result is not None:
            # Verify
            if _is_valid(result, n):
                return result
    
    # Fallback: use a constructive approach with 2-adic valuation
    # v2(x) = exponent of 2 in x
    # For 4-term AP a, a+d, a+2d, a+3d: at least one term has odd v2
    # Actually, let's use a different constructive method
    
    # Use modulo-based coloring: color(x) = x mod 3
    # Check if this works
    colors = [x % 3 for x in range(1, n + 1)]
    if _is_valid(colors, n):
        return colors
    
    # Try color(x) = (x // 2) % 4  (based on binary representation)
    colors = [(x // 2) % 4 for x in range(1, n + 1)]
    if _is_valid(colors, n):
        return colors
    
    # Try a mixed approach: use 2-adic valuation modulo some K
    for mod in range(2, 10):
        colors = []
        for x in range(1, n + 1):
            v = 0
            while x % 2 == 0:
                v += 1
                x //= 2
            colors.append(v % mod)
        if _is_valid(colors, n):
            return colors
    
    # Last resort: random with many colors
    best_colors = None
    best_nc = float('inf')
    for _ in range(1000):
        if _elapsed() > 65:
            break
        K = random.randint(8, 20)
        colors = [random.randint(0, K - 1) for _ in range(n)]
        if _is_valid(colors, n):
            nc = _num_colors(colors)
            if nc < best_nc:
                best_nc = nc
                best_colors = colors[:]
    
    if best_colors is not None:
        return best_colors
    
    # Absolute fallback: return a valid coloring with more colors
    # Use the seed program's approach but with a time limit
    colors = []
    for x in range(1, n + 1):
        if _elapsed() > 65:
            # Fill rest with a simple pattern
            for j in range(x, n + 1):
                colors.append(j % 10)
            break
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    break
                idxs = [x - 1 - j * d for j in range(4)]
                vals = {color} | {colors[i] for i in idxs[1:] if i < len(colors)}
                if len(vals) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
        else:
            colors.append(x % 128)
    
    return colors

_START = time.time()
# EVOLVE-BLOCK-END
