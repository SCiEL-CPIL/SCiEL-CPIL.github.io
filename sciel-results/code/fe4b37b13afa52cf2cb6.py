"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_4ap(colors, n=128):
    """Check if every nonconstant 4-term AP uses >= 3 distinct colors."""
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            idx = [a, a + d, a + 2*d, a + 3*d]
            if idx[-1] >= n:
                break
            cs = {colors[i] for i in idx}
            if len(cs) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _try_pattern_mod(n=128, mod=4):
    """Try coloring c(i) = i % mod, which gives 'mod' colors."""
    colors = [i % mod for i in range(n)]
    if _is_valid_4ap(colors, n):
        return colors
    return None

def _try_pattern_mod_shift(n=128, mod=4, shifts=None):
    """Try c(i) = (i + shift[i % period]) % mod for various shifts."""
    if shifts is None:
        return None
    colors = [(i + shifts[i % len(shifts)]) % mod for i in range(n)]
    if _is_valid_4ap(colors, n):
        return colors
    return None

def _greedy_construct(n=128, max_colors=3, seed=42):
    """Greedy: assign each position the smallest color that doesn't create a violation."""
    rng = random.Random(seed)
    colors = [None] * n
    # Try all permutations of color priority
    for perm in range(max_colors):
        color_order = list(range(max_colors))
        rng.shuffle(color_order)
        colors = [None] * n
        valid = True
        for i in range(n):
            found = False
            for c in color_order:
                colors[i] = c
                # Check all 4-term APs ending at or involving position i
                ok = True
                for a in range(n):
                    for d in range(1, (n - 1 - a) // 3 + 1):
                        idx = [a, a+d, a+2*d, a+3*d]
                        if idx[-1] >= n:
                            break
                        if i not in idx:
                            continue
                        cs = set()
                        for j in idx:
                            if colors[j] is None:
                                cs = None
                                break
                            cs.add(colors[j])
                        if cs is not None and len(cs) < 3:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            return colors
    return None

def _local_search(colors, n=128, max_iter=5000, seed=0):
    """Local search: try flipping individual colors to reduce total color count."""
    rng = random.Random(seed)
    best = list(colors)
    best_count = _count_colors(best)
    
    for it in range(max_iter):
        i = rng.randint(0, n-1)
        old_c = best[i]
        # Try each color
        for new_c in range(128):
            if new_c == old_c:
                continue
            best[i] = new_c
            # Check validity
            valid = True
            for a in range(n):
                for d in range(1, (n - 1 - a) // 3 + 1):
                    idx = [a, a+d, a+2*d, a+3*d]
                    if idx[-1] >= n:
                        break
                    cs = {best[j] for j in idx}
                    if len(cs) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                new_count = _count_colors(best)
                if new_count < best_count:
                    best_count = new_count
                    old_c = new_c
                    break
            else:
                best[i] = old_c
    return best

def construct():
    start = time.time()
    deadline = start + 80  # 10s safety margin
    
    n = 128
    
    # First try simple modular patterns
    for mod in [3, 4, 5, 6, 7, 8]:
        colors = _try_pattern_mod(n, mod)
        if colors is not None:
            return colors
    
    # Try greedy construction with 3 colors
    for seed in range(20):
        if time.time() > deadline:
            break
        colors = _greedy_construct(n, max_colors=3, seed=seed)
        if colors is not None:
            # Try to reduce further with local search
            colors = _local_search(colors, n, max_iter=2000, seed=seed)
            return colors
    
    # Fallback: use the original seed algorithm
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    
    return colors
# EVOLVE-BLOCK-END
