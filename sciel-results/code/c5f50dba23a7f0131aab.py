import time
import random
import numpy as np

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _precompute_aps(n=128):
    """Precompute all 4-APs as (a, b, c, e) tuples and index by position."""
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                ap = (a, b, c, e)
                all_aps.append(ap)
                for p in ap:
                    pos_to_aps[p].append(ap)
    return all_aps, pos_to_aps

def _is_valid_assignment(colors, pos, color, pos_to_aps):
    """Check if assigning `color` to `pos` keeps all APs containing pos valid."""
    for ap in pos_to_aps[pos]:
        a, b, c, e = ap
        cols = [colors[a], colors[b], colors[c], colors[e]]
        # replace pos's current value with candidate
        for idx, p in enumerate((a, b, c, e)):
            if p == pos:
                cols[idx] = color
                break
        if len(set(cols)) < 3:
            return False
    return True

def _greedy_3color_fast(n=128, seed=0):
    """Fast randomized greedy trying to use only 3 colors, with incremental constraint check."""
    rng = random.Random(seed)
    colors = [-1] * n
    for i in range(n):
        valid_colors = [0, 1, 2]
        # check all 4-APs that include i and have other positions already assigned
        # APs where i is the last term: i, i-d, i-2d, i-3d
        for d in range(1, (i) // 3 + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            if a < 0:
                break
            c1, c2, c3 = colors[a], colors[b], colors[c]
            for col in list(valid_colors):
                if len({c1, c2, c3, col}) < 3:
                    valid_colors.remove(col)
        # APs where i is not the last term: i, i+d, i+2d, i+3d but those positions are unassigned (>= i)
        # so they don't constrain yet. Only APs fully in [0, i] with i as last matter.
        # Also APs where i is an interior point: a, i, i+d, i+2d? No - 4-AP is a, a+d, a+2d, a+3d.
        # If i = a+d: a = i-d, a+2d = i+d (unassigned), a+3d = i+2d (unassigned) -> not all assigned.
        # If i = a+2d: a = i-2d, a+d = i-d (assigned), a+3d = i+d (unassigned) -> not all assigned.
        # So only the "i is last" case matters for a forward greedy.
        if not valid_colors:
            return None
        # prefer already-used colors to minimize count
        used = set(colors[:i])
        preferred = [c for c in valid_colors if c in used]
        if preferred:
            colors[i] = rng.choice(preferred)
        else:
            colors[i] = rng.choice(valid_colors)
    return colors

def _simulated_annealing(colors, n=128, pos_to_aps=None, max_time=10.0, seed=0):
    """Simulated annealing to minimize the number of distinct colors used.
    
    State: a valid coloring. Move: recolor one position to any color in 0..127.
    Accept if the new coloring is valid; prefer moves that reduce distinct color count.
    """
    if pos_to_aps is None:
        _, pos_to_aps = _precompute_aps(n)
    rng = random.Random(seed)
    current = list(colors)
    current_colors = set(current)
    current_count = len(current_colors)
    best = list(current)
    best_count = current_count
    
    start_time = time.time()
    # temperature schedule
    T = 2.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start_time < max_time:
        # pick a random position
        pos = rng.randrange(n)
        old_color = current[pos]
        # pick a new color: bias toward existing colors to reduce count
        if rng.random() < 0.8:
            new_color = rng.choice(sorted(current_colors))
        else:
            new_color = rng.randrange(128)
        if new_color == old_color:
            continue
        if not _is_valid_assignment(current, pos, new_color, pos_to_aps):
            continue
        # compute new color count
        new_colors = set(current_colors)
        # old_color may be removed
        # count occurrences
        # cheap: track counts
        # We'll just compute set after tentative change
        current[pos] = new_color
        tentative = set(current)
        new_count = len(tentative)
        delta = new_count - current_count
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            # accept
            current_colors = tentative
            current_count = new_count
            if new_count < best_count:
                best = list(current)
                best_count = new_count
                if best_count <= 3:
                    break
        else:
            # reject, revert
            current[pos] = old_color
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best

def _local_search_reduce_colors(colors, n=128, max_iters=5000, seed=0):
    """Try to reduce color count by merging colors."""
    rng = random.Random(seed)
    colors = list(colors)
    current_count = len(set(colors))
    
    if current_count <= 3:
        return colors
    
    color_list = sorted(set(colors))
    
    for _ in range(max_iters):
        if len(color_list) <= 3:
            break
        c1, c2 = rng.sample(color_list, 2)
        new_colors = list(colors)
        for i in range(n):
            if new_colors[i] == c2:
                new_colors[i] = c1
        if _validate(new_colors):
            colors = new_colors
            color_list = sorted(set(colors))
            current_count = len(color_list)
            if current_count <= 3:
                break
    return colors

def _construct_3color_direct(n=128, seed=0, time_limit=3.0):
    """Direct construction of 3-coloring using pattern-based approach."""
    start_time = time.time()
    rng = random.Random(seed)
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        if attempt == 0:
            colors = []
            for x in range(1, n + 1):
                v2 = 0
                tmp = x
                while tmp % 2 == 0:
                    tmp //= 2
                    v2 += 1
                colors.append(v2 % 3)
        elif attempt == 1:
            colors = []
            for x in range(1, n + 1):
                v3 = 0
                tmp = x
                while tmp % 3 == 0:
                    tmp //= 3
                    v3 += 1
                colors.append(v3 % 3)
        elif attempt == 2:
            colors = []
            for x in range(1, n + 1):
                s = bin(x - 1).count('1')
                colors.append(s % 3)
        elif attempt == 3:
            colors = []
            for x in range(1, n + 1):
                base = (x - 1) % 3
                if rng.random() < 0.1:
                    base = (base + 1) % 3
                colors.append(base)
        elif attempt < 10:
            colors = [-1] * n
            valid = True
            for i in range(n):
                valid_colors = {0, 1, 2}
                for d in range(1, (i) // 3 + 1):
                    a = i - 3 * d
                    b = i - 2 * d
                    c = i - d
                    if a >= 0:
                        c1, c2, c3 = colors[a], colors[b], colors[c]
                        for col in list(valid_colors):
                            if len({c1, c2, c3, col}) < 3:
                                valid_colors.discard(col)
                if not valid_colors:
                    valid = False
                    break
                colors[i] = rng.choice(list(valid_colors))
            if not valid:
                continue
        else:
            colors = [-1] * n
            valid = True
            for i in range(n):
                valid_colors = {0, 1, 2}
                for d in range(1, (i) // 3 + 1):
                    a = i - 3 * d
                    b = i - 2 * d
                    c = i - d
                    if a >= 0:
                        c1, c2, c3 = colors[a], colors[b], colors[c]
                        for col in list(valid_colors):
                            if len({c1, c2, c3, col}) < 3:
                                valid_colors.discard(col)
                if not valid_colors:
                    valid = False
                    break
                used = set(colors[:i])
                preferred = valid_colors & used
                if preferred:
                    colors[i] = rng.choice(list(preferred))
                else:
                    colors[i] = rng.choice(list(valid_colors))
            if not valid:
                continue
        
        if _validate(colors):
            if len(set(colors)) == 3:
                return colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    all_aps, pos_to_aps = _precompute_aps(n)
    
    # First, get the seed fallback (guaranteed valid)
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # Try fast 3-color greedy with many seeds
    remaining = deadline - time.time()
    if remaining > 5:
        for seed in range(200):
            if time.time() > deadline - 20:
                break
            result = _greedy_3color_fast(n, seed=seed)
            if result is not None and _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # Try direct 3-coloring construction (patterns)
    remaining = deadline - time.time()
    if remaining > 10:
        result = _construct_3color_direct(n, seed=0, time_limit=min(4.0, remaining - 5))
        if result is not None:
            best = result
            best_count = 3
            return best
    
    # Simulated annealing from the best valid coloring found so far
    remaining = deadline - time.time()
    if remaining > 10 and best_count > 3:
        sa_time = min(remaining - 5, 30.0)
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _simulated_annealing(best, n, pos_to_aps, max_time=sa_time / 10.0, seed=seed)
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # Fallback local search merge
    remaining = deadline - time.time()
    if remaining > 2 and best_count > 3:
        for seed in range(50):
            if time.time() > deadline - 1:
                break
            result = _local_search_reduce_colors(best, n, max_iters=1000, seed=seed)
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    return best
