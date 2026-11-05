<<<<<<< SEARCH
def _try_circulant_construction(n):
    """
    Try to find a circulant graph on n vertices that is triangle-free with alpha < 10.
    """
    if _remaining_time() < 5:
        return None

    rng = random.Random(12345)
    half = n // 2

    def _shifts_create_triangle(shifts_set):
        """Check if the shift set creates a triangle in the circulant graph."""
        dist_set = set()
        for s in shifts_set:
            dist_set.add(s)
            dist_set.add(n - s)
        
        for a in shifts_set:
            for b in shifts_set:
                c = (a + b) % n
                if c != 0 and c in dist_set:
                    if a in dist_set and b in dist_set and c in dist_set:
                        return True
                c2 = (a - b) % n
                if c2 != 0 and c2 in dist_set:
                    if a in dist_set and b in dist_set and c2 in dist_set:
                        return True
        return False

    best_adj = None
    best_alpha = float('inf')
    
    for strategy_seed in range(30):
        if _remaining_time() < 3:
            break
        
        rng_s = random.Random(12345 + strategy_seed)
        available_shifts = list(range(1, half + 1))
        rng_s.shuffle(available_shifts)
        
        current_shifts = set()
        for s in available_shifts:
            trial = current_shifts | {s}
            if not _shifts_create_triangle(trial):
                current_shifts.add(s)
        
        if not current_shifts:
            continue
        
        adj = _cyclic_graph(n, list(current_shifts))
        if _has_triangle_bitmask(adj, n):
            continue
        
        alpha = _max_independent_set_size_bitmask(adj, n)
        if alpha < 10:
            return adj
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj
    
    for attempt in range(100):
        if _remaining_time() < 3:
            break

        num_shifts = rng.randint(n // 5, n // 3)
        num_shifts = min(num_shifts, half)
        shifts = rng.sample(range(1, half + 1), num_shifts)

        adj = _cyclic_graph(n, shifts)

        if _has_triangle_bitmask(adj, n):
            continue

        alpha = _max_independent_set_size_bitmask(adj, n)
        if alpha < 10:
            return adj
        if alpha < best_alpha and best_adj is not None:
            best_alpha = alpha
            best_adj = adj

    return best_adj
=======
def _is_triangle_free_shifts(n, shifts_set):
    """
    Check if a set of shifts for a circulant graph on n vertices is triangle-free.
    A triangle exists if there are a, b, c in shifts_set (or their negatives) such that
    a + b + c = 0 (mod n) or a + b = c (mod n) etc.
    Essentially, we need to ensure that for any x, y in S (where S includes negatives),
    x+y is not in S (unless x+y=0, which implies x=-y, i.e., x and -x are both in S,
    which is fine as long as they don't form a 2-cycle triangle? No, simple graph.
    Wait, triangle 0-1-2-0 requires 1, 2, 1+2 to be in the difference set.
    So if S is the set of distances, we check if s1 + s2 = s3 (mod n) for any s1, s2, s3 in S.
    """
    dists = set()
    for s in shifts_set:
        dists.add(s % n)
        dists.add((-s) % n)
    
    # Remove 0 if present (shouldn't be)
    dists.discard(0)
    
    for a in dists:
        for b in dists:
            c = (a + b) % n
            if c in dists:
                # Check if this forms a non-trivial triangle
                # If a+b=c, and a,b,c are distinct non-zero distances, we have a triangle.
                # If a=-b, c=0, not a triangle.
                if a != b and a != c and b != c:
                    return False
    return True

def _deterministic_circulant_search(n, time_limit=10.0):
    """
    Perform a deterministic local search on circulant graphs.
    We start with a set of shifts and try to add/remove shifts to minimize alpha.
    """
    start_time = time.time()
    half = n // 2
    all_shifts = list(range(1, half + 1))
    
    best_adj = None
    best_alpha = float('inf')
    
    # Initialize with a few different starting configurations
    # 1. Random sparse
    # 2. Random dense (triangle-free check)
    # 3. Greedy max triangle-free
    
    configs = []
    
    # Config 1: Greedy triangle-free
    greedy_shifts = set()
    for s in all_shifts:
        if _is_triangle_free_shifts(n, greedy_shifts | {s}):
            greedy_shifts.add(s)
    if greedy_shifts:
        configs.append(greedy_shifts)
        
    # Config 2: Random sparse
    rng = random.Random(1)
    for _ in range(5):
        num_shifts = rng.randint(2, 5)
        s_set = set(rng.sample(all_shifts, num_shifts))
        if _is_triangle_free_shifts(n, s_set):
            configs.append(s_set)
            
    # Config 3: Random dense (try to fill up)
    for _ in range(5):
        s_set = set()
        shuffled = all_shifts[:]
        rng.shuffle(shuffled)
        for s in shuffled:
            if _is_triangle_free_shifts(n, s_set | {s}):
                s_set.add(s)
        if s_set:
            configs.append(s_set)

    for initial_shifts in configs:
        if time.time() - start_time > time_limit:
            break
            
        current_shifts = set(initial_shifts)
        if not _is_triangle_free_shifts(n, current_shifts):
            continue
            
        current_adj = _cyclic_graph(n, list(current_shifts))
        current_alpha = _max_independent_set_size_bitmask(current_adj, n)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_adj = current_adj[:]
            
        if current_alpha < 10:
            return current_adj
            
        # Local Search: Try adding shifts
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            for s in all_shifts:
                if s in current_shifts:
                    continue
                trial_shifts = current_shifts | {s}
                if not _is_triangle_free_shifts(n, trial_shifts):
                    continue
                trial_adj = _cyclic_graph(n, list(trial_shifts))
                trial_alpha = _max_independent_set_size_bitmask(trial_adj, n)
                if trial_alpha < current_alpha:
                    current_shifts = trial_shifts
                    current_adj = trial_adj
                    current_alpha = trial_alpha
                    improved = True
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_adj = current_adj[:]
                    if current_alpha < 10:
                        return current_adj
                        
        # Local Search: Try removing shifts (if we are stuck and alpha is high, maybe removing a "bad" shift helps? 
        # Actually, adding edges usually decreases alpha. So we mostly want to add.
        # But maybe a specific shift creates a large independent set structure?
        # Let's try removing one shift to see if it allows adding two others later?
        # This is complex. Let's stick to addition first.
        
    return best_adj
>>>>>>> REPLACE

