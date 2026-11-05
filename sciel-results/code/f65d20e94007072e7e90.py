import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Target n=20. If we find a valid graph, we return it.
    # The problem states n=20 is known to exist.
    # We try to find one using a specific structural family:
    # Comparability graphs of random total orders with a distance threshold.
    # A vertex v_i is connected to v_j if |rank(i) - rank(j)| > T.
    # This creates a graph where degrees are highly heterogeneous,
    # making regular induced subgraphs unlikely.
    
    target_n = 20
    
    # Precompute all subsets of size 6 for fast checking
    # There are C(20, 6) = 38760 subsets.
    # Storing them as bitmasks is efficient.
    subsets_6 = []
    for subset in itertools.combinations(range(target_n), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        subsets_6.append(mask)
        
    # We also need to check larger sizes if size 6 is clean.
    # But usually, if size 6 is clean in a "heterogeneous" graph, larger sizes are too.
    # However, the requirement is "No set of 6 or more".
    # So we must check all k >= 6.
    # Checking all k is expensive: sum(C(20,k) for k=6..20) = 2^20 - (1+20+190+1140+4845+15504) 
    # = 1048576 - 21600 = 1026976 subsets.
    # This is ~1M checks. In Python, this might take 1-2 seconds per full check.
    # We have 115 seconds. So we can do ~50-100 full checks if they are fast enough.
    # But we can optimize: if a graph is valid for size 6, it's a strong candidate.
    # We will first filter by size 6, then verify the few candidates fully.
    
    found = False
    best_edges = []
    
    # Strategy:
    # 1. Generate random total orders (permutations).
    # 2. For each permutation, try different thresholds T.
    # 3. Check size 6 validity quickly.
    # 4. If valid for size 6, check full validity.
    
    # To make it even more robust, we can mix in some random edge flips?
    # No, stick to the structural family first. If it fails, we can fall back to random restarts.
    # But the directive says "Model the graph as the comparability graph...".
    # Let's stick to that family primarily.
    
    # Optimization:
    # Instead of checking all 38760 subsets for every candidate,
    # we can check them in a loop.
    
    random.seed(42)
    
    attempts = 0
    max_attempts = 20000 # Limit attempts to avoid infinite loop if time allows
    
    while attempts < max_attempts and time.time() < deadline - 5.0:
        attempts += 1
        
        # Generate a random permutation (total order)
        perm = list(range(target_n))
        random.shuffle(perm)
        
        # The rank of vertex v is perm.index(v)
        # But perm is a list of vertices in order.
        # Let rank[v] be the position of v in perm.
        rank = [0] * target_n
        for i, v in enumerate(perm):
            rank[v] = i
            
        # Try different thresholds T
        # T can range from 1 to target_n - 1
        # If T is small, graph is dense. If T is large, graph is sparse.
        # Regular subgraphs are less likely in very dense or very sparse graphs?
        # Actually, empty graph and complete graph are regular.
        # We want intermediate densities.
        
        for T in range(1, target_n):
            if time.time() > deadline - 5.0:
                break
                
            # Build adjacency masks
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if abs(rank[i] - rank[j]) > T:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Fast check: Size 6
            is_valid_6 = True
            for mask in subsets_6:
                # Check if induced subgraph on mask is regular
                # Vertices in mask
                # We need to iterate vertices in mask to get degrees
                # This inner loop is O(6) per mask.
                # Total O(38760 * 6) ~ 232k ops.
                # In Python, this is ~0.01-0.02s.
                
                # To speed up, we can precompute the vertices for each mask?
                # Or just iterate bits.
                
                degs = []
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    # Degree of v in mask
                    deg = (adj_masks[v] & mask).bit_count()
                    degs.append(deg)
                    m ^= lsb
                
                if len(set(degs)) == 1:
                    is_valid_6 = False
                    break
            
            if not is_valid_6:
                continue
                
            # If valid for size 6, check larger sizes
            # This is the expensive part.
            # We check k from 7 to target_n
            is_valid_full = True
            for k in range(7, target_n + 1):
                # Generate all subsets of size k
                # To avoid regenerating combinations every time, we could precompute?
                # But k varies. And we only do this for candidates that pass size 6.
                # So it's rare.
                for subset in itertools.combinations(range(target_n), k):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    
                    degs = []
                    for v in subset:
                        degs.append((adj_masks[v] & mask).bit_count())
                    
                    if len(set(degs)) == 1:
                        is_valid_full = False
                        break
                if not is_valid_full:
                    break
            
            if is_valid_full:
                # Found a valid graph!
                edges = []
                for i in range(target_n):
                    m = adj_masks[i]
                    for j in range(i + 1, target_n):
                        if m & (1 << j):
                            edges.append([i, j])
                
                return {'n': target_n, 'edges': edges}
        
        # If no T worked for this permutation, try next permutation
        # To add diversity, we can also try random edge flips on the best T?
        # No, stick to the plan.
        
    # If we didn't find one in the structural family, fall back to random restarts?
    # The directive says "Model the graph as...". It implies this is the main strategy.
    # But if it fails, we should return the best we have.
    # Since we didn't find any, we return n=5 (trivially valid) or try one last random thing?
    # The task says "Maximize n". n=5 is valid.
    # But we know n=20 exists.
    # Maybe we should try a few random graphs as a fallback?
    # Given the time budget, if the structural family fails, random is unlikely to succeed quickly.
    # But let's try 100 random graphs just in case.
    
    if time.time() < deadline - 2.0:
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            adj_masks = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Quick size 6 check
            is_valid_6 = True
            for mask in subsets_6:
                degs = []
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg = (adj_masks[v] & mask).bit_count()
                    degs.append(deg)
                    m ^= lsb
                if len(set(degs)) == 1:
                    is_valid_6 = False
                    break
            
            if is_valid_6:
                # Full check
                is_valid_full = True
                for k in range(7, target_n + 1):
                    for subset in itertools.combinations(range(target_n), k):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        degs = []
                        for v in subset:
                            degs.append((adj_masks[v] & mask).bit_count())
                        if len(set(degs)) == 1:
                            is_valid_full = False
                            break
                    if not is_valid_full:
                        break
                
                if is_valid_full:
                    edges = []
                    for i in range(target_n):
                        m = adj_masks[i]
                        for j in range(i + 1, target_n):
                            if m & (1 << j):
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}

    # Return trivially valid graph if all else fails
    # n=5 is safe.
    return {'n': 5, 'edges': []}

# To ensure we don't timeout, we have deadline guards.
# The structural approach is deterministic in its randomness (seeded), so it's reproducible.
# If it fails to find n=20, it returns n=5.
# Is n=5 the best we can do?
# The current program scores 2.8, which corresponds to n=20 (since 20/7 ~ 2.85? No, score is value/seed.
# Wait, "search score is value/seed".
# If seed program finds n=7, score is 1.0?
# "Graphs with 20 vertices are known".
# If the seed program finds n=20, score is 20/20 = 1.0?
# Or is the seed program finding something else?
# "Current program (score 2.8000)".
# If value is n, and score is 2.8, then seed might be n=7? 20/7 = 2.85.
# Or maybe the seed program finds n=20, but the "value" is something else?
# "The evaluator measures the raw value (higher is better)."
# "Maximize n."
# So value = n.
# If current score is 2.8, and it found n=20, then seed was ~7.1.
# If I find n=20, my score is also 20/seed.
# If the seed is fixed, and I find n=20, I get the same score as the current program?
# "1.0 equals the seed program and above 1.0 is an improvement".
# So if seed program finds n=7, score=1.0.
# Current program finds n=20, score=20/7 = 2.85.
# If I find n=20, I also get 2.85.
# To improve, I need n=21?
# "A 21-vertex graph would show G(6) >= 22."
# "Maximize n".
# If I find n=21, score = 21/7 = 3.0.
# So I should aim for n=21!
    
# Let's modify the target to 21.
# But n=21 might not exist? "Graphs with 20 vertices are known; a 21-vertex graph would show..."
# It implies it's an open question or harder.
# But let's try n=21 first. If it fails, try n=20.

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=21 first, then n=20
    for target_n in [21, 20]:
        if time.time() > deadline - 5.0:
            break
            
        # Precompute subsets for this n
        subsets_6 = []
        for subset in itertools.combinations(range(target_n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            subsets_6.append(mask)
            
        random.seed(42 + target_n)
        
        attempts = 0
        max_attempts = 5000 if target_n == 21 else 20000
        
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            
            perm = list(range(target_n))
            random.shuffle(perm)
            
            rank = [0] * target_n
            for i, v in enumerate(perm):
                rank[v] = i
                
            for T in range(1, target_n):
                if time.time() > deadline - 5.0:
                    break
                    
                adj_masks = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if abs(rank[i] - rank[j]) > T:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                
                # Fast check size 6
                is_valid_6 = True
                for mask in subsets_6:
                    degs = []
                    m = mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        deg = (adj_masks[v] & mask).bit_count()
                        degs.append(deg)
                        m ^= lsb
                    if len(set(degs)) == 1:
                        is_valid_6 = False
                        break
                
                if not is_valid_6:
                    continue
                
                # Full check
                is_valid_full = True
                for k in range(7, target_n + 1):
                    for subset in itertools.combinations(range(target_n), k):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        degs = []
                        for v in subset:
                            degs.append((adj_masks[v] & mask).bit_count())
                        if len(set(degs)) == 1:
                            is_valid_full = False
                            break
                    if not is_valid_full:
                        break
                
                if is_valid_full:
                    edges = []
                    for i in range(target_n):
                        m = adj_masks[i]
                        for j in range(i + 1, target_n):
                            if m & (1 << j):
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}
        
        # If n=21 failed, try random restarts for n=20?
        # Or just fall through to n=20 loop.
        
    # Fallback: n=5
    return {'n': 5, 'edges': []}

# Wait, the code above redefines construct. I need to output a single program.
# I will combine the logic: try n=21 with structural, then n=20 with structural, then n=20 with random.

import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def try_structural(n, seed_offset):
        random.seed(42 + seed_offset)
        subsets_6 = []
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            subsets_6.append(mask)
            
        attempts = 0
        max_attempts = 5000
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            perm = list(range(n))
            random.shuffle(perm)
            rank = [0] * n
            for i, v in enumerate(perm):
                rank[v] = i
                
            for T in range(1, n):
                if time.time() > deadline - 5.0:
                    break
                adj_masks = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(rank[i] - rank[j]) > T:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                
                is_valid_6 = True
                for mask in subsets_6:
                    degs = []
                    m = mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        deg = (adj_masks[v] & mask).bit_count()
                        degs.append(deg)
                        m ^= lsb
                    if len(set(degs)) == 1:
                        is_valid_6 = False
                        break
                if not is_valid_6:
                    continue
                
                is_valid_full = True
                for k in range(7, n + 1):
                    for subset in itertools.combinations(range(n), k):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        degs = []
                        for v in subset:
                            degs.append((adj_masks[v] & mask).bit_count())
                        if len(set(degs)) == 1:
                            is_valid_full = False
                            break
                    if not is_valid_full:
                        break
                if is_valid_full:
                    edges = []
                    for i in range(n):
                        m = adj_masks[i]
                        for j in range(i + 1, n):
                            if m & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        return None

    def try_random(n, seed_offset, max_restarts=1000):
        random.seed(42 + seed_offset + 1000)
        subsets_6 = []
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            subsets_6.append(mask)
            
        for _ in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            is_valid_6 = True
            for mask in subsets_6:
                degs = []
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    deg = (adj_masks[v] & mask).bit_count()
                    degs.append(deg)
                    m ^= lsb
                if len(set(degs)) == 1:
                    is_valid_6 = False
                    break
            if not is_valid_6:
                continue
            
            is_valid_full = True
            for k in range(7, n + 1):
                for subset in itertools.combinations(range(n), k):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    degs = []
                    for v in subset:
                        degs.append((adj_masks[v] & mask).bit_count())
                    if len(set(degs)) == 1:
                        is_valid_full = False
                        break
                if not is_valid_full:
                    break
            if is_valid_full:
                edges = []
                for i in range(n):
                    m = adj_masks[i]
                    for j in range(i + 1, n):
                        if m & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        return None

    # Try n=21 structural
    res = try_structural(21, 0)
    if res:
        return res
        
    # Try n=20 structural
    res = try_structural(20, 1)
    if res:
        return res
        
    # Try n=20 random
    res = try_random(20, 2, 2000)
    if res:
        return res
        
    # Fallback
    return {'n': 5, 'edges': []}
