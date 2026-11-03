import time
import random
import math
import sys

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _generate_rows(n_rows, k, max_scope, rng, target_count=2000):
    """
    Generate a pool of valid rows (Sidon sets) with max entry <= max_scope.
    """
    rows = []
    # We need to generate enough rows to have a good chance of finding a clique of size n_rows.
    # A clique of size 7 in a random graph with edge probability p requires roughly log(n)/p nodes.
    # Here, the "edge" is disjointness.
    
    # Strategy: Randomized backtracking to generate rows quickly.
    attempts = 0
    max_attempts = target_count * 50
    
    while len(rows) < target_count and attempts < max_attempts:
        attempts += 1
        row = [0]
        last = 0
        success = True
        
        for j in range(1, k + 1):
            # Try to pick the next element
            # We want to keep it small to stay within max_scope
            # Heuristic: pick from a small range of candidates
            candidates = []
            # Look ahead a bit to ensure we can finish
            # Simple greedy with some randomness
            start_val = last + 1
            end_val = min(max_scope, last + 20) # Limit search width for speed
            
            if start_val > end_val:
                success = False
                break
                
            # Sample candidates
            # To ensure diversity, we can shuffle the range or pick random valid ones
            # But for speed, let's just pick the first few valid ones and choose randomly among them
            valid_vals = []
            for val in range(start_val, end_val + 1):
                # Check if adding val to current row creates a conflict within the row
                # Note: We don't check against other rows here, just internal Sidon property
                # Wait, the row itself must be a Sidon set.
                # Differences between new val and existing vals must be distinct from each other and not 0.
                # Actually, Sidon set definition: all differences a_j - a_i (i<j) are distinct.
                # So for the new val x, the differences x - row[i] for all i < j must be distinct.
                
                new_diffs = []
                conflict = False
                for i in range(len(row)):
                    d = val - row[i]
                    if d in new_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if not conflict:
                    valid_vals.append(val)
                    
                    # We don't need all valid vals, just a few to choose from
                    if len(valid_vals) >= 10:
                        break
            
            if not valid_vals:
                success = False
                break
                
            # Choose one randomly to diversify
            chosen_val = rng.choice(valid_vals)
            row.append(chosen_val)
            last = chosen_val
            
            # Check if we are already exceeding max_scope? No, we bound end_val.
            # But we should ensure we can still finish.
            # Simple check: remaining slots * (min step 1) <= max_scope - last
            if last + (k + 1 - len(row)) > max_scope:
                # We might not be able to finish with distinct integers if we are too tight?
                # Actually, min step is 1, so we need k+1-len(row) more steps.
                # If last + (k+1-len(row)) > max_scope, we are in trouble.
                # But since we picked from valid_vals which are <= max_scope, and we need strictly increasing,
                # this check is good.
                pass

        if success and len(row) == k + 1 and row[-1] <= max_scope:
            # Verify it's a valid Sidon set
            diffs = _row_diffs(row)
            if len(diffs) == len(row) * (len(row) - 1) // 2:
                # Check for duplicates in diffs? _row_diffs returns a set, so len check confirms uniqueness.
                # But we also need to ensure no zero diffs (guaranteed by strictly increasing)
                rows.append(row)
                
    return rows

def _max_clique_size(adj, n, target_size, rng):
    """
    Find if there is a clique of size target_size in the graph defined by adj.
    adj[i][j] is 1 if i and j are compatible (disjoint diffs).
    Uses a simple branch and bound / backtracking.
    """
    if n < target_size:
        return False, []
        
    # Order vertices by degree (descending) to prune faster?
    # For simplicity, we stick to index order but maybe shuffle if we want randomness?
    # Actually, for finding *any* clique, order matters for speed.
    # Let's use a greedy coloring bound or just simple backtracking.
    # Given small target_size (7) and potentially large n (2000), we need to be smart.
    
    # We can use a recursive search.
    # To make it faster, we can use the property that if the remaining vertices
    # cannot form a clique of the required size, we prune.
    
    # A better approach for "Is there a clique of size K":
    # We can try to pick vertices one by one.
    # Candidate set C.
    # Pick v in C.
    # Recurse on C intersect neighbors(v).
    
    # To avoid exponential explosion, we can limit the branching factor or use randomization.
    # Since we just need to find *one* clique, randomization helps escape bad local choices.
    
    def dfs(candidates, current_clique):
        if len(current_clique) == target_size:
            return current_clique[:]
            
        if not candidates:
            return None
            
        # Pruning: if len(candidates) < target_size - len(current_clique), return None
        if len(candidates) < target_size - len(current_clique):
            return None
            
        # Heuristic: pick a vertex with high degree in the subgraph induced by candidates
        # For speed, we can just pick the first one, but that might be slow.
        # Let's pick a random one to explore different branches?
        # Or pick the one with max degree in candidates.
        
        # Calculate degrees in subgraph
        # This is O(N^2) per step, which might be slow if N is large.
        # But N starts at ~2000 and shrinks quickly.
        
        # Optimization: Precompute neighbors?
        # adj is a list of lists or a set of sets?
        # Let's assume adj is a list of sets for efficiency.
        
        # Pick a vertex
        # Random selection from candidates might be better for finding *a* solution quickly
        # if the graph is dense enough.
        idx = rng.choice(range(len(candidates)))
        v = candidates[idx]
        
        # New candidates: neighbors of v that are also in candidates
        # And have index > v? No, graph is undirected.
        # We must ensure we don't pick the same vertex twice.
        # Standard clique search:
        # C = candidates
        # Pick v from C
        # C' = C \ {v} intersect N(v)
        
        new_candidates = []
        for u in candidates:
            if u != v and u in adj[v]:
                new_candidates.append(u)
                
        # Recurse
        res = dfs(new_candidates, current_clique + [v])
        if res:
            return res
            
        # If this branch fails, try another vertex?
        # In a simple DFS, we would loop through all vertices in candidates.
        # But that's expensive.
        # Let's try a few random vertices.
        
        # To keep it simple and fast, we will just do a limited random restart strategy outside,
        # or a limited depth-first search here.
        
        # Let's try to iterate through a few random candidates from the original set
        # This function is called many times, so it should be fast.
        
        # Actually, a better heuristic:
        # If we fail with v, try another v'.
        # But we can't easily "undo" the selection in this recursive structure without passing state.
        
        # Let's modify:
        # We will shuffle candidates and try the first few.
        
        return None

    # The above DFS is too weak if it only tries one path.
    # Let's implement a more robust search.
    
    # We will use a stack-based or recursive search that tries multiple options.
    # But to respect the time limit, we need to be careful.
    
    # Alternative: Use a greedy heuristic to find a large clique, then check if it's size K.
    # Greedy:
    # 1. Start with all vertices.
    # 2. Pick vertex with max degree.
    # 3. Add to clique.
    # 4. Remove non-neighbors.
    # 5. Repeat.
    # This finds *a* clique, not necessarily the maximum.
    # If the graph is dense (high probability of disjointness), this works well.
    
    # Let's combine:
    # 1. Try Greedy multiple times with different random starts.
    # 2. If Greedy fails, try Randomized Backtracking.
    
    # For now, let's implement a randomized backtracking that explores a few branches.
    
    def search(candidates, current_clique, depth_limit):
        if len(current_clique) == target_size:
            return current_clique[:]
        
        if depth_limit <= 0 or not candidates:
            return None
            
        if len(candidates) < target_size - len(current_clique):
            return None
            
        # Pick a random vertex
        # To improve success rate, pick one with high degree
        # But calculating degree is expensive.
        # Let's just pick random.
        
        # Shuffle candidates to explore different branches
        # But we can't shuffle the list passed in if we want to keep it for other branches?
        # We create a new list.
        
        # Heuristic: Order candidates by degree in the subgraph?
        # Too slow.
        
        # Let's try the first few random vertices.
        # We will try up to `branch_factor` vertices at each level.
        branch_factor = 3
        
        # Get indices to try
        # We want to try diverse vertices.
        # Just pick random indices.
        
        # To avoid re-visiting, we just recurse on the subset.
        
        # Pick a vertex
        # Let's try to pick the vertex with the highest "potential" (degree).
        # Approximate degree: len(adj[v] & set(candidates))
        
        # For speed, we'll just pick random, but maybe sort by index to be deterministic?
        # No, random is better for escape.
        
        # Let's try a few random vertices from candidates.
        # We will try up to branch_factor distinct vertices.
        
        tried = set()
        for _ in range(min(branch_factor, len(candidates))):
            if not candidates:
                break
            idx = rng.randint(0, len(candidates) - 1)
            v = candidates[idx]
            if v in tried:
                continue
            tried.add(v)
            
            # Next candidates
            new_candidates = [u for u in candidates if u != v and u in adj[v]]
            
            res = search(new_candidates, current_clique + [v], depth_limit - 1)
            if res:
                return res
                
        return None
        
    # Initial candidates are all vertices
    # We need to map indices to vertices?
    # adj is indexed by vertex index (0 to n-1).
    # So candidates is a list of indices.
    
    initial_candidates = list(range(n))
    # Shuffle initial candidates to randomize the search space
    rng.shuffle(initial_candidates)
    
    # Depth limit should be target_size
    res = search(initial_candidates, [], target_size)
    
    if res:
        return True, res
    return False, []

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # The best known scope for (7,5) is 112.
    # Simple counting bound is 105.
    # Let's try to find a solution with scope < 112.
    # We will iterate max_scope from a low value upwards.
    
    # Estimate a good starting max_scope.
    # If n_rows=7, k=5, scope 105 is the bound.
    # Let's start at 105 and go up.
    
    # However, generating rows for a specific max_scope is expensive.
    # We can generate rows for a larger max_scope and filter?
    # No, the structure of the row depends on the bound.
    
    # Strategy:
    # 1. Try max_scope = 105.
    # 2. If no clique found, try 106, 107, etc.
    # 3. Keep the best valid set found.
    
    # To speed up, we can generate a pool of rows for max_scope = 120 (safe upper bound)
    # and then filter them?
    # No, if we generate rows with max entry 120, they might not fit in scope 105.
    # We need rows with max entry <= current_target.
    
    # So we must regenerate or filter.
    # Regenerating is expensive.
    # Let's generate a large pool for a generous bound (e.g., 130) and then
    # filter subsets that fit in smaller bounds?
    # This is tricky because a row with max entry 120 is invalid for scope 110.
    
    # Better:
    # Iterate target_scope from 105 to 130.
    # For each target_scope:
    #   1. Generate rows with max entry <= target_scope.
    #   2. Build graph.
    #   3. Find clique.
    #   4. If found, update best and break (or continue to see if we can find lower? No, we go up).
    #   Wait, we want MINIMUM scope. So we should go DOWN?
    #   But we don't know if 105 is possible.
    #   We know 112 is possible.
    #   So we can try 111, 110, ... 105.
    #   If we find one, we can stop?
    #   But maybe 104 is possible?
    #   The problem says "largest entry as small as possible".
    #   So we should try to minimize it.
    
    # Let's try a binary search or linear search downwards from a known valid bound.
    # Known valid: 112.
    # Let's try 111, 110, ...
    
    # But generating rows for each scope is slow.
    # Can we reuse rows?
    # A row valid for scope M is also valid for scope M+1.
    # So we can generate rows for scope 112.
    # Then filter for scope 111 (remove rows with max entry > 111).
    # Then filter for 110.
    
    # This is efficient!
    
    # 1. Generate rows for max_scope = 120 (generous).
    # 2. Sort rows by max entry.
    # 3. Iterate target_scope from 105 to 120.
    #    - Filter rows to those with max entry <= target_scope.
    #    - If enough rows, build graph and find clique.
    #    - If clique found, update best and break?
    #      We want the MINIMUM scope.
    #      So we should iterate from LOW to HIGH?
    #      If we find a clique at 105, that's the best possible (given bound).
    #      But we don't know if 105 is possible.
    #      So we should try 105, 106, ...
    
    # However, if we start with rows generated for 120, we have all rows with max <= 120.
    # We can filter them.
    
    # Let's generate a large pool for max_scope = 130.
    # This ensures we have plenty of candidates for lower scopes.
    
    print(f"Generating rows for max_scope 130...", file=sys.stderr)
    pool = _generate_rows(n_rows, k, 130, rng, target_count=5000)
    
    if not pool:
        # Fallback to greedy
        rows = _greedy_construct(n_rows, k, rng, bias=1.0)
        if _valid(rows, k):
            return rows
        return [[0, 1, 2, 3, 4, 5]] * n_rows

    # Sort pool by max entry
    pool.sort(key=lambda r: r[-1])
    
    # We want to find the minimum scope.
    # Let's try scopes from 105 upwards.
    # But we can stop if we find a valid set.
    
    # To be safe, let's try a few scopes.
    # If we find a set at scope S, we can try S-1, S-2...
    # But generating the graph is the expensive part.
    
    # Let's just iterate from 105 to 130.
    # For each scope, we filter the pool.
    # If the filtered pool is too small, skip.
    
    # We need to be careful with time.
    # Building the graph is O(N^2 * K^2) roughly?
    # N=5000, K=5.
    # 5000^2 = 25M. Too slow for Python.
    
    # We need to limit the pool size or the graph construction.
    # Let's reduce target_count to 1000.
    # 1000^2 = 1M. Manageable.
    
    # Let's regenerate with smaller count?
    # Or just take a subset of the generated pool.
    
    # Let's take the first 1000 rows from the sorted pool (smallest entries).
    # These are the most likely to form a low-scope clique.
    
    active_pool = pool[:1000]
    
    # We will try scopes from 105 to 130.
    # For each scope, we filter active_pool.
    
    # To avoid rebuilding the graph from scratch every time,
    # we can just check if a clique exists in the filtered subgraph.
    
    # But the graph changes (vertices removed).
    # So we must rebuild.
    
    # To optimize, we can stop early if we find a solution.
    
    # Let's try scopes: 105, 106, ..., 130
    # But we have time limits.
    # Let's try a few.
    
    # Actually, if we find a valid set at scope 112, we can return it.
    # We don't need to find 105 if it takes too long.
    # But the score is better for lower scope.
    
    # Strategy:
    # 1. Try scope 110.
    # 2. If fail, try 111.
    # 3. If fail, try 112.
    # 4. If fail, try 115.
    # 5. If fail, use the best greedy result.
    
    # Let's define a list of target scopes to try.
    target_scopes = [105, 108, 110, 112, 115, 120]
    
    for scope in target_scopes:
        if time.time() > deadline - 10:
            break
            
        # Filter pool
        filtered = [r for r in active_pool if r[-1] <= scope]
        
        if len(filtered) < n_rows:
            continue
            
        # Build graph
        # adj[i] is set of j such that filtered[i] and filtered[j] are disjoint.
        n = len(filtered)
        adj = [set() for _ in range(n)]
        
        # Precompute diffs for all rows in filtered
        diffs_list = [_row_diffs(r) for r in filtered]
        
        for i in range(n):
            for j in range(i + 1, n):
                if not diffs_list[i].intersection(diffs_list[j]):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Find clique
        found, clique_indices = _max_clique_size(adj, n, n_rows, rng)
        
        if found:
            # Reconstruct rows
            rows = [filtered[i] for i in clique_indices]
            
            # Verify
            if _valid(rows, k):
                current_scope = _scope(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = rows
                    
                # If we found a solution at a low scope, we might want to try even lower?
                # But we are iterating upwards.
                # If we found one at 105, it's the best we can hope for (given bound).
                # So we can break.
                if scope == 105:
                    break
                # If we found one at 110, maybe 108 is possible?
                # But we already tried 108 and failed?
                # Yes, we iterate in order.
                # So if we found one at 110, it means 105 and 108 failed.
                # So 110 is the best found so far.
                # We can break and return this.
                break
                
    if best_rows:
        return best_rows
        
    # Fallback
    rows = _greedy_construct(n_rows, k, rng, bias=1.0)
    if _valid(rows, k):
        return rows
    return [[0, 1, 2, 3, 4, 5]] * n_rows

def _greedy_construct(n_rows, k, rng, bias=1.0):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_x = last + 60
            while x <= max_x and len(candidates) < 10:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
            if not candidates:
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        rows.append(row)
    return rows

def _sa_optimize(rows, k, deadline, rng, T0=10.0, cooling=0.9995, min_T=0.01):
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    
    row_diff_sets = [_row_diffs(r) for r in rows]
    global_diffs = set()
    for ds in row_diff_sets:
        global_diffs.update(ds)
    
    scope = _scope(rows)
    best_scope = scope
    best_rows = [list(r) for r in rows]
    
    T = T0
    penalty = 5000
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = rows[i][j]
        prev_val = rows[i][j - 1]
        next_val = rows[i][j + 1] if j < k else scope + 200
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        if rng.random() < 0.7 and old_val > lo:
            max_drop = old_val - lo
            drop = rng.randint(1, max(1, max_drop // 3))
            new_val = old_val - drop
        else:
            span = hi - lo
            if span <= 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.2)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        old_row = rows[i]
        old_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(old_row[t] - old_val)
                old_diffs.add(d)
        
        new_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(old_row[t] - new_val)
                new_diffs.add(d)
        
        remaining = global_diffs - old_diffs
        collisions = len(new_diffs & remaining)
        
        if new_val > scope:
            new_scope = new_val
        elif old_val == scope:
            new_scope = max(r[-1] for r in rows)
        else:
            new_scope = scope
        
        old_cost = scope + penalty * 0
        new_cost = new_scope + penalty * collisions
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            global_diffs = remaining | new_diffs
            rows[i][j] = new_val
            scope = new_scope
            
            if collisions == 0 and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if rng.random() < 0.03:
                    T = min(T * 1.5, 15.0)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _backtrack_construct(n_rows, k, max_scope, rng):
    used = set()
    rows = [None] * n_rows
    
    def backtrack(row_idx, col_idx, current_row):
        if row_idx == n_rows:
            return True
        
        if col_idx == 0:
            current_row = [0]
            return backtrack(row_idx, 1, current_row)
        
        if col_idx > k:
            rows[row_idx] = current_row
            row_diffs = _row_diffs(current_row)
            prev_used = set(used)
            used.update(row_diffs)
            
            if backtrack(row_idx + 1, 0, []):
                return True
            
            used.clear()
            used.update(prev_used)
            return False
        
        prev_val = current_row[-1]
        candidates = []
        for val in range(prev_val + 1, max_scope + 1):
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used:
                    valid = False
                    break
            if valid:
                candidates.append(val)
                if len(candidates) > 20:
                    break
        
        if not candidates:
            return False
        
        rng.shuffle(candidates)
        
        for val in candidates:
            current_row.append(val)
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            current_row.pop()
        
        return False
    
    if backtrack(0, 0, []):
        return rows
    return None
