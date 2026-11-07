import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Helper to check if a set of sums is valid (no divisibility)
    # This is used for final validation or initial checks
    def is_valid_sums(sums):
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i + 1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True

    # Efficient check for adding x to current_elements with current_sums
    def can_add(x, current_sums):
        # Generate new sums
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        # Check for duplicates between new_sums and current_sums
        if new_sums & current_sums:
            return None
        
        # Check for duplicates within new_sums (handled by set, but we need to ensure 
        # that the mapping from subsets to sums is injective. 
        # Actually, if s1+x == s2+x then s1==s2. 
        # If x == s+x then s=0 (not possible as sums are non-empty positive).
        # If x == s1+x1 and x = s2+x2... 
        # The set `new_sums` automatically collapses duplicates.
        # We need to ensure that no two *different* subsets produce the same sum.
        # If len(new_sums) != len(current_sums) + 1, there's a collision.
        # Wait, new_sums has size at most len(current_sums) + 1.
        # If len(new_sums) < len(current_sums) + 1, it means some s1+x == s2+x (impossible if s1!=s2) 
        # or s1+x == x (impossible) 
        # or s1+x == s2+x' where x' is another new sum? No, we are adding x once.
        # The collision happens if s1+x == s2+x is false, but s1+x == x is false.
        # The only way new_sums has fewer elements is if s1+x == s2+x (implies s1=s2) 
        # OR if s1+x equals an existing sum in current_sums (checked above).
        # OR if two different s1, s2 produce the same s+x? No, s+x is unique for s.
        # So len(new_sums) should be exactly len(current_sums) + 1.
        if len(new_sums) != len(current_sums) + 1:
            return None
            
        all_sums = current_sums.union(new_sums)
        
        # Check divisibility
        # We only need to check pairs involving at least one element from new_sums
        # Let N = new_sums, O = current_sums
        # Check N vs N
        # Check N vs O
        
        # Optimization: Sort new_sums
        n_list = sorted(new_sums)
        o_list = sorted(current_sums)
        
        # Check within new_sums
        for i in range(len(n_list)):
            ni = n_list[i]
            for j in range(i + 1, len(n_list)):
                nj = n_list[j]
                if nj % ni == 0:
                    return None

        # Check new_sums vs current_sums
        # For each n in new_sums, check against all o in current_sums
        # If n > o: check n % o == 0
        # If n < o: check o % n == 0
        
        for n in new_sums:
            for o in current_sums:
                if n > o:
                    if n % o == 0:
                        return None
                else:
                    if o % n == 0:
                        return None

        return all_sums

    # DFS to find a valid set
    def dfs(current_elements, current_sums, candidates, depth, max_depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return

        if len(current_elements) > best_size:
            best_set = list(current_elements)
            best_size = len(current_elements)
            
        if depth == max_depth:
            return

        # Pruning: if remaining candidates can't beat best_size, stop
        # This is a weak prune, but we rely on time limit more.
        
        # Heuristic: Try larger candidates first
        # candidates is already sorted descending
        
        for x in candidates:
            new_sums = can_add(x, current_sums)
            if new_sums is not None:
                # Recurse
                # Create a new candidate list excluding x
                # To avoid copying the whole list every time, we can pass an index or filter
                # Given the small size (max 256), filtering is okay, but let's be efficient
                new_candidates = [c for c in candidates if c != x]
                dfs(current_elements + [x], new_sums, new_candidates, depth + 1, max_depth)
                
                if time.time() > deadline:
                    return

    best_set = []
    best_size = 0

    # Candidate pool: We focus on larger numbers as they are more likely to be valid
    # Start with a wide range, but prioritize the upper end
    # If we start with [256, 1], the first few picks will be large.
    
    # Strategy: 
    # 1. Try to find a set of size 10, 9, 8...
    # 2. Use DFS with increasing depth limits or just a deep DFS with time limit.
    
    # Let's use a deep DFS but with a "best effort" update.
    # We will run the DFS on the full range [256, 1] but sorted descending.
    
    # To speed up, we can pre-filter candidates that are obviously bad?
    # No, let's just run the DFS.
    
    # To avoid deep recursion overhead and copying lists, we can use an index-based approach
    # or just accept the copy cost for small depths.
    
    # Let's try to find a set of size 10 first. If not found, 9, etc.
    # But DFS doesn't guarantee finding size 10 if it gets stuck in size 9 branches.
    # However, if a size 10 set exists, DFS *will* find it if it explores enough.
    
    # Better approach: Iterative Deepening DFS (IDDFS)
    # Try max_depth = 10, then 9, etc.
    # But if 10 is impossible, we waste time.
    
    # Alternative: Just run DFS with max_depth = 12 (or higher) and update best as we go.
    # The time limit will cut it off.
    
    # To ensure we find *large* sets, we should prioritize paths that look promising.
    # Greedy initialization:
    # Find a good starting set using greedy, then use it to prune or start DFS?
    
    # Let's combine:
    # 1. Greedy to find a baseline set (size ~5-7).
    # 2. DFS to extend or find better sets.
    
    def greedy_construct(candidates):
        cur = []
        cur_sums = set()
        for x in candidates:
            new_sums = can_add(x, cur_sums)
            if new_sums is not None:
                cur.append(x)
                cur_sums = new_sums
        return cur, cur_sums

    # Run greedy to get a baseline
    baseline_candidates = list(range(256, 1, -1))
    baseline_set, baseline_sums = greedy_construct(baseline_candidates)
    
    if len(baseline_set) > best_size:
        best_set = baseline_set
        best_size = len(baseline_set)

    # Now, use DFS to try to find larger sets.
    # We can start DFS from the empty set, but prune if current depth + remaining < best_size?
    # No, that's not right.
    
    # Let's run DFS from scratch, but with a time limit.
    # The DFS will naturally explore larger numbers first.
    
    # To make it faster, we can limit the branching factor in early stages?
    # No, let's trust the time limit.
    
    # One optimization: If we find a set of size 10, we can stop?
    # The task says "Maximize set size". 10 is likely the max or close to it.
    # Let's see if we can hit 10.
    
    # Let's run the DFS
    # To avoid excessive recursion depth and overhead, we will use a iterative stack or 
    # just rely on Python's recursion limit (set to 10000).
    
    # We will run the DFS on the full candidate list.
    # To speed up, we can sort candidates by "promisingness". 
    # Larger numbers are generally better.
    
    # Let's add a small randomization to avoid getting stuck in the same local optima 
    # if we were doing multiple restarts, but since we have one long run, 
    # deterministic DFS with large-first ordering is a good start.
    
    # However, deterministic DFS might miss valid sets if the "large first" order 
    # leads to a dead end quickly.
    
    # Let's do a few random restarts of the DFS with different candidate shuffles 
    # or different starting points.
    
    # Strategy:
    # 1. Greedy (done).
    # 2. DFS with large-first ordering.
    # 3. If time remains, DFS with slightly randomized ordering.
    
    # Let's implement the DFS carefully.
    
    def run_dfs(candidates, max_depth):
        nonlocal best_set, best_size
        dfs([], set(), candidates, 0, max_depth)

    # Try to find size 10
    # If we find size 10, we might still try for 11 if time permits, 
    # but 11 is very hard.
    
    # Let's run DFS with max_depth = 11
    # But we want to update best_set as we go.
    
    # To prevent the DFS from spending all time on a branch that only yields size 5,
    # we can't easily prune based on "potential" without lookahead.
    
    # Let's just run it.
    
    # One more optimization: 
    # If current_sums is large, can_add is slow.
    # But for k < 10, it's fine.
    
    # Let's run the DFS.
    # We will use the full range 256..1.
    
    # To ensure we don't miss the "best" set, we should probably 
    # run the DFS multiple times with different seeds/orderings if time permits.
    
    # Let's start with the deterministic large-first DFS.
    
    # To avoid hitting the recursion limit or overhead, we can use an iterative approach?
    # No, recursion is fine for depth 10-12.
    
    # Let's go.
    
    # We need to be careful with `candidates` list copying.
    # For depth 10, list size 256, copying is O(256). 
    # Number of nodes? 
    # If branching factor is small (due to constraints), it's fine.
    
    # Let's run it.
    
    # We will wrap the DFS in a try-except for RecursionError just in case.
    
    try:
        # Run DFS with max depth 12 to be safe, but we hope to find 10.
        # If we find 10, we can still keep searching for 11.
        run_dfs(list(range(256, 1, -1)), 12)
    except RecursionError:
        pass

    # If we still have time, try some randomized restarts to escape local optima
    if time.time() < deadline - 5:
        for i in range(5):
            if time.time() > deadline - 5:
                break
            # Shuffle candidates slightly? 
            # Or pick a random subset of top candidates?
            # Let's shuffle the full list.
            random_candidates = list(range(256, 1, -1))
            random.shuffle(random_candidates)
            # But we want to bias towards large numbers.
            # So, take top 100, shuffle them, then append the rest?
            top = random_candidates[:100]
            rest = random_candidates[100:]
            random.shuffle(top)
            mixed = top + rest
            
            try:
                run_dfs(mixed, 12)
            except RecursionError:
                pass

    # Final validation
    if best_set:
        # Double check validity
        if not is_valid_sums(compute_sums(best_set)):
            # Fallback to greedy result or simple valid set
            if baseline_set and is_valid_sums(compute_sums(baseline_set)):
                best_set = baseline_set
            else:
                best_set = [100, 101]
    
    # Ensure types
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set

def compute_sums(elements):
    sums = set()
    for x in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums = sums.union(new_sums)
    return sums

# Re-defining compute_sums inside construct to avoid global namespace issues if needed, 
# but it's cleaner outside. However, the prompt asks for a single function `construct`.
# I will move compute_sums inside construct or make it a nested helper.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def compute_sums(elements):
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums

    def is_valid_sums(sums):
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i + 1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True

    def can_add(x, current_sums):
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        
        if new_sums & current_sums:
            return None
        
        if len(new_sums) != len(current_sums) + 1:
            return None
            
        all_sums = current_sums.union(new_sums)
        
        n_list = sorted(new_sums)
        
        for i in range(len(n_list)):
            ni = n_list[i]
            for j in range(i + 1, len(n_list)):
                nj = n_list[j]
                if nj % ni == 0:
                    return None

        for n in new_sums:
            for o in current_sums:
                if n > o:
                    if n % o == 0:
                        return None
                else:
                    if o % n == 0:
                        return None

        return all_sums

    def dfs(current_elements, current_sums, candidates, depth, max_depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return

        if len(current_elements) > best_size:
            best_set = list(current_elements)
            best_size = len(current_elements)
            
        if depth == max_depth:
            return
        
        for x in candidates:
            new_sums = can_add(x, current_sums)
            if new_sums is not None:
                new_candidates = [c for c in candidates if c != x]
                dfs(current_elements + [x], new_sums, new_candidates, depth + 1, max_depth)
                
                if time.time() > deadline:
                    return

    def greedy_construct(candidates):
        cur = []
        cur_sums = set()
        for x in candidates:
            if time.time() > deadline:
                break
            new_sums = can_add(x, cur_sums)
            if new_sums is not None:
                cur.append(x)
                cur_sums = new_sums
        return cur, cur_sums

    best_set = []
    best_size = 0

    baseline_candidates = list(range(256, 1, -1))
    baseline_set, baseline_sums = greedy_construct(baseline_candidates)
    
    if len(baseline_set) > best_size:
        best_set = baseline_set
        best_size = len(baseline_set)

    try:
        run_dfs = lambda c, m: dfs([], set(), c, 0, m)
        run_dfs(list(range(256, 1, -1)), 12)
    except RecursionError:
        pass

    if time.time() < deadline - 5:
        for i in range(5):
            if time.time() > deadline - 5:
                break
            random_candidates = list(range(256, 1, -1))
            random.shuffle(random_candidates)
            top = random_candidates[:100]
            rest = random_candidates[100:]
            random.shuffle(top)
            mixed = top + rest
            
            try:
                dfs([], set(), mixed, 0, 12)
            except RecursionError:
                pass

    if best_set:
        if not is_valid_sums(compute_sums(best_set)):
            if baseline_set and is_valid_sums(compute_sums(baseline_set)):
                best_set = baseline_set
            else:
                best_set = [100, 101]
    
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set

