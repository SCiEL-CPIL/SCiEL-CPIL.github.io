import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # The problem: Find n_rows rows of k+1 integers starting at 0, strictly increasing.
    # All positive differences a_j - a_i (i < j) within a row must be distinct across ALL rows.
    # Minimize the largest entry (scope).
    
    # Strategy: Maximum Clique in Conflict Graph
    # 1. Generate a pool of candidate rows (Sidon sets of size k+1 starting at 0).
    # 2. Two rows are compatible if they share no differences.
    # 3. Find a clique of size n_rows in this graph.
    # 4. Minimize the maximum element among the clique.
    
    # Since we don't know the optimal scope, we can search for rows with small scope.
    # However, generating all possible rows with scope <= M is expensive if M is large.
    # Instead, we generate random valid rows and build the graph.
    
    # A "valid row" is a set of k+1 integers {0 = x_0 < x_1 < ... < x_k}
    # such that all differences x_j - x_i (i < j) are distinct.
    # This is a Sidon set (B_2 sequence) property for the set itself.
    
    # We will maintain a list of candidate rows.
    # To find a clique, we use Bron-Kerbosch with pivot and coloring.
    
    # To improve the chance of finding a small scope, we prioritize rows with smaller max element.
    
    # Let's estimate a reasonable max scope to search within.
    # Counting bound is 112. Best known 134.
    # Let's try to find a solution with scope <= 150 first, then refine.
    # Actually, we can just generate rows and see what scope we get.
    
    # Generate candidate rows
    # We need enough candidates to form a clique of size n_rows.
    # If rows are random, probability of two rows being compatible is roughly (1 - 1/M)^{k(k+1)/2}
    # For k=7, num_diffs = 28. If M=100, prob ~ 0.75.
    # We need a clique of size 4.
    
    # Let's generate a pool of candidates.
    # We will generate candidates with increasing scope limits.
    
    def generate_random_sidon_row(k, max_val):
        """Generate a random Sidon set of size k+1 starting at 0 with max <= max_val."""
        # Greedy with randomization
        row = [0]
        # We need k more elements.
        # To ensure distinct differences, we can build it up.
        
        # A simpler way: Pick k random distinct numbers from 1..max_val, sort them,
        # then check if differences are distinct.
        # But this is slow if max_val is large and rejection is high.
        
        # Better: Greedy construction with random jumps.
        current_max = 0
        for _ in range(k):
            # Pick next element
            # It must be > current_max
            # And differences with all previous elements must be new.
            
            # Let's try a few random candidates
            found = False
            for _ in range(50):
                # Bias towards smaller values to keep scope low?
                # Or uniform?
                # Let's try uniform in [current_max + 1, max_val]
                if current_max + 1 > max_val:
                    break
                x = random.randint(current_max + 1, max_val)
                
                # Check differences
                valid = True
                diffs = set()
                for a in row:
                    d = x - a
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                
                # Also need to check if these diffs conflict with existing diffs in row?
                # Wait, the condition is "all differences ... are distinct over all rows".
                # Within a single row, differences must be distinct?
                # "all differences between two entries of the same row are distinct over all rows"
                # This implies:
                # 1. Within a row, diffs must be distinct (otherwise two diffs in same row are not distinct from each other? No, "distinct over all rows" usually means the set of all diffs from all rows has no duplicates).
                # Yes, standard Sidon set definition: all pairwise sums are distinct. Here pairwise differences.
                # So within a row, diffs must be distinct.
                
                # But we also need to ensure that when we combine rows, no diff is repeated.
                # So for generation, we just need the row itself to be a Sidon set.
                
                if valid:
                    # Check if any diff already exists in the row's diff set
                    # We didn't track the full set, just checked against each other.
                    # Let's re-verify:
                    # The check above `if d in diffs` ensures diffs in this step are unique among themselves.
                    # But we need them to be unique against ALL previous diffs in the row.
                    
                    # Let's redo the check properly.
                    existing_diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            existing_diffs.add(row[j] - row[i])
                            
                    new_diffs = [x - a for a in row]
                    
                    # Check for duplicates within new_diffs
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                        
                    # Check for overlap with existing_diffs
                    if any(d in existing_diffs for d in new_diffs):
                        continue
                        
                    # Valid!
                    row.append(x)
                    current_max = x
                    found = True
                    break
            
            if not found:
                return None
                
        return row

    # This generation is a bit slow due to the re-checking.
    # Let's optimize generation.
    
    def generate_fast_sidon_row(k, max_val):
        row = [0]
        current_diffs = set()
        current_max = 0
        
        for _ in range(k):
            # Try to find next x
            # We can iterate x from current_max + 1 upwards, with some randomization?
            # Or just random sampling.
            
            # Random sampling is better for diversity.
            # But we want small scope.
            
            # Let's try a mix: prefer smaller values but allow larger.
            
            success = False
            # Number of attempts
            for _ in range(200):
                # Sample x
                # Bias: exponential distribution?
                # Let's just uniform for now, but constrain upper bound.
                if current_max + 1 > max_val:
                    break
                x = random.randint(current_max + 1, max_val)
                
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in current_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                # Check internal duplicates in new_diffs
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                    
                if valid:
                    row.append(x)
                    current_diffs.update(new_diffs)
                    current_max = x
                    success = True
                    break
            
            if not success:
                return None
                
        return row

    # Pool of candidates
    # We will generate candidates in batches.
    # We want to find a clique of size n_rows.
    
    # To use coloring heuristic effectively, we need a graph.
    # Building the full graph is O(N^2).
    # N can be a few hundred.
    
    # Let's start with a small pool and grow it.
    
    candidates = []
    # Generate initial pool
    # We need to know a good max_val.
    # Let's start with 150.
    
    target_pool_size = 200
    
    # Generate candidates
    # To ensure we have a chance, we generate rows with max_val up to 150.
    
    for _ in range(target_pool_size):
        r = generate_fast_sidon_row(k, 150)
        if r:
            # Check if this row is already in pool (optional, for efficiency)
            # Just add it
            candidates.append(r)
            
    # If we didn't get enough, try higher max_val?
    # For now, assume 150 is enough.
    
    # Build compatibility matrix
    # N = len(candidates)
    # compat[i][j] = True if candidates[i] and candidates[j] share no diffs.
    
    N = len(candidates)
    if N < n_rows:
        # Not enough candidates, return best effort?
        # Or generate more?
        # Let's just return a greedy solution if we fail.
        pass
        
    # Precompute diff sets for each candidate
    diff_sets = []
    for r in candidates:
        ds = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                ds.add(r[j] - r[i])
        diff_sets.append(ds)
        
    # Build adjacency list
    # adj[i] is a set of j such that i and j are compatible.
    adj = [set() for _ in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            if not (diff_sets[i] & diff_sets[j]):
                adj[i].add(j)
                adj[j].add(i)
                
    # Bron-Kerbosch with pivot and coloring
    # We want to find a clique of size n_rows.
    # And minimize the max element.
    
    # Coloring heuristic:
    # Order vertices.
    # Color them greedily.
    # If max color < current best clique size, prune.
    
    # We want to minimize scope.
    # So we should order vertices by max element (ascending).
    # And try to find a clique early.
    
    # However, standard BK finds *a* clique.
    # We can modify it to search for cliques of size n_rows and track the min scope.
    
    # To be efficient, we can stop if we find a clique with scope <= counting_bound (112).
    
    best_scope = float('inf')
    best_clique = None
    
    # Order vertices by max element
    # idx_order = sorted(range(N), key=lambda i: candidates[i][-1])
    
    # We will run BK on the graph.
    # To incorporate scope minimization, we can use a branch and bound approach.
    # But standard BK with coloring is fast enough for N=200.
    
    # Let's implement BK to find ALL maximal cliques of size >= n_rows?
    # No, that's too many.
    # We just need one with min scope.
    
    # Strategy:
    # 1. Sort vertices by max element.
    # 2. Run BK.
    # 3. When a clique is found of size n_rows, calculate its scope.
    # 4. If scope < best_scope, update.
    # 5. Prune if potential clique size < n_rows.
    
    # To prune based on scope?
    # If the smallest max element in the current candidate set (P) is > best_scope,
    # then any clique formed from P will have scope > best_scope?
    # No, because the clique might not include the vertex with the smallest max element.
    # But if ALL vertices in P have max element > best_scope, then yes.
    # So if min(max_element(v) for v in P) > best_scope, prune.
    
    def bron_kerbosch(P, X, C):
        """
        P: set of vertices available
        X: set of vertices already processed (for maximal clique condition)
        C: current clique
        """
        nonlocal best_scope, best_clique
        
        # Pruning: if we can't form a clique of size n_rows
        if len(P) + len(C) < n_rows:
            return
            
        # Pruning: if current best scope is better than any possible clique from P
        # min_max_in_P = min(candidates[v][-1] for v in P)
        # if min_max_in_P > best_scope:
        #     return
        
        if not P and not X:
            # Found a maximal clique
            if len(C) >= n_rows:
                # Calculate scope
                scope = max(candidates[v][-1] for v in C)
                if scope < best_scope:
                    best_scope = scope
                    best_clique = list(C)
                    # Optional: if best_scope <= 112, we can stop?
                    # But we need to be sure.
                    pass
            return
            
        # Coloring heuristic
        # We need to color P.
        # Order P by max element (ascending) to try to find small scope cliques early?
        # Actually, for coloring, we just need an ordering.
        
        # Let's order P by max element ascending.
        p_list = sorted(list(P), key=lambda v: candidates[v][-1])
        
        # Greedy Coloring
        colors = {}
        color_class = []
        # We assign colors 1..k
        # If max color < n_rows - len(C), we can prune.
        
        # Standard coloring:
        # For each vertex in P, assign the smallest possible color.
        
        # To speed up, we can just use a simple greedy coloring.
        
        # We will iterate through p_list.
        # We need to check adjacency with previously colored vertices.
        
        # This is O(|P|^2) per node, which might be slow if P is large.
        # But P shrinks quickly.
        
        # Let's implement a simple coloring.
        
        # We'll store the color of each vertex.
        # And the list of vertices for each color class.
        
        # Actually, we can just compute the chromatic number bound.
        
        # Let's use a simpler pruning:
        # If len(P) < n_rows - len(C), return. (Already done)
        
        # Pivot selection: choose u in P U X with max degree.
        # Here X is usually empty in our recursive calls if we don't maintain it properly?
        # Standard BK maintains X.
        
        # Let's stick to a simpler version: Recursive search without X for now?
        # No, that finds all cliques, not just maximal.
        # We want maximal to ensure we don't miss larger ones?
        # Actually, we just need *a* clique of size n_rows.
        # If we find a clique of size n_rows, we can stop exploring that branch?
        # No, we might find a better one.
        
        # Let's use the standard BK with X.
        
        # For simplicity and speed, let's try a greedy clique construction first?
        # Greedy:
        # 1. Sort vertices by max element.
        # 2. Start with empty clique.
        # 3. Pick vertex with smallest max element that is compatible with all in clique.
        # 4. Repeat until n_rows or stuck.
        
        # This is fast but might get stuck.
        
        # Let's combine:
        # 1. Try greedy.
        # 2. If fails or scope is high, run BK.
        
        pass # Placeholder

    # Let's implement the Greedy approach first, as it's much faster.
    
    def greedy_clique():
        # Sort candidates by max element
        sorted_cands = sorted(range(N), key=lambda i: candidates[i][-1])
        
        clique = []
        # We want to pick n_rows compatible candidates.
        
        # We will iterate through sorted_cands.
        # If a candidate is compatible with all in clique, add it.
        
        for i in sorted_cands:
            if len(clique) == n_rows:
                break
                
            # Check compatibility
            compatible = True
            for j in clique:
                if i not in adj[j]: # i and j not connected
                    compatible = False
                    break
                    
            if compatible:
                clique.append(i)
                
        if len(clique) == n_rows:
            scope = max(candidates[i][-1] for i in clique)
            return scope, clique
        else:
            return float('inf'), None

    scope_g, clique_g = greedy_clique()
    
    if clique_g:
        best_scope = scope_g
        best_clique = clique_g
        
    # If greedy didn't find one, or found a bad one, run BK.
    # But BK is complex to implement correctly with pruning.
    
    # Alternative: Local Search on the Clique?
    # Start with a random valid set of rows (if possible) or a partial one.
    
    # Given time constraints, let's try to improve the greedy result with a simple local search.
    # Or just generate more candidates and retry greedy.
    
    # Let's generate more candidates with slightly higher scope if greedy failed or scope is high.
    
    # If we have a valid solution, we can try to swap rows to reduce scope.
    
    # But for now, let's just return the greedy result if it exists.
    # If not, we need a fallback.
    
    # Fallback: Greedy construction from scratch (like the seed program).
    
    if not best_clique:
        # Fall back to seed program logic (greedy row construction)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
            
        return rows

    # Convert clique indices to rows
    final_rows = [candidates[i] for i in best_clique]
    
    return final_rows
