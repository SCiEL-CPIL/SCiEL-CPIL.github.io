import time
import random
import math
import sys
from collections import defaultdict
from itertools import combinations

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a conflict-graph based heuristic search to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # If the problem size is not the specific hard case, just return greedy
    if n_rows != 7 or k != 5:
        return best_solution

    # Specific solver for (7, 5)
    # Theoretical lower bound is 105.
    # We iterate M from 105 upwards.
    
    # Precompute valid rows for a given M
    # A row is a tuple of length k+1, starting with 0, increasing strictly.
    # We represent a row by its set of differences (frozenset) and its max element.
    
    def get_valid_rows(M):
        """
        Generates all valid rows of length k+1 starting with 0 and max <= M.
        Returns a list of (row_tuple, diffs_frozenset).
        """
        valid_rows = []
        # Use recursion to generate rows
        def gen(idx, current_row, current_diffs):
            if time.time() > deadline - 1:
                return
            if idx == k + 1:
                # Check if all diffs are distinct within the row
                # Since we build it up, current_diffs is a set
                # If len(current_diffs) == k*(k+1)/2, it's valid
                if len(current_diffs) == k*(k+1)//2:
                    valid_rows.append((tuple(current_row), frozenset(current_diffs)))
                return
            
            start_val = current_row[-1] + 1
            # Optimization: we can bound the upper limit of x
            # The remaining elements must be distinct and fit in [x, M]
            # Simple bound: x <= M - (k + 1 - idx)
            end_val = M - (k + 1 - idx - 1) # need at least k+1-idx-1 more steps
            if start_val > end_val:
                return
            
            for x in range(start_val, end_val + 1):
                new_diffs = set()
                valid = True
                for i in range(idx):
                    d = x - current_row[i]
                    if d in current_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                gen(idx + 1, current_row, current_diffs | new_diffs)
                current_row.pop()
        
        gen(1, [0], set())
        return valid_rows

    # Conflict graph construction
    # Nodes are valid rows.
    # Edge between two nodes if their difference sets intersect.
    # We want to find a clique of size n_rows.
    # Since we want to minimize the max entry, we iterate M.
    
    def solve_dts_specific(M):
        """
        Finds a DTS(7,5) with max entry <= M using conflict graph clique finding.
        Returns list of rows or None.
        """
        # 1. Generate valid rows
        rows_data = get_valid_rows(M)
        if not rows_data:
            return None
        
        n = len(rows_data)
        
        # 2. Build conflict graph (adjacency list of non-conflicting rows)
        # Actually, it's easier to build the graph of CONFLICTS and look for an Independent Set,
        # or build the graph of COMPATIBILITY and look for a Clique.
        # Let's build Compatibility Graph: Edge (i, j) if diffs_i & diffs_j == empty.
        
        # To save memory and time, we can use bitsets for diffs.
        # Max diff is M. So diffs fit in an integer of M+1 bits.
        # M is around 105-112, so ~112 bits. Python ints handle this fine.
        
        row_diffs_masks = []
        row_tuples = []
        for row, diffs in rows_data:
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            row_diffs_masks.append(mask)
            row_tuples.append(row)
        
        # Sort rows by their max element (ascending) to prioritize smaller scopes
        # This might help in finding a solution with smaller max earlier in the search
        # But we already fixed M. However, within M, smaller max is better for "tightness".
        # Let's just keep them as is or sort by max.
        
        # Build adjacency list for compatibility
        # adj[i] is a set of j such that row i and row j are compatible.
        # For n=7, we need a clique of size 7.
        
        # To speed up, we can use a greedy clique finding algorithm or backtracking.
        # Given the size of the graph could be large (thousands of rows?),
        # let's estimate the number of rows.
        # For M=105, number of Sidon sequences of length 6 is not huge.
        
        # Let's use a simple backtracking clique finder with pruning.
        # We want to find a clique of size n_rows.
        
        # Optimization: 
        # 1. Sort vertices by degree (descending) or just index.
        # 2. Use recursive search.
        
        # Precompute compatibility bitsets?
        # If n is small enough (< 10000), we can store adj as list of sets.
        
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            # Pruning: if i has too many conflicts, it might be less useful?
            # No, we just build the graph.
            for j in range(i + 1, n):
                if time.time() > deadline - 1:
                    return None
                if (row_diffs_masks[i] & row_diffs_masks[j]) == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Now find a clique of size n_rows in this graph.
        # We can use a simple backtracking.
        # To improve performance, we can order vertices.
        # Vertices with smaller max element are "better" candidates?
        # Let's sort indices by the max element of the row.
        
        max_elements = [max(row) for row in row_tuples]
        sorted_indices = sorted(range(n), key=lambda i: max_elements[i])
        
        # Remap indices? Or just use the sorted order in the search.
        # Let's create a new adjacency structure based on sorted indices?
        # Or just iterate through sorted_indices in the search.
        
        # Backtracking search for clique
        def find_clique(candidates, current_clique):
            if time.time() > deadline - 1:
                return None
            
            if len(current_clique) == n_rows:
                return current_clique[:]
            
            # Pruning: if we don't have enough candidates left, fail
            if len(candidates) < n_rows - len(current_clique):
                return None
            
            # Try each candidate
            # To find a solution faster, we can iterate candidates in order of "potential"
            # Candidates are already sorted by max element in the initial list?
            # Wait, `candidates` is a list of indices. We should process them in a smart order.
            
            for idx in range(len(candidates)):
                v = candidates[idx]
                
                # Check if v is compatible with all in current_clique
                # Since we built adj, we just need to check if v is in adj[u] for all u in current_clique
                # Actually, if we maintain candidates as "compatible with all current_clique", 
                # then any v in candidates is compatible.
                
                # New candidates: intersection of current candidates (excluding v and earlier) 
                # and adj[v].
                # Also, they must be compatible with v.
                
                new_candidates = []
                # Iterate remaining candidates
                for jdx in range(idx + 1, len(candidates)):
                    u = candidates[jdx]
                    if u in adj[v]:
                        new_candidates.append(u)
                
                # Recurse
                res = find_clique(new_candidates, current_clique + [v])
                if res is not None:
                    return res
            
            return None
        
        # Initial candidates: all vertices
        # But we should start with the "best" vertices (smallest max)
        # The sorted_indices list gives us vertices in order of increasing max.
        # So we pass sorted_indices as the initial candidates.
        
        # However, the adjacency check `u in adj[v]` is O(1) for sets.
        # The bottleneck is the number of vertices.
        
        # Let's try to find the clique.
        # To avoid deep recursion issues, we can limit the search or use iterative deepening?
        # No, we just need ONE clique of size 7.
        
        # One optimization: if the first few vertices (smallest max) don't form a clique,
        # we might need to go deeper.
        
        # Let's run the search.
        # To prevent infinite loops or excessive time, we rely on the time check inside.
        
        clique = find_clique(sorted_indices, [])
        
        if clique:
            return [list(row_tuples[i]) for i in clique]
        
        return None

    # Iterate M from 105
    current_time = start_time
    # We will try M from 105 to 115 (or higher if time permits)
    # The known best is 112.
    
    for M in range(105, 120):
        if time.time() > deadline - 5:
            break
            
        sol = solve_dts_specific(M)
        
        if sol:
            # Verify solution
            all_diffs = set()
            valid = True
            for row in sol:
                if len(row) != k + 1:
                    valid = False
                    break
                if row[0] != 0:
                    valid = False
                    break
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
                if max(row) > M:
                    valid = False
                    break
            if valid:
                return sol
            else:
                pass
                
    return best_solution

# EVOLVE-BLOCK-END
