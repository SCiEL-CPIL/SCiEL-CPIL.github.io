"""
Solves the maximum subset of [1, 128] avoiding equal-LCM triples.
Reformulated as Minimum Hitting Set on the hypergraph of bad triples.
"""
import time
import itertools
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _l_c(b, c) if False else lab == _lcm(b, c)

# Corrected logic for _is_bad_triple
def _is_bad_triple_correct(a, b, c):
    lab = _lcm(a, b)
    lac = _lcm(a, c)
    lbc = _lcm(b, c)
    return lab == lac == lbc

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    n = 128
    elements = list(range(1, n + 1))
    
    # 1. Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple_correct(a, b, c):
            # Store as 0-indexed tuple
            bad_triples.append((a - 1, b - 1, c - 1))
            
    if not bad_triples:
        return elements

    # 2. Precompute node-to-triples mapping for fast filtering
    # triples_by_node[i] = list of indices in bad_triples that contain node i
    triples_by_node = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(bad_triples):
        triples_by_node[a].append(idx)
        triples_by_node[b].append(idx)
        triples_by_node[c].append(idx)
        
    total_triples = len(bad_triples)
    
    # 3. Minimum Hitting Set Branch and Bound
    # State: active_triples (list of indices of bad_triples that are not yet hit)
    # We want to minimize the size of U (the hitting set).
    
    best_U_size = n + 1
    best_U_mask = 0
    search_count = 0
    time_up = False

    def lower_bound_matching(active_triples):
        """
        Greedy maximal matching on the set of active triples.
        Returns the size of the matching, which is a lower bound on
        the number of additional nodes needed to hit all active triples.
        """
        if not active_triples:
            return 0
        
        # Use a set for fast removal
        remaining = set(active_triples)
        match_count = 0
        
        while remaining:
            # Pick an arbitrary triple (first in set iteration)
            # To be deterministic, we could sort, but for LB speed, arbitrary is fine
            # However, picking a "dense" one might be better, but for LB, any maximal matching works.
            t_idx = next(iter(remaining))
            a, b, c = bad_triples[t_idx]
            
            match_count += 1
            
            # Remove all triples that intersect {a, b, c}
            # A triple intersects if it shares any node with {a, b, c}
            to_remove = set()
            # Check all remaining triples
            # Optimization: We can iterate through nodes a,b,c and find their triples
            # But we need to ensure we only remove from 'remaining'
            
            # Efficient removal:
            # For each node in the chosen triple, find all triples containing it
            # and add them to to_remove if they are in remaining.
            
            # Since n is small, we can just filter the remaining set
            new_remaining = set()
            for rt_idx in remaining:
                ra, rb, rc = bad_triples[rt_idx]
                # Check intersection with (a,b,c)
                if ra == a or ra == b or ra == c or \
                   rb == a or rb == b or rb == c or \
                   rc == a or rc == b or rc == c:
                    continue
                new_remaining.add(rt_idx)
            
            remaining = new_remaining
            
        return match_count

    def search(active_triples, U_mask, U_size, depth):
        nonlocal best_U_size, best_U_mask, search_count, time_up
        
        if time.time() > deadline:
            time_up = True
            return
            
        search_count += 1
        
        if not active_triples:
            if U_size < best_U_size:
                best_U_size = U_size
                best_U_mask = U_mask
            return
            
        # Pruning 1: Current size + Lower Bound
        lb = lower_bound_matching(active_triples)
        if U_size + lb >= best_U_size:
            return
            
        # Branching: Pick a triple to hit
        # Heuristic: Pick the triple that appears in the fewest other active triples?
        # Or just pick the first one.
        # Let's pick the first active triple for simplicity, as LB pruning is strong.
        t_idx = active_triples[0]
        a, b, c = bad_triples[t_idx]
        
        # Branch 1: Add 'a' to U
        # New active triples: those in active_triples that do NOT contain 'a'
        # We can precompute a bitmask of triples containing 'a'?
        # Or just filter the list.
        
        # To speed up filtering, we can maintain a list of active triple indices.
        # Filtering a list of size M takes O(M).
        
        # Branch 1
        new_active_1 = []
        for idx in active_triples:
            ta, tb, tc = bad_triples[idx]
            if ta != a and tb != a and tc != a:
                new_active_1.append(idx)
        
        if not time_up:
            search(new_active_1, U_mask | (1 << a), U_size + 1, depth + 1)
            
        # Branch 2: Add 'b' to U
        if not time_up:
            new_active_2 = []
            for idx in active_triples:
                ta, tb, tc = bad_triples[idx]
                if ta != b and tb != b and tc != b:
                    new_active_2.append(idx)
            search(new_active_2, U_mask | (1 << b), U_size + 1, depth + 1)
            
        # Branch 3: Add 'c' to U
        if not time_up:
            new_active_3 = []
            for idx in active_triples:
                ta, tb, tc = bad_triples[idx]
                if ta != c and tb != c and tc != c:
                    new_active_3.append(idx)
            search(new_active_3, U_mask | (1 << c), U_size + 1, depth + 1)

    # Initial active triples: all of them
    initial_active = list(range(total_triples))
    
    # Initial U is empty
    search(initial_active, 0, 0, 0)
    
    # If no solution found (should not happen, as U=all nodes is always a solution), 
    # or if we hit time limit, use the best found.
    # If best_U_size is still n+1, it means we didn't finish or something went wrong.
    # But U = all nodes is size 128, so best_U_size <= 128.
    
    # Construct result S
    # S = [1..128] \ U
    U_nodes = set()
    for i in range(n):
        if (best_U_mask >> i) & 1:
            U_nodes.add(i + 1)
            
    result = [x for x in elements if x not in U_nodes]
    
    # Fallback: If result is empty (should not happen) or if we failed to find a valid set
    # Verify the result
    def is_valid(S):
        S_set = set(S)
        for a, b, c in itertools.combinations(S, 3):
            if _is_bad_triple_correct(a, b, c):
                return False
        return True
        
    if not result or not is_valid(result):
        # Fallback to a greedy solution if B&B failed or returned invalid
        # Greedy: Start with all, remove nodes from most frequent bad triples
        S_set = set(elements)
        # Count frequency of each node in bad triples
        freq = [0] * n
        for a, b, c in bad_triples:
            freq[a] += 1
            freq[b] += 1
            freq[c] += 1
            
        # Sort nodes by frequency descending
        order = sorted(range(n), key=lambda x: freq[x], reverse=True)
        
        current_S = set(elements)
        # We need to remove nodes until no bad triple is fully in current_S
        # This is a greedy hitting set
        active_triples = list(range(total_triples))
        for node_idx in order:
            if time.time() > deadline - 5:
                break
            node_val = node_idx + 1
            if node_val not in current_S:
                continue
            # Check if removing this node helps
            # If node is in any active triple, removing it hits those triples
            # We should remove it if it hits at least one active triple
            hits = 0
            for t_idx in active_triples:
                a, b, c = bad_triples[t_idx]
                if a == node_idx or b == node_idx or c == node_idx:
                    hits += 1
            if hits > 0:
                current_S.remove(node_val)
                # Update active triples
                new_active = []
                for t_idx in active_triples:
                    a, b, c = bad_triples[t_idx]
                    if a != node_idx and b != node_idx and c != node_idx:
                        new_active.append(t_idx)
                active_triples = new_active
                if not active_triples:
                    break
                    
        result = list(current_S)
        if not result:
            result = [1]
            
    return result

# EVOLVE-BLOCK-END
