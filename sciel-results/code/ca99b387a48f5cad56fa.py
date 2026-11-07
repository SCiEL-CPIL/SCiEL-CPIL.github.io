import time
import random
import sys
import math

N = 128
DEADLINE_MARGIN = 5.0

def _build_edges():
    edges = []
    edge_vertices = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                # The six values are a, b, c, s1, s2, s3
                # Note: they need not be distinct, but the forbidden condition is
                # that all three of a,b,c are in S AND all three of s1,s2,s3 are in S.
                # Actually, the constraint is: forbid a,b,c in S such that a+b, a+c, b+c are also in S.
                # This is equivalent to: for every triple (a,b,c) with a<b<c,
                # NOT (a in S AND b in S AND c in S AND (a+b) in S AND (a+c) in S AND (b+c) in S).
                
                # We model this as a hyperedge. If all vertices in the hyperedge are selected, it's a violation.
                # The hyperedge contains {a, b, c, a+b, a+c, b+c}.
                # We need to be careful if some of these are the same.
                # e.g. if a+b == a+c, then b==c which is impossible since b<c.
                # if a+b == b+c, then a==c which is impossible.
                # if a+c == b+c, then a==b which is impossible.
                # Can a == a+b? No, b>0.
                # Can a == b+c? No, c>0, b>0.
                # Can a == a+c? No, c>0.
                # Can a+b == a? No.
                # Can a+b == c? Possible. e.g. 1+2=3.
                # If a+b == c, then the set is {a, b, c, c, a+c, b+c} = {a, b, c, a+c, b+c}.
                # The condition is a,b,c in S AND c in S AND a+c in S AND b+c in S.
                # So the hyperedge is effectively {a, b, c, a+c, b+c}.
                
                # To be safe, we just use a frozenset.
                # If the frozenset has size < 6, it means there were duplicates.
                # The violation occurs if ALL elements in the frozenset are in S.
                
                edge = frozenset([a, b, c, s1, s2, s3])
                edges.append(edge)
                edge_vertices.append(list(edge))
    return edges, edge_vertices

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges, edge_vertices = _build_edges()
    num_edges = len(edges)
    
    # Map vertex to list of edge indices containing it
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge sizes
    edge_sizes = [len(e) for e in edges]

    # Precompute for each edge, the list of vertices as a tuple for faster iteration?
    # edge_vertices is already a list of lists.
    
    def _can_add(v, edge_counts):
        """Check if vertex v can be added without creating a violation."""
        for ei in vert_to_edges[v]:
            # If adding v makes the count equal to the size, it becomes a violation
            if edge_counts[ei] == edge_sizes[ei] - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        """Update edge counts for adding/removing vertex v."""
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    def _compute_violations(edge_counts):
        """Count the number of violated edges (fully selected hyperedges)."""
        violations = 0
        for i in range(num_edges):
            if edge_counts[i] == edge_sizes[i]:
                violations += 1
        return violations

    def _greedy_add_all(cur_set, cur_edge_counts, order=None):
        """Add all possible missing vertices greedily. Returns count added."""
        added = 0
        if order is None:
            missing = [v for v in range(1, N + 1) if v not in cur_set]
            random.shuffle(missing)
        else:
            missing = [v for v in order if v not in cur_set]
        
        for v in missing:
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                added += 1
        return added

    def _local_search(cur_set, cur_edge_counts, cur_size):
        """Perform local search with multi-swap. Returns updated set, counts, size."""
        stuck_count = 0
        max_stuck = 20
        all_vertices = list(range(1, N + 1))
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add all possible missing vertices
            added = _greedy_add_all(cur_set, cur_edge_counts)
            if added > 0:
                cur_size += added
                stuck_count = 0
                continue
                
            # 2. No adds possible. Try multi-pair swaps.
            found_swap = False
            if cur_set:
                cur_list = list(cur_set)
                random.shuffle(cur_list)
                
                for num_rem in [1, 2, 3]:
                    if num_rem > len(cur_list):
                        continue
                    
                    rem_vertices = cur_list[:num_rem]
                    
                    old_set = cur_set.copy()
                    old_edge_counts = cur_edge_counts.copy()
                    old_size = cur_size
                    
                    for v in rem_vertices:
                        cur_set.remove(v)
                        _update_counts(v, -1, cur_edge_counts)
                    
                    missing_after = [v for v in all_vertices if v not in cur_set]
                    random.shuffle(missing_after)
                    
                    local_added = 0
                    for v in missing_after:
                        if _can_add(v, cur_edge_counts):
                            cur_set.add(v)
                            _update_counts(v, 1, cur_edge_counts)
                            local_added += 1
                    
                    new_size = cur_size - num_rem + local_added
                    
                    if new_size > old_size:
                        cur_size = new_size
                        improved = True
                        found_swap = True
                        break
                    else:
                        cur_set = old_set
                        cur_edge_counts = old_edge_counts
                        cur_size = old_size
                    
                    if found_swap:
                        break
            
            if improved:
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
        
        return cur_set, cur_edge_counts, cur_size

    def _polish(best_set):
        """Final exhaustive polish on best_set."""
        if time.time() >= deadline:
            return best_set
        
        be = [0] * num_edges
        for v in best_set:
            _update_counts(v, 1, be)
        
        all_vertices = list(range(1, N + 1))
        
        # Try adding in different orders
        for _ in range(10):
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        # Try removing one and re-adding
        improved_final = True
        while improved_final and time.time() < deadline:
            improved_final = False
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            
            for rem_v in list(best_set):
                old_set = best_set.copy()
                old_be = be.copy()
                
                best_set.remove(rem_v)
                _update_counts(rem_v, -1, be)
                
                missing_after = [v for v in all_vertices if v not in best_set]
                random.shuffle(missing_after)
                
                added = 0
                for v in missing_after:
                    if _can_add(v, be):
                        best_set.add(v)
                        _update_counts(v, 1, be)
                        added += 1
                
                if len(best_set) > len(old_set):
                    improved_final = True
                    break
                else:
                    best_set = old_set
                    be = old_be
        
        # Final add pass
        if time.time() < deadline:
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        return best_set

    best_set = set()
    best_size = 0
    all_vertices = list(range(1, N + 1))
    
    # Strategy:
    # 1. Greedy construction with different biases
    # 2. Local search
    # 3. Simulated Annealing / Random Restart Local Search
    
    # Phase 1: Greedy with different orderings
    # Try ascending, descending, random
    for _ in range(5):
        cur_set = set()
        cur_counts = [0] * num_edges
        order = all_vertices[:]
        random.shuffle(order)
        _greedy_add_all(cur_set, cur_counts, order)
        if len(cur_set) > best_size:
            best_size = len(cur_set)
            best_set = cur_set.copy()
            
        # Local search on the greedy result
        if time.time() < deadline:
            ls_set, ls_counts, ls_size = _local_search(cur_set, cur_counts, len(cur_set))
            if ls_size > best_size:
                best_size = ls_size
                best_set = ls_set.copy()

    # Phase 2: Random Restart Local Search
    # Start with random sets and improve
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random subset
        # Bias towards larger numbers? Or random?
        # Let's try a mix.
        
        # Option 1: Random subset with p=0.5
        if r % 3 == 0:
            cur_set = set(v for v in all_vertices if random.random() < 0.5)
        # Option 2: Large numbers
        elif r % 3 == 1:
            cur_set = set(range(65, N + 1))
        # Option 3: Small numbers
        else:
            cur_set = set(range(1, 65))
        
        # Ensure validity by removing violators?
        # Or just let local search fix it?
        # Local search only adds if valid. It doesn't remove invalid ones unless it swaps.
        # If the initial set is invalid, _can_add might fail for many things.
        # We should probably start with a valid set or clean it up.
        
        # Let's clean up: remove vertices that cause violations?
        # Actually, if the set is invalid, it's not a valid solution.
        # The local search assumes the current set is valid?
        # _can_add checks if adding v creates a violation.
        # If the current set is already invalid, edge_counts might be == edge_sizes for some edges.
        # Then _can_add will return False for any v in those edges.
        # This might get stuck.
        
        # Better: Start with a valid set.
        # Greedy construction ensures validity.
        # So let's use greedy construction with random order as the base for restarts.
        
        cur_set = set()
        cur_counts = [0] * num_edges
        order = all_vertices[:]
        random.shuffle(order)
        _greedy_add_all(cur_set, cur_counts, order)
        
        # Perturb: remove a few random elements
        if cur_set:
            num_remove = random.randint(0, 5)
            to_remove = random.sample(list(cur_set), min(num_remove, len(cur_set)))
            for v in to_remove:
                cur_set.remove(v)
                _update_counts(v, -1, cur_counts)
        
        cur_size = len(cur_set)
        
        # Local search
        ls_set, ls_counts, ls_size = _local_search(cur_set, cur_counts, cur_size)
        
        if ls_size > best_size:
            best_size = ls_size
            best_set = ls_set.copy()

    # Phase 3: Simulated Annealing style search
    # Objective: Maximize size, Penalty for violations.
    # Score = size - lambda * violations
    # We want to find sets with 0 violations and high size.
    
    def _sa_search():
        nonlocal best_set, best_size
        
        # Start from best found or random
        if best_set:
            cur_set = best_set.copy()
        else:
            cur_set = set(v for v in all_vertices if random.random() < 0.5)
        
        cur_counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, cur_counts)
        
        cur_violations = _compute_violations(cur_counts)
        cur_size = len(cur_set)
        
        # Temperature
        T = 5.0
        T_min = 0.1
        decay = 0.99
        
        sa_best_set = cur_set.copy()
        sa_best_size = cur_size
        sa_best_violations = cur_violations
        
        while T > T_min and time.time() < deadline:
            # Pick a random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                missing = [v for v in all_vertices if v not in cur_set]
                if not missing:
                    continue
                v = random.choice(missing)
                
                # Calculate delta
                old_violations = cur_violations
                new_violations = 0
                # Check edges affected by v
                # If we add v, any edge containing v might become violated if it was at size-1.
                # Also, edges not containing v are unaffected.
                
                # Wait, we need to be careful.
                # Let's just simulate and count? Too slow.
                # Let's use the edge_counts logic.
                
                # Tentative add
                delta_violations = 0
                for ei in vert_to_edges[v]:
                    if cur_counts[ei] == edge_sizes[ei] - 1:
                        # This edge will become violated
                        delta_violations += 1
                    elif cur_counts[ei] == edge_sizes[ei]:
                        # This edge is already violated, adding v doesn't change violation status (still violated)
                        # But wait, if it's already violated, it's still violated.
                        # So no change in violation count for this edge.
                        pass
                
                new_size = cur_size + 1
                new_violations = cur_violations + delta_violations
                
                # Score change
                # We want to maximize size - lambda * violations
                # lambda should be large enough to prevent violations in the final solution,
                # but allow them during search.
                lambda_pen = 10.0
                delta_score = 1 - lambda_pen * delta_violations
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    # Accept
                    cur_set.add(v)
                    _update_counts(v, 1, cur_counts)
                    cur_size = new_size
                    cur_violations = new_violations
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_set = cur_set.copy()
                        sa_best_violations = 0
            
            elif move_type == 'remove':
                if not cur_set:
                    continue
                v = random.choice(list(cur_set))
                
                delta_violations = 0
                for ei in vert_to_edges[v]:
                    if cur_counts[ei] == edge_sizes[ei]:
                        # This edge was violated, removing v fixes it
                        delta_violations -= 1
                    # If it was not violated, removing v makes it less full, so still not violated.
                
                new_size = cur_size - 1
                new_violations = cur_violations + delta_violations
                
                lambda_pen = 10.0
                delta_score = -1 - lambda_pen * delta_violations
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    # Accept
                    cur_set.remove(v)
                    _update_counts(v, -1, cur_counts)
                    cur_size = new_size
                    cur_violations = new_violations
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_set = cur_set.copy()
                        sa_best_violations = 0
            
            else: # swap
                if len(cur_set) < 2:
                    continue
                missing = [v for v in all_vertices if v not in cur_set]
                if not missing:
                    continue
                
                v_out = random.choice(list(cur_set))
                v_in = random.choice(missing)
                
                # Remove v_out
                delta_viol_out = 0
                for ei in vert_to_edges[v_out]:
                    if cur_counts[ei] == edge_sizes[ei]:
                        delta_viol_out -= 1
                
                # Add v_in (on the modified counts)
                # We need to simulate the counts after removal
                # This is tricky to do in O(1) without actually updating.
                # Let's just do it and revert if rejected.
                
                old_set = cur_set.copy()
                old_counts = cur_counts.copy()
                old_size = cur_size
                old_violations = cur_violations
                
                cur_set.remove(v_out)
                _update_counts(v_out, -1, cur_counts)
                cur_set.add(v_in)
                _update_counts(v_in, 1, cur_counts)
                
                new_violations = _compute_violations(cur_counts)
                new_size = cur_size # Size unchanged
                
                lambda_pen = 10.0
                delta_score = 0 - lambda_pen * (new_violations - old_violations)
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    # Accept
                    cur_violations = new_violations
                    # cur_size is unchanged
                    # cur_set and cur_counts are already updated
                    
                    if cur_violations == 0 and cur_size > sa_best_size:
                        sa_best_size = cur_size
                        sa_best_set = cur_set.copy()
                        sa_best_violations = 0
                else:
                    # Revert
                    cur_set = old_set
                    cur_counts = old_counts
                    cur_size = old_size
                    cur_violations = old_violations
            
            T *= decay
        
        # After SA, try to polish the best found in SA
        if sa_best_violations == 0 and sa_best_size > 0:
            # Rebuild counts for sa_best_set
            sa_counts = [0] * num_edges
            for v in sa_best_set:
                _update_counts(v, 1, sa_counts)
            
            # Local search
            ls_set, ls_counts, ls_size = _local_search(sa_best_set, sa_counts, sa_best_size)
            if ls_size > best_size:
                best_size = ls_size
                best_set = ls_set.copy()
        
        return sa_best_set, sa_best_size

    # Run SA a few times
    for _ in range(3):
        if time.time() > deadline:
            break
        _sa_search()

    # Final Polish
    if best_set:
        best_set = _polish(best_set)
    
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
