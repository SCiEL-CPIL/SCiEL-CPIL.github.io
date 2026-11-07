import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0

def _build_edges():
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                edge = frozenset([a, b, c, s1, s2, s3])
                edges.append(edge)
    return edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges = _build_edges()
    num_edges = len(edges)
    
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge sizes for quick access
    edge_sizes = [len(e) for e in edges]

    def _can_add(v, edge_counts):
        for ei in vert_to_edges[v]:
            if edge_counts[ei] == edge_sizes[ei] - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    best_set = set()
    best_size = 0
    
    all_vertices = list(range(1, N + 1))
    
    def _greedy_add_all(cur_set, cur_edge_counts):
        """Add all possible missing vertices greedily. Returns count added."""
        added = 0
        missing = [v for v in all_vertices if v not in cur_set]
        random.shuffle(missing)
        for v in missing:
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                added += 1
        return added

    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        cur_set = set(range(65, N + 1))
        
        # Perturb: remove random elements
        num_remove = random.randint(0, 15)
        for _ in range(num_remove):
            if cur_set:
                v = random.choice(list(cur_set))
                cur_set.remove(v)
        
        cur_edge_counts = [0] * num_edges
        for v in cur_set:
            _update_counts(v, 1, cur_edge_counts)
            
        cur_size = len(cur_set)
        
        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            
        # Local Search with multi-swap
        stuck_count = 0
        max_stuck = 30
        while time.time() < deadline:
            improved = False
            
            # 1. Try to add all possible missing vertices
            added = _greedy_add_all(cur_set, cur_edge_counts)
            if added > 0:
                cur_size += added
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set.copy()
                stuck_count = 0
                continue
                
            # 2. No adds possible. Try multi-pair swaps.
            # Try removing 1-3 vertices and adding back as many as possible
            found_swap = False
            if cur_set:
                cur_list = list(cur_set)
                random.shuffle(cur_list)
                
                # Try different numbers of removals
                for num_rem in [1, 2, 3]:
                    if num_rem > len(cur_list):
                        continue
                    
                    # Pick random subset of vertices to remove
                    rem_vertices = cur_list[:num_rem]
                    
                    # Save state
                    old_set = cur_set.copy()
                    old_edge_counts = cur_edge_counts.copy()
                    old_size = cur_size
                    
                    # Remove vertices
                    for v in rem_vertices:
                        cur_set.remove(v)
                        _update_counts(v, -1, cur_edge_counts)
                    
                    # Try to add all possible missing vertices
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
                        # Keep the swap
                        cur_size = new_size
                        if cur_size > best_size:
                            best_size = cur_size
                            best_set = cur_set.copy()
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Rollback
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

    # Final Polish: Exhaustive greedy add on best_set
    if time.time() < deadline:
        be = [0] * num_edges
        for v in best_set:
            _update_counts(v, 1, be)
        
        # Multiple passes with different orderings
        for _ in range(5):
            missing_final = [v for v in all_vertices if v not in best_set]
            random.shuffle(missing_final)
            for v in missing_final:
                if _can_add(v, be):
                    best_set.add(v)
                    _update_counts(v, 1, be)
        
        # Try removing each vertex and re-adding to see if we can do better
        improved_final = True
        while improved_final and time.time() < deadline:
            improved_final = False
            be = [0] * num_edges
            for v in best_set:
                _update_counts(v, 1, be)
            
            for rem_v in list(best_set):
                # Save
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
                
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
