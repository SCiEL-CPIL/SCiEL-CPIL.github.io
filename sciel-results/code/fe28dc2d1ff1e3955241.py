import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_up_to_length(edges, max_len=6):
        """Find all simple cycles up to max_len in the graph defined by edges."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = set()
        n = max(adj.keys(), default=0) + 1
        vertices = list(range(n))
        
        # To avoid duplicates, we only start DFS from the smallest vertex in the cycle
        # and ensure the path is lexicographically minimal or something?
        # Simpler: Just find all, then deduplicate by frozenset of edges.
        
        for start in vertices:
            # DFS
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_len:
                    continue
                
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        cycle_edges = frozenset((min(path[i], path[(i+1)%len(path)]), max(path[i], path[(i+1)%len(path)])) for i in range(len(path)))
                        cycles.add(cycle_edges)
                    elif neighbor not in path and neighbor > start:
                        # Only go to neighbors greater than start to reduce duplicates?
                        # No, this misses cycles where start is not the min.
                        # Better to just find all and dedup.
                        # Optimization: only extend if neighbor > start? 
                        # This works if we consider the cycle starting at min vertex.
                        # If we enforce min(path) == start, we must only visit neighbors > start?
                        # No, if start is min, all others are > start.
                        # So we should only start from a vertex if it is the minimum in the cycle.
                        # We can check this at the end.
                        if neighbor > start:
                            stack.append((neighbor, path + [neighbor]))
        
        # Convert to list of edge tuples
        return [list(c) for c in cycles]

    def greedy_local_search(edges, candidate_cycles, time_limit=10.0):
        """
        Greedy selection with local search.
        """
        # Sort candidates by value (length - 1) descending, then by length
        # Actually, we want to maximize sum(value).
        # Greedy: Pick highest value, then next that doesn't conflict, etc.
        
        edge_to_cycles = defaultdict(list)
        for i, c in enumerate(candidate_cycles):
            for e in c:
                edge_to_cycles[e].append(i)
        
        used_edges = set()
        selected = []
        
        # Sort by value descending
        indices = sorted(range(len(candidate_cycles)), key=lambda i: len(candidate_cycles[i]), reverse=True)
        
        for i in indices:
            c = candidate_cycles[i]
            if all(e not in used_edges for e in c):
                selected.append(i)
                for e in c:
                    used_edges.add(e)
        
        # Local Search
        best_score = sum(len(candidate_cycles[i]) - 1 for i in selected)
        current_selected = set(selected)
        
        start_ls = time.time()
        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            # Try to add a cycle
            for i in range(len(candidate_cycles)):
                if i in current_selected:
                    continue
                c = candidate_cycles[i]
                if all(e not in used_edges for e in c):
                    # Add it
                    new_score = best_score + len(c) - 1
                    if new_score > best_score:
                        best_score = new_score
                        current_selected.add(i)
                        for e in c:
                            used_edges.add(e)
                        improved = True
                        break # Restart scan after improvement
            
            # Try to swap: Remove one, add another
            if not improved:
                for i_remove in list(current_selected):
                    c_remove = candidate_cycles[i_remove]
                    # Temporarily remove
                    temp_used = set(used_edges)
                    for e in c_remove:
                        temp_used.discard(e)
                    
                    best_swap_gain = 0
                    best_swap_i = -1
                    for i_add in range(len(candidate_cycles)):
                        if i_add in current_selected:
                            continue
                        c_add = candidate_cycles[i_add]
                        # Check if c_add fits in temp_used
                        if all(e not in temp_used for e in c_add):
                            gain = (len(c_add) - 1) - (len(c_remove) - 1)
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_i = i_add
                    
                    if best_swap_i != -1 and best_swap_gain > 0:
                        # Execute swap
                        current_selected.discard(i_remove)
                        current_selected.add(best_swap_i)
                        used_edges = temp_used
                        for e in candidate_cycles[best_swap_i]:
                            used_edges.add(e)
                        best_score += best_swap_gain
                        improved = True
                        break # Restart scan

        # Construct result
        pieces = []
        for i in current_selected:
            c = candidate_cycles[i]
            # Reconstruct vertex order from edges
            # This is a bit tricky, we just need a valid cycle representation
            # We can store the vertex list in candidate_cycles generation
            # Let's modify candidate_cycles to store (vertex_list, edge_set)
            
        # The above logic is slightly flawed as I didn't store vertex lists properly.
        # Let's re-implement with proper data structures.
        return None 

    def decompose(edges):
        if not edges:
            return []
        
        # 1. Generate candidate cycles
        # We need vertex lists to output the pieces.
        # Let's generate cycles as lists of vertices.
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n = max(adj.keys(), default=0) + 1
        vertices = list(range(n))
        
        candidates = [] # List of (vertex_list, edge_set)
        seen_edges = set()
        
        # Find small cycles
        for max_len in [3, 4, 5, 6]:
            for start in vertices:
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    if len(path) > max_len:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == start and len(path) >= 3:
                            # Check if this cycle is new
                            edge_set = frozenset((min(path[i], path[(i+1)%len(path)]), max(path[i], path[(i+1)%len(path)])) for i in range(len(path)))
                            if edge_set not in seen_edges:
                                seen_edges.add(edge_set)
                                candidates.append((list(path), edge_set))
                        elif neighbor not in path and neighbor > start:
                            stack.append((neighbor, path + [neighbor]))
        
        # Add some random longer cycles
        rng = random.Random(42)
        for _ in range(100):
            if not adj: break
            start = rng.choice(vertices)
            path = [start]
            current = start
            for _ in range(10):
                neighbors = list(adj[current])
                if not neighbors: break
                # Prefer unvisited
                unvisited = [n for n in neighbors if n not in path]
                if unvisited:
                    next_node = rng.choice(unvisited)
                else:
                    if start in neighbors and len(path) >= 3:
                        next_node = start
                    else:
                        break
                path.append(next_node)
                current = next_node
                if current == start and len(path) > 1:
                    break
            if len(path) >= 3 and path[0] == path[-1]:
                path = path[:-1] # Remove duplicate start
                edge_set = frozenset((min(path[i], path[(i+1)%len(path)]), max(path[i], path[(i+1)%len(path)])) for i in range(len(path)))
                if edge_set not in seen_edges:
                    seen_edges.add(edge_set)
                    candidates.append((list(path), edge_set))

        if not candidates:
            return [[u, v] for u, v in edges]

        # 2. Solve Set Packing
        # Greedy + Local Search
        edge_to_cands = defaultdict(list)
        for i, (vlist, eset) in enumerate(candidates):
            for e in eset:
                edge_to_cands[e].append(i)
        
        used_edges = set()
        selected = set()
        
        # Sort by length descending
        indices = sorted(range(len(candidates)), key=lambda i: len(candidates[i][0]), reverse=True)
        
        for i in indices:
            vlist, eset = candidates[i]
            if all(e not in used_edges for e in eset):
                selected.add(i)
                for e in eset:
                    used_edges.add(e)
        
        best_score = sum(len(candidates[i][0]) - 1 for i in selected)
        
        start_ls = time.time()
        improved = True
        while improved and time.time() - start_ls < 5.0:
            improved = False
            # Try Add
            for i in range(len(candidates)):
                if i in selected:
                    continue
                vlist, eset = candidates[i]
                if all(e not in used_edges for e in eset):
                    selected.add(i)
                    for e in eset:
                        used_edges.add(e)
                    best_score += len(vlist) - 1
                    improved = True
                    break
            
            # Try Swap
            if not improved:
                for i_remove in list(selected):
                    vlist_r, eset_r = candidates[i_remove]
                    temp_used = set(used_edges)
                    for e in eset_r:
                        temp_used.discard(e)
                    
                    best_gain = 0
                    best_i = -1
                    for i_add in range(len(candidates)):
                        if i_add in selected:
                            continue
                        vlist_a, eset_a = candidates[i_add]
                        if all(e not in temp_used for e in eset_a):
                            gain = (len(vlist_a) - 1) - (len(vlist_r) - 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_i = i_add
                    
                    if best_i != -1 and best_gain > 0:
                        selected.discard(i_remove)
                        selected.add(best_i)
                        used_edges = temp_used
                        for e in candidates[best_i][1]:
                            used_edges.add(e)
                        best_score += best_gain
                        improved = True
                        break

        # 3. Construct Result
        pieces = []
        for i in selected:
            vlist, _ = candidates[i]
            pieces.append(list(vlist))
        
        # Add remaining edges as single-edge pieces
        for u, v in edges:
            e = (min(u, v), max(u, v))
            if e not in used_edges:
                pieces.append([u, v])
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results

if __name__ == "__main__":
    construct()
