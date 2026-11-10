import time
import random
from collections import defaultdict
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe against 60s kill
    
    n = 12
    
    def get_cycle_lengths_fast(edges):
        """
        Returns a set of cycle lengths. 
        Optimized for small graphs (n=12).
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        for start in range(n):
            def dfs(node, path, visited):
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 4:
                        cycle_lengths.add(len(path))
                    elif neighbor > start and neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            
            dfs(start, [start], {start})
            
        return cycle_lengths

    def is_valid_fast(edges):
        """
        Checks:
        1. Simple (no loops, no multi-edges)
        2. Degree >= 3
        3. Triangle-free (Girth >= 4)
        4. Has a cycle
        """
        n = 12
        edge_set = set()
        deg = [0] * n
        
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
            
        for d in deg:
            if d < 3:
                return False
        
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        for u, v in edge_set:
            if adj[u] & adj[v]:
                return False
                
        return True

    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([offset + i, offset + 3 + j])
        return edges

    # Default best: Two K3,3 components. Cycle lengths {4, 6}. Raw value 2.
    best_edges = make_k33(0) + make_k33(6)
    best_score = 2
    
    random.seed(42)
    
    def generate_random_3reg_bipartite_v2():
        """Generate a random 3-regular bipartite graph using rejection sampling on 0-1 matrices."""
        # We need a 6x6 binary matrix with row sums = 3 and col sums = 3
        # Use rejection sampling: generate random subsets of size 3 for each row,
        # then check if col sums are all 3.
        
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random row assignments
            rows = []
            for i in range(6):
                row = random.sample(range(6), 3)
                rows.append(set(row))
            
            # Check column sums
            col_counts = [0] * 6
            for i in range(6):
                for j in rows[i]:
                    col_counts[j] += 1
            
            if all(c == 3 for c in col_counts):
                # Build edges
                edges = []
                seen = set()
                valid = True
                for i in range(6):
                    for j in rows[i]:
                        u = i
                        v = 6 + j
                        key = (min(u, v), max(u, v))
                        if key in seen:
                            valid = False
                            break
                        seen.add(key)
                        edges.append([u, v])
                    if not valid:
                        break
                if valid:
                    return edges
        
        return None

    def generate_random_3reg_bipartite_v3():
        """Generate using three random perfect matchings that are disjoint."""
        # Start with identity matching, then add two more random matchings
        # that don't overlap with existing edges.
        
        for _ in range(200):
            # First matching: identity
            m1 = list(range(6))
            random.shuffle(m1)
            
            # Second matching: must be different from m1 for each row
            m2 = []
            valid = True
            for i in range(6):
                candidates = [j for j in range(6) if j != m1[i]]
                if not candidates:
                    valid = False
                    break
                m2.append(random.choice(candidates))
            if not valid:
                continue
            
            # Check m2 is a valid permutation
            if len(set(m2)) != 6:
                continue
            
            # Check m2 doesn't create duplicate edges with m1
            # m1[i] != m2[i] for all i (already ensured)
            
            # Third matching: must be different from m1 and m2 for each row
            m3 = []
            valid = True
            for i in range(6):
                candidates = [j for j in range(6) if j != m1[i] and j != m2[i]]
                if not candidates:
                    valid = False
                    break
                m3.append(random.choice(candidates))
            if not valid:
                continue
            
            # Check m3 is a valid permutation
            if len(set(m3)) != 6:
                continue
            
            # Build edges and check for duplicates
            edges = []
            seen = set()
            valid = True
            for i in range(6):
                for j in [m1[i], m2[i], m3[i]]:
                    u = i
                    v = 6 + j
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        valid = False
                        break
                    seen.add(key)
                    edges.append([u, v])
                if not valid:
                    break
            
            if valid:
                return edges
        
        return None

    def local_search(edges, max_iters=5000):
        """Try to reduce number of distinct cycle lengths via edge swaps."""
        if edges is None:
            return None, best_score
        
        current_edges = [list(e) for e in edges]
        current_cl = get_cycle_lengths_fast(current_edges)
        current_score = len(current_cl)
        
        if current_score <= 1:
            return current_edges, current_score
        
        edge_set = set()
        for u, v in current_edges:
            key = (min(u, v), max(u, v))
            edge_set.add(key)
        
        adj = defaultdict(set)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # Try a random edge swap: remove edge (a,b) and add edge (c,d)
            # where a,b,c,d are vertices
            # Must maintain: no loops, no multi-edges, degree >= 3, triangle-free
            
            # Pick two edges to remove
            if len(current_edges) < 2:
                break
            e1_idx = random.randint(0, len(current_edges) - 1)
            e2_idx = random.randint(0, len(current_edges) - 1)
            if e1_idx == e2_idx:
                continue
            
            a, b = current_edges[e1_idx]
            c, d = current_edges[e2_idx]
            
            # Try swapping: remove (a,b) and (c,d), add (a,c) and (b,d) or (a,d) and (b,c)
            for swap_type in range(2):
                if swap_type == 0:
                    new_e1 = (a, c)
                    new_e2 = (b, d)
                else:
                    new_e1 = (a, d)
                    new_e2 = (b, c)
                
                # Check validity of new edges
                valid_swap = True
                for ne in [new_e1, new_e2]:
                    u, v = ne
                    if u == v:
                        valid_swap = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        valid_swap = False
                        break
                
                if not valid_swap:
                    continue
                
                # Check that removing e1, e2 and adding new_e1, new_e2 maintains degree >= 3
                # Vertices affected: a, b, c, d
                # a loses b, gains one of c/d
                # b loses a, gains one of c/d  
                # c loses d, gains one of a/b
                # d loses c, gains one of a/b
                
                # All lose 1 and gain 1, so degrees stay the same. 
                # But we need to check triangle-freeness.
                
                # Build new adjacency
                new_adj = defaultdict(set)
                for u, v in current_edges:
                    if (min(u,v), max(u,v)) == (min(a,b), max(a,b)):
                        continue
                    if (min(u,v), max(u,v)) == (min(c,d), max(c,d)):
                        continue
                    new_adj[u].add(v)
                    new_adj[v].add(u)
                
                u1, v1 = new_e1
                u2, v2 = new_e2
                new_adj[u1].add(v1)
                new_adj[v1].add(u1)
                new_adj[u2].add(v2)
                new_adj[v2].add(u2)
                
                # Check triangle-free
                tri_free = True
                for u in range(n):
                    neighbors = list(new_adj[u])
                    for i in range(len(neighbors)):
                        for j in range(i+1, len(neighbors)):
                            if neighbors[j] in new_adj[neighbors[i]]:
                                tri_free = False
                                break
                        if not tri_free:
                            break
                    if not tri_free:
                        break
                
                if not tri_free:
                    continue
                
                # Build new edge list
                new_edges = []
                for u, v in current_edges:
                    if (min(u,v), max(u,v)) == (min(a,b), max(a,b)):
                        continue
                    if (min(u,v), max(u,v)) == (min(c,d), max(c,d)):
                        continue
                    new_edges.append([u, v])
                new_edges.append([u1, v1])
                new_edges.append([u2, v2])
                
                # Check degree
                deg = [0] * n
                for u, v in new_edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d < 3 for d in deg):
                    continue
                
                # Compute cycle lengths
                new_cl = get_cycle_lengths_fast(new_edges)
                new_score = len(new_cl)
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_cl = new_cl
                    current_score = new_score
                    improved = True
                    
                    # Update edge_set and adj
                    edge_set = set()
                    adj = defaultdict(set)
                    for u, v in current_edges:
                        key = (min(u, v), max(u, v))
                        edge_set.add(key)
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    if current_score <= 1:
                        return current_edges, current_score
                    break  # Restart the swap search with new graph
        
        return current_edges, current_score

    # Strategy 1: Try to find a graph with 1 cycle length via random generation + local search
    # Strategy 2: Try specific constructions
    
    # First, try many random bipartite graphs with local search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Alternate between generators
        if attempts % 2 == 0:
            edges = generate_random_3reg_bipartite_v2()
        else:
            edges = generate_random_3reg_bipartite_v3()
        
        if edges is None:
            continue
        
        if not is_valid_fast(edges):
            continue
        
        cl = get_cycle_lengths_fast(edges)
        if not cl:
            continue
        
        score = len(cl)
        if score < best_score:
            best_score = score
            best_edges = edges
            if score == 1:
                break
        
        # Try local search if score is promising
        if score <= 2 and time.time() < deadline - 5.0:
            ls_edges, ls_score = local_search(edges, max_iters=2000)
            if ls_score < best_score:
                best_score = ls_score
                best_edges = ls_edges
                if ls_score == 1:
                    break
    
    # Strategy 3: Try specific known constructions
    # The complete bipartite graph K_{3,3} has cycles {4, 6}
    # What about other 3-regular bipartite graphs?
    
    # Try the "crown" graph or other specific structures
    # For now, also try a few hand-crafted graphs
    
    # Check if we found score 1
    if best_score == 1:
        result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
        return result
    
    # Ensure types are correct
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
    
    return result
# EVOLVE-BLOCK-END
