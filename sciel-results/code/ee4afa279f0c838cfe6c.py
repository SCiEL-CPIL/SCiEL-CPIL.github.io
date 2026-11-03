import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return deadline - elapsed()

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g):
        """Lazebnik-Ustimenko-Woldar type construction."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    seed_n, seed_edges = seed_construct(k, g)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check_nonbacktracking(n, adj, g):
        """
        Check girth >= g using non-backtracking walks.
        For girth >= g, no non-backtracking walk of length < g can return to its start.
        We check: for each directed edge (u,v), can we reach (v,u) via a non-backtracking
        walk of length < g-2? If yes, there's a cycle of length < g.
        
        More efficiently: do BFS from each vertex on the "line graph" where states are
        (current, prev) and we track shortest non-backtracking path back.
        """
        # For each vertex s, do a BFS where state is (node, prev_node)
        # Start: for each neighbor v of s, state (v, s) at depth 1
        # If we ever reach state (s, v) for any v, we have a cycle of length depth+1
        # Stop when depth >= g-1 (since cycle length would be depth+1 >= g)
        
        max_depth = g - 1  # We only need to check up to this depth
        
        for s in range(n):
            # visited[state] = depth at which we first reached it
            visited = {}
            queue = deque()
            
            for v in adj[s]:
                state = (v, s)
                if state not in visited:
                    visited[state] = 1
                    queue.append(state)
            
            while queue:
                curr, prev = queue.popleft()
                depth = visited[(curr, prev)]
                
                if depth >= max_depth:
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # non-backtracking
                    new_state = (nxt, curr)
                    if new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append(new_state)
                    # If we reach (s, *), that means we came back to s
                    if nxt == s:
                        # Cycle of length depth + 1
                        if depth + 1 < g:
                            return False
        
        return True

    def greedy_cage_construct(k, g, match_strategy=0):
        """
        Construct a k-regular graph with girth >= g using BFS tree + matching.
        For g = 2r+1, build the Moore-bound tree, then match leaf stubs
        across different branches to avoid short cycles.
        """
        r = (g - 1) // 2  # radius of BFS tree
        
        # Build BFS tree explicitly
        tree_edges = []
        vertex_data = {}  # vertex_id -> (level, parent, branch, level1_parent)
        
        vid = 0
        root = vid
        vertex_data[vid] = (0, -1, -1, -1)
        vid += 1
        
        # Level 1
        level1_vertices = []
        for i in range(k):
            v = vid
            vertex_data[v] = (1, root, v, v)
            level1_vertices.append(v)
            tree_edges.append((root, v))
            vid += 1
        
        # Levels 2 to r
        current_level = level1_vertices
        for lvl in range(2, r + 1):
            next_level = []
            for parent in current_level:
                branch = vertex_data[parent][2]
                l1p = vertex_data[parent][3]
                for i in range(k - 1):
                    v = vid
                    vertex_data[v] = (lvl, parent, branch, l1p)
                    next_level.append(v)
                    tree_edges.append((parent, v))
                    vid += 1
            current_level = next_level
        
        n_tree = vid
        
        # Group level-r vertices by their level-1 ancestor (branch)
        branch_groups = {}
        for v, (lvl, parent, branch, l1p) in vertex_data.items():
            if lvl == r:
                if l1p not in branch_groups:
                    branch_groups[l1p] = []
                branch_groups[l1p].append(v)
        
        # Each level-r vertex has (k-1) free stubs
        branch_list = sorted(branch_groups.keys())
        num_branches = len(branch_list)
        
        # Create stub list
        all_stubs = []  # (vertex_id, branch, stub_idx)
        for branch in branch_list:
            for v in branch_groups[branch]:
                for s in range(k - 1):
                    all_stubs.append((v, branch, s))
        
        total_stubs = len(all_stubs)
        
        # Deterministic matching strategies
        new_edges = []
        used = [False] * total_stubs
        
        # Group stubs by branch
        stubs_by_branch = {b: [] for b in branch_list}
        for idx, (v, b, s) in enumerate(all_stubs):
            stubs_by_branch[b].append(idx)
        
        # Strategy 0: round-robin across branches
        if match_strategy == 0:
            for bi, branch in enumerate(branch_list):
                stub_indices = stubs_by_branch[branch]
                for si, stub_idx in enumerate(stub_indices):
                    if used[stub_idx]:
                        continue
                    v1 = all_stubs[stub_idx][0]
                    
                    found = False
                    for offset in range(1, num_branches):
                        other_branch = branch_list[(bi + offset) % num_branches]
                        other_stubs = stubs_by_branch[other_branch]
                        for oj in range(len(other_stubs)):
                            other_idx = other_stubs[(si + oj) % len(other_stubs)]
                            if used[other_idx]:
                                continue
                            v2 = all_stubs[other_idx][0]
                            if v1 == v2:
                                continue
                            new_edges.append((v1, v2))
                            used[stub_idx] = True
                            used[other_idx] = True
                            found = True
                            break
                        if found:
                            break
                    if not found:
                        return None
        
        # Strategy 1: shuffle-based deterministic matching
        elif match_strategy == 1:
            # Use a fixed random seed for reproducibility
            rng = random.Random(42 + match_strategy)
            
            # Create a permutation of stub indices
            perm = list(range(total_stubs))
            rng.shuffle(perm)
            
            # Match consecutive pairs, ensuring they're from different branches
            i = 0
            while i < total_stubs:
                if used[perm[i]]:
                    i += 1
                    continue
                stub1_idx = perm[i]
                v1 = all_stubs[stub1_idx][0]
                b1 = all_stubs[stub1_idx][1]
                
                found = False
                for j in range(i + 1, total_stubs):
                    if used[perm[j]]:
                        continue
                    stub2_idx = perm[j]
                    v2 = all_stubs[stub2_idx][0]
                    b2 = all_stubs[stub2_idx][1]
                    
                    if b1 == b2:
                        continue
                    if v1 == v2:
                        continue
                    
                    new_edges.append((v1, v2))
                    used[stub1_idx] = True
                    used[stub2_idx] = True
                    found = True
                    break
                
                if not found:
                    return None
                i += 1
        
        # Strategy 2: greedy with distance maximization
        elif match_strategy == 2:
            # For each stub, match to the stub from a different branch that
            # is "farthest" in terms of tree distance (approximated by branch difference)
            for bi, branch in enumerate(branch_list):
                stub_indices = stubs_by_branch[branch]
                for si, stub_idx in enumerate(stub_indices):
                    if used[stub_idx]:
                        continue
                    v1 = all_stubs[stub_idx][0]
                    
                    best_other = -1
                    best_score = -1
                    for other_branch in branch_list:
                        if other_branch == branch:
                            continue
                        for other_idx in stubs_by_branch[other_branch]:
                            if used[other_idx]:
                                continue
                            v2 = all_stubs[other_idx][0]
                            if v1 == v2:
                                continue
                            # Score: prefer different level-1 ancestors
                            score = abs(branch_list.index(other_branch) - bi)
                            if score > best_score:
                                best_score = score
                                best_other = other_idx
                    
                    if best_other == -1:
                        return None
                    
                    v2 = all_stubs[best_other][0]
                    new_edges.append((v1, v2))
                    used[stub_idx] = True
                    used[best_other] = True
        
        if len(new_edges) != total_stubs // 2:
            return None
        
        # Check for duplicate edges and self-loops
        edge_set = set()
        for u, v in new_edges:
            key = (min(u, v), max(u, v))
            if key in edge_set or u == v:
                return None
            edge_set.add(key)
        
        # Check no edge duplicates tree edges
        tree_edge_set = set()
        for u, v in tree_edges:
            tree_edge_set.add((min(u, v), max(u, v)))
        for u, v in new_edges:
            if (min(u, v), max(u, v)) in tree_edge_set:
                return None
        
        all_edges = tree_edges + new_edges
        return n_tree, all_edges

    # Try multiple matching strategies
    best_result = None
    for strategy in range(3):
        if time_remaining() < 5:
            break
        result = greedy_cage_construct(k, g, match_strategy=strategy)
        if result is not None:
            n, edges = result
            # Quick degree check
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == k for d in deg):
                adj = build_adj(edges, n)
                # Verify girth with non-backtracking check
                if exact_girth_check_nonbacktracking(n, adj, g):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                    break  # First valid is good enough for this strategy
    
    if best_result is not None:
        return best_result
    
    # Fallback to seed
    return seed_n, seed_edges
