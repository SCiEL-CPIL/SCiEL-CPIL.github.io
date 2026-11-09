import time
import random
import sys
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(1, N + 1))

    # Precompute conflict graph: conflict[i] = set of j != i where (i*j)%(i+j)==0
    # Using the efficient check: let g=gcd(a,b), u=a/g, v=b/g; conflict iff (u+v)%g==0
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            u = i // g
            v = j // g
            if (u + v) % g == 0:
                conflict[i].add(j)
                conflict[j].add(i)

    # Build adjacency as lists for quick access
    adj = [sorted(conflict[i]) for i in range(N + 1)]
    degree = [len(adj[i]) for i in range(N + 1)]

    # Greedy independent set construction (min-degree heuristic)
    def greedy_min_degree(rng):
        remaining = set(range(1, N + 1))
        selected = []
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u in conflict[v])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            selected.append(best_v)
            remaining.discard(best_v)
            for n in conflict[best_v]:
                remaining.discard(n)
        return selected

    # Local search: try to add a vertex, if conflict, try to remove one conflicting vertex
    def local_search_improve(S, rng, max_iters=200):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        all_nums = set(range(1, N + 1))

        for it in range(max_iters):
            if elapsed() > deadline - 5:
                break
            # Try to add a random vertex not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                break
            rng.shuffle(candidates)
            added_any = False
            for add_elem in candidates[:20]:
                # Check conflicts
                conflicts_with_S = conflict[add_elem] & S
                if not conflicts_with_S:
                    S.add(add_elem)
                    added_any = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
                else:
                    # Try removing one conflicting element and adding
                    for rem in list(conflicts_with_S):
                        S.remove(rem)
                        # Check if add_elem now fits
                        new_conflicts = conflict[add_elem] & S
                        if not new_conflicts:
                            S.add(add_elem)
                            added_any = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                            break
                        else:
                            S.add(rem)  # restore
            if not added_any:
                # Try 2-swap: remove 2, add 2
                if len(S) >= 2:
                    S_list = list(S)
                    rng.shuffle(S_list)
                    a, b = S_list[0], S_list[1]
                    S.remove(a)
                    S.remove(b)
                    candidates2 = [x for x in range(1, N + 1) if x not in S]
                    rng.shuffle(candidates2)
                    added_count = 0
                    for add_elem in candidates2[:30]:
                        if added_count >= 2:
                            break
                        if not (conflict[add_elem] & S):
                            S.add(add_elem)
                            added_count += 1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    else:
                        # Revert
                        for x in list(S):
                            if x not in best_S:
                                S.discard(x)
                        S.add(a)
                        S.add(b)
        return best_S

    # Tomita-style Maximum Independent Set with coloring bound
    # Vertices are 1..N, use 0-indexed internally (vertex i -> index i-1)
    # We work with bitmasks where bit i represents vertex (i+1)

    sys.setrecursionlimit(100000)

    def popcount(x):
        return bin(x).count('1')

    # Precompute for each vertex, the bitmask of its neighbors (among 1..N, 0-indexed)
    # neighbor_mask[v] where v is 0-indexed (0..255), bit j set if vertex (j+1) is neighbor of (v+1)
    neighbor_mask = [0] * N
    for v in range(1, N + 1):
        mask = 0
        for u in conflict[v]:
            mask |= (1 << (u - 1))
        neighbor_mask[v - 1] = mask

    # Tomita's algorithm: color sort for upper bound
    # Colors are assigned greedily; the color number of a vertex is an upper bound on the
    # size of independent set in the subgraph induced by that vertex and later vertices.

    def color_sort(vertices):
        """
        vertices: list of 0-indexed vertex indices in the current candidate set.
        Returns (sorted_vertices, colors) where colors[i] is the color class number
        for sorted_vertices[i]. The last color is an upper bound on MIS size.
        """
        if not vertices:
            return [], []
        
        # Greedy coloring: process vertices, assign smallest available color
        # We want to sort by color (higher color = potentially larger independent set contribution)
        color_of = {}
        colors_used = []  # list of sets of vertices per color
        
        for v in vertices:
            # Find smallest color not used by neighbors of v in the colored set
            neighbor_colors = set()
            for u in vertices:
                if u != v and color_of.get(u) is not None:
                    if (1 << v) & neighbor_mask[u]:  # v is neighbor of u
                        neighbor_colors.add(color_of[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            if c >= len(colors_used):
                colors_used.append(set())
            colors_used[c].add(v)
        
        max_color = max(color_of.values()) if color_of else 0
        # Sort vertices by color descending, then by degree descending
        sorted_v = sorted(vertices, key=lambda v: (-color_of[v], -degree[v+1]))
        colors = [color_of[v] for v in sorted_v]
        return sorted_v, colors

    def mis_branch_and_bound(candidates, current_set, current_size, best_size_holder):
        """
        candidates: bitmask of 0-indexed vertices available
        current_set: list of 0-indexed vertices in current independent set
        current_size: len(current_set)
        best_size_holder: [best_size, best_set] mutable
        """
        if elapsed() > deadline - 2:
            return
        
        if candidates == 0:
            if current_size > best_size_holder[0]:
                best_size_holder[0] = current_size
                best_size_holder[1] = list(current_set)
            return
        
        # Convert bitmask to list of vertices
        verts = []
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
        
        # Color sort
        sorted_verts, colors = color_sort(verts)
        
        if not sorted_verts:
            return
        
        # Upper bound: current_size + max color
        if current_size + colors[-1] <= best_size_holder[0]:
            return
        
        # Branch: try vertices in order (highest color first)
        for i, v in enumerate(sorted_verts):
            if elapsed() > deadline - 2:
                return
            
            # Upper bound for this branch: current_size + 1 + (max color of remaining after v)
            # The remaining candidates are those in sorted_verts[i+1:] that are not neighbors of v
            remaining_mask = 0
            for j in range(i + 1, len(sorted_verts)):
                u = sorted_verts[j]
                if not ((1 << v) & neighbor_mask[u]):
                    remaining_mask |= (1 << u)
            
            # Recompute upper bound for remaining
            if remaining_mask:
                rem_verts = []
                temp2 = remaining_mask
                while temp2:
                    lsb2 = temp2 & (-temp2)
                    idx2 = lsb2.bit_length() - 1
                    rem_verts.append(idx2)
                    temp2 ^= lsb2
                if rem_verts:
                    _, rem_colors = color_sort(rem_verts)
                    ub = current_size + 1 + (rem_colors[-1] if rem_colors else 0)
                else:
                    ub = current_size + 1
            else:
                ub = current_size + 1
            
            if ub <= best_size_holder[0]:
                break  # No further vertices can improve
            
            # Include v
            new_candidates = remaining_mask  # already excludes neighbors of v
            current_set.append(v)
            mis_branch_and_bound(new_candidates, current_set, current_size + 1, best_size_holder)
            current_set.pop()
            
            # Exclude v: remove v from candidates
            # We just continue the loop, which naturally excludes v by moving to next i

    # Strategy: multiple greedy starts + local search, then try B&B if time permits
    best_size = 0
    best_solution = set()

    # Phase 1: Multiple greedy min-degree trials
    for trial in range(50):
        if elapsed() > deadline - 30:
            break
        rng = random.Random(trial * 7919 + 13)
        S = greedy_min_degree(rng)
        if len(S) > best_size:
            best_size = len(S)
            best_solution = set(S)
    
    # Phase 2: Local search on best solution
    if elapsed() < deadline - 20:
        rng = random.Random(12345)
        improved = local_search_improve(best_solution, rng, max_iters=300)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved

    # Phase 3: More local search iterations with different seeds
    for trial in range(10):
        if elapsed() > deadline - 10:
            break
        rng = random.Random(trial * 31337 + 999)
        S = set(best_solution)
        improved = local_search_improve(S, rng, max_iters=100)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved

    # Phase 4: Try Tomita B&B from the best solution as initial lower bound
    if elapsed() < deadline - 5:
        # Use the current best as lower bound
        best_size_holder = [best_size, [v - 1 for v in best_solution]]  # 0-indexed
        
        # Start B&B from full candidate set
        full_mask = (1 << N) - 1
        mis_branch_and_bound(full_mask, [], 0, best_size_holder)
        
        if best_size_holder[0] > best_size:
            best_size = best_size_holder[0]
            best_solution = set(v + 1 for v in best_size_holder[1])

    # Final verification
    result = sorted(best_solution)
    
    # Verify validity
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: return just odds (always valid since for odd a,b: a+b even, a*b odd, so (a+b)|a*b impossible)
        result = [x for x in range(1, N + 1) if x % 2 == 1]

    return result

if __name__ == "__main__":
    result = construct()
    print(len(result))
