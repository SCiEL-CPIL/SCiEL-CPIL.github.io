"""Rewritten program using MAC (Maintaining Arc Consistency) and Trail-Based Backtracking with Bitmasks."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Bitmask domains: bit c is 1 if color c is in domain
        # Colors are 0..4, so we use bits 0..4
        doms = []
        for l in lists:
            mask = 0
            for c in l:
                mask |= (1 << c)
            doms.append(mask)
        
        if any(d == 0 for d in doms):
            return None

        # Trail for undoing: stores (var, old_mask)
        trail = []

        def ac3(initial_queue=None):
            if initial_queue is None:
                queue = [(u, v) for u in range(n) for v in adj[u]]
            else:
                queue = list(initial_queue)
            
            head = 0
            while head < len(queue):
                i, j = queue[head]
                head += 1
                
                # If dom[j] is a singleton, remove that color from dom[i]
                # Check if dom[j] is a power of 2 (singleton)
                dj = doms[j]
                if dj and (dj & (dj - 1)) == 0:
                    # dj is a singleton
                    # Check if dom[i] contains this color
                    if doms[i] & dj:
                        old_mask = doms[i]
                        new_mask = doms[i] & ~dj
                        if new_mask != old_mask:
                            doms[i] = new_mask
                            trail.append((i, old_mask))
                            if new_mask == 0:
                                return False
                            for k in adj[i]:
                                if k != j:
                                    queue.append((k, i))
                elif dj and doms[i]:
                    # General arc-consistency: remove colors from dom[i]
                    # that have no support in dom[j]
                    # For each color c in dom[i], check if dom[j] has a
                    # different color (support exists iff dom[j] != {c})
                    # i.e., dom[j] & ~(1<<c) != 0
                    di = doms[i]
                    # Build mask of colors in di that are NOT supported by dj
                    # A color c in di is unsupported if dj == (1<<c)
                    # So unsupported = di & dj  (colors that are in both,
                    # and dj is singleton with that color)
                    # Wait: if dj has multiple colors, then for any c in di,
                    # there's some d in dj with d != c (since |dj|>=2 > 1).
                    # So no removal needed when |dj| >= 2.
                    # Only singleton dj causes removals, handled above.
                    pass
            return True

        # Initial AC-3
        mark = len(trail)
        if not ac3():
            return None
        
        # Quick greedy attempt with forward checking (cheap fast path)
        assignment = [-1] * n
        greedy_ok = True
        for _ in range(n):
            # Find MRV
            min_size = 6
            min_node = -1
            for i in range(n):
                if assignment[i] == -1:
                    sz = bin(doms[i]).count('1')
                    if sz < min_size:
                        min_size = sz
                        min_node = i
                        if sz == 1:
                            break
            if min_node == -1:
                break
            if min_size == 0:
                greedy_ok = False
                break
            # Pick first available color
            mask = doms[min_node]
            c = 0
            while mask:
                if mask & 1:
                    break
                mask >>= 1
                c += 1
            # Check neighbors
            valid = True
            for nb in adj[min_node]:
                if assignment[nb] == c:
                    valid = False
                    break
            if not valid:
                greedy_ok = False
                break
            assignment[min_node] = c
            # Forward check: remove c from unassigned neighbors' domains
            bit = 1 << c
            for nb in adj[min_node]:
                if assignment[nb] == -1 and doms[nb] & bit:
                    old = doms[nb]
                    doms[nb] = old & ~bit
                    trail.append((nb, old))
                    if doms[nb] == 0:
                        greedy_ok = False
                        break
            if not greedy_ok:
                break
        if greedy_ok:
            # All assigned?
            all_assigned = all(assignment[i] != -1 for i in range(n))
            if all_assigned:
                return [int(assignment[i]) for i in range(n)]
            # Partial assignment with no conflicts; undo and go to backtrack
            while len(trail) > mark:
                var, old_mask = trail.pop()
                doms[var] = old_mask
            assignment = [-1] * n

        def backtrack():
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            # Find MRV node
            min_size = 6 # Max possible size is 5, so 6 is inf
            min_node = -1
            for i in range(n):
                if assignment[i] == -1:
                    size = bin(doms[i]).count('1')
                    if size < min_size:
                        min_size = size
                        min_node = i
                        if size == 0:
                            return False
                        if min_size == 1:
                            break

            if min_node == -1:
                return True # Success

            # Get values from domain
            mask = doms[min_node]
            vals = []
            c = 0
            while mask:
                if mask & 1:
                    vals.append(c)
                mask >>= 1
                c += 1
            
            # LVC: Sort values by least constraining (impact)
            # Impact: number of unassigned neighbors that have this color in their domain
            def impact(c):
                bit = 1 << c
                cnt = 0
                for nb in adj[min_node]:
                    if assignment[nb] == -1:
                        if doms[nb] & bit:
                            cnt += 1
                return cnt
            vals.sort(key=impact)

            for c in vals:
                bit = 1 << c
                
                # Check consistency with assigned neighbors
                valid = True
                for nb in adj[min_node]:
                    if assignment[nb] != -1:
                        if assignment[nb] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Save state
                mark = len(trail)
                
                # Assign c to min_node
                assignment[min_node] = c
                # Restrict doms[min_node] to {c}
                old_mask = doms[min_node]
                doms[min_node] = bit
                trail.append((min_node, old_mask))
                
                # Propagate: Run AC-3 on neighbors
                # Queue arcs (nb, min_node)
                queue = [(nb, min_node) for nb in adj[min_node] if assignment[nb] == -1]
                
                if not ac3(queue):
                    # Conflict, undo and try next value
                    assignment[min_node] = -1
                    while len(trail) > mark:
                        var, old_mask = trail.pop()
                        doms[var] = old_mask
                    continue
                
                # Recurse
                if backtrack():
                    return True
                
                # Undo
                assignment[min_node] = -1
                while len(trail) > mark:
                    var, old_mask = trail.pop()
                    doms[var] = old_mask
            
            return False

        try:
            if backtrack():
                return [int(assignment[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            n, _, _ = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            out.append([0] * n)
        else:
            out.append([int(c) for c in sol])
            
    return out
# EVOLVE-BLOCK-END
