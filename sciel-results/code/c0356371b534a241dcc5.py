"""Rewritten program using AC-3 and Backtracking with Domain Copying."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: 120s limit, leave 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0

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

        doms = [set(l) for l in lists]
        if any(not d for d in doms):
            return None

        # AC-3
        q = [(u, v) for u in range(n) for v in adj[u]]
        def revise(i, j):
            if len(doms[j]) == 1:
                val = next(iter(doms[j]))
                if val in doms[i]:
                    doms[i].remove(val)
                    return True
            return False

        head = 0
        while head < len(q):
            i, j = q[head]
            head += 1
            if revise(i, j):
                if not doms[i]:
                    return None
                for k in adj[i]:
                    if k != j:
                        q.append((k, i))
        
        if any(not d for d in doms):
            return None

        def bt(doms, assign):
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            # MRV
            min_sz = float('inf')
            min_v = -1
            for i in range(n):
                if i not in assign:
                    sz = len(doms[i])
                    if sz < min_sz:
                        min_sz = sz
                        min_v = i
                        if sz == 0:
                            return None
                        if sz == 1:
                            break
            if min_v == -1:
                return assign # Solution found

            # LVC
            vals = list(doms[min_v])
            vals.sort(key=lambda c: sum(1 for nb in adj[min_v] if nb not in assign and c in doms[nb]))
            
            for c in vals:
                # Check validity
                ok = True
                for nb in adj[min_v]:
                    if nb in assign:
                        if assign[nb] == c:
                            ok = False
                            break
                    else:
                        if c in doms[nb] and len(doms[nb]) == 1:
                            ok = False
                            break
                if not ok:
                    continue
                
                # Propagate
                new_doms = [d.copy() for d in doms]
                new_doms[min_v] = {c}
                for nb in adj[min_v]:
                    if nb not in assign:
                        if c in new_doms[nb]:
                            new_doms[nb].remove(c)
                            if not new_doms[nb]:
                                ok = False
                                break
                if not ok:
                    continue
                
                new_assign = dict(assign)
                new_assign[min_v] = c
                
                res = bt(new_doms, new_assign)
                if res is not None:
                    return res
            return None

        try:
            sol_dict = bt(doms, {})
            if sol_dict:
                return [int(sol_dict[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Generate n to keep structure
            # We need n. We can estimate or just generate.
            # Generating is fast.
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
