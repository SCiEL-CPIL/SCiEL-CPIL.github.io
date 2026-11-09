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

        # Use bitmasks for domains for faster operations
        # Colors are 0..4, so we use bits 0..4
        doms = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            doms[i] = mask
        
        if any(d == 0 for d in doms):
            return None

        # Full AC-3 with proper revision
        q = [(u, v) for u in range(n) for v in adj[u]]
        
        def revise(i, j):
            di = doms[i]
            dj = doms[j]
            if not di or not dj:
                return di == 0 or dj == 0
            # For each color in di, check if it has a support in dj
            # i.e., there exists a color in dj different from it
            # Equivalent: di & ~dj gives colors in di with no support in dj
            # But we need: for each c in di, there must be some c' in dj with c' != c
            # This fails only if dj == {c} for some c in di
            # So: remove from di any color c where dj == {c}
            removed = 0
            d = di
            while d:
                lsb = d & -d
                c_bit = lsb
                # Check if dj is exactly {c_bit}
                if dj == c_bit:
                    removed |= c_bit
                d ^= lsb
            if removed:
                doms[i] = di & ~removed
                return True
            return False

        head = 0
        while head < len(q):
            i, j = q[head]
            head += 1
            if revise(i, j):
                if doms[i] == 0:
                    return None
                for k in adj[i]:
                    if k != j:
                        q.append((k, i))
        
        if any(d == 0 for d in doms):
            return None

        # Try greedy assignment first (fast path)
        assign = [-1] * n
        unassigned = list(range(n))
        
        def try_greedy():
            nonlocal assign, unassigned
            while unassigned:
                # MRV
                min_idx = 0
                min_sz = 999
                for idx, v in enumerate(unassigned):
                    sz = doms[v].bit_count()
                    if sz < min_sz:
                        min_sz = sz
                        min_idx = idx
                        if sz <= 1:
                            break
                v = unassigned[min_idx]
                if min_sz == 0:
                    return False
                
                # Find a valid color
                mask = doms[v]
                valid_colors = []
                while mask:
                    lsb = mask & -mask
                    c = lsb.bit_length() - 1
                    valid_colors.append(c)
                    mask ^= lsb
                
                # Check against assigned neighbors
                found = False
                for c in valid_colors:
                    ok = True
                    for nb in adj[v]:
                        if assign[nb] == c:
                            ok = False
                            break
                    if ok:
                        assign[v] = c
                        # Forward check
                        bit = 1 << c
                        conflict = False
                        for nb in adj[v]:
                            if assign[nb] == -1 and doms[nb] & bit:
                                doms[nb] &= ~bit
                                if doms[nb] == 0:
                                    conflict = True
                                    break
                        if not conflict:
                            unassigned.pop(min_idx)
                            found = True
                            break
                        
                        # Undo forward check
                        for nb in adj[v]:
                            if assign[nb] == -1:
                                pass  # We'll restore from original below
            
            if not found:
                return False
            return True
        
        # Save original domains for backtracking
        orig_doms = doms[:]
        
        # Attempt greedy
        if try_greedy():
            if all(a != -1 for a in assign):
                return [int(a) for a in assign]
        
        # Restore and do full backtracking
        doms = orig_doms[:]
        assign = [-1] * n
        unassigned = list(range(n))
        
        def bt():
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            if not unassigned:
                return True
            
            # MRV: Find unassigned vertex with smallest domain size
            min_idx = 0
            min_sz = 999
            for idx, v in enumerate(unassigned):
                sz = doms[v].bit_count()
                if sz < min_sz:
                    min_sz = sz
                    min_idx = idx
                    if sz <= 1:
                        break
            
            v = unassigned[min_idx]
            if min_sz == 0:
                return False
            
            # Get colors
            mask = doms[v]
            vals = []
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                vals.append(c)
                mask ^= lsb
            
            # Sort by least constraining value
            def count_conflicts(c):
                bit = 1 << c
                cnt = 0
                for nb in adj[v]:
                    if assign[nb] == -1:
                        if doms[nb] & bit:
                            cnt += 1
                return cnt
            
            vals.sort(key=count_conflicts)
            
            for c in vals:
                bit = 1 << c
                
                # Check validity against assigned neighbors
                ok = True
                for nb in adj[v]:
                    if assign[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Save state for rollback
                saved_doms = doms[:]
                assign[v] = c
                unassigned.pop(min_idx)
                
                # Forward checking
                conflict = False
                for nb in adj[v]:
                    if assign[nb] == -1 and doms[nb] & bit:
                        doms[nb] &= ~bit
                        if doms[nb] == 0:
                            conflict = True
                            break
                
                if not conflict:
                    if bt():
                        return True
                
                # Rollback
                unassigned.insert(min_idx, v)
                assign[v] = -1
                doms[:] = saved_doms
            
            return False

        try:
            if bt():
                return [int(a) for a in assign]
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
