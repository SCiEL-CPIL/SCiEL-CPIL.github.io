"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    # Try to find a valid coloring for n=30 using random search
    n_target = 30
    time_limit = 100
    
    def verify(colors, n):
        adj0 = set()
        adj1 = set()
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0.add((i, j))
                else:
                    adj1.add((i, j))
                idx += 1
        
        def contains_k222(adj):
            verts = list(range(n))
            for combo in itertools.combinations(verts, 6):
                a, b, c, d, e, f = combo
                partitions = [
                    [(a,b),(c,d),(e,f)],
                    [(a,c),(b,d),(e,f)],
                    [(a,d),(b,c),(e,f)],
                    [(a,b),(c,e),(d,f)],
                    [(a,c),(b,e),(d,f)],
                    [(a,e),(b,c),(d,f)],
                    [(a,b),(c,f),(d,e)],
                    [(a,c),(b,f),(d,e)],
                    [(a,f),(b,c),(d,e)],
                    [(a,d),(b,e),(c,f)],
                    [(a,e),(b,d),(c,f)],
                    [(a,f),(b,d),(c,e)],
                    [(a,d),(b,f),(c,e)],
                    [(a,e),(b,f),(c,d)],
                    [(a,f),(b,e),(c,d)],
                ]
                for part in partitions:
                    ok = True
                    for idx1 in range(3):
                        for idx2 in range(idx1 + 1, 3):
                            u1, v1 = part[idx1]
                            u2, v2 = part[idx2]
                            for u in (u1, v1):
                                for v in (u2, v2):
                                    if (u, v) not in adj and (v, u) not in adj:
                                        ok = False
                                        break
                                if not ok: break
                            if not ok: break
                        if not ok: break
                    if ok:
                        return True
            return False
        
        return contains_k222(adj0) or contains_k222(adj1)

    def generate_paley(n):
        if n == 29:
            squares = set()
            for x in range(1, 29):
                squares.add((x * x) % 29)
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 29
                    colors.append(0 if diff in squares else 1)
            return colors
        return None

    # Fallback
    fallback_colors = generate_paley(29)
    
    # Random search for n=30
    start_time = time.time()
    found = False
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        colors = [random.randint(0, 1) for _ in range(n_target * (n_target - 1) // 2)]
        # Simple local search
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
            idx = random.randint(0, len(colors) - 1)
            colors[idx] = 1 - colors[idx]
            # Check if valid (expensive, so do it less frequently)
            if step % 100 == 0:
                if not verify(colors, n_target):
                    found = True
                    break
        if found:
            return {'n': n_target, 'colors': colors}
            
    return {'n': 29, 'colors': fallback_colors}
# EVOLVE-BLOCK-END
