import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def get_seed_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    seed_n = 32
    current_n = seed_n
    current_colors = get_seed_coloring(seed_n)
    best_n = seed_n
    best_colors = current_colors

    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_triangles(colors, n):
        adj = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        count = 0
        for c in range(4):
            for i in range(n):
                neigh = adj[c][i]
                for j in neigh:
                    if j <= i: continue
                    for k in neigh:
                        if k > j and k in adj[c][j]:
                            count += 1
        return count

    for target_n in range(seed_n + 1, 63):
        if time.time() > deadline - 30:
            break

        solved = False
        # Try up to 3 restarts for this n
        for attempt in range(3):
            if time.time() > deadline - 10:
                break

            colors = current_colors[:]
            for v in range(current_n, target_n):
                for u in range(v):
                    colors.append(random.randint(0, 3))

            curr_energy = count_triangles(colors, target_n)
            best_energy = curr_energy
            best_state = colors[:]

            T = 1.0
            T_min = 0.01
            cooling = 0.999

            while T > T_min:
                if time.time() > deadline - 5:
                    break

                u = random.randint(0, target_n - 1)
                v = random.randint(u + 1, target_n - 1)
                idx = get_idx(u, v, target_n)
                old_c = colors[idx]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue

                delta = 0
                for w in range(target_n):
                    if w == u or w == v:
                        continue
                    idx_uw = get_idx(u, w, target_n)
                    idx_vw = get_idx(v, w, target_n)
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]

                    if c_uw == c_vw:
                        if c_uw == old_c:
                            delta -= 1
                        if c_uw == new_c:
                            delta += 1

                if delta <= 0:
                    colors[idx] = new_c
                    curr_energy += delta
                    if curr_energy < best_energy:
                        best_energy = curr_energy
                        best_state = colors[:]
                else:
                    if random.random() < math.exp(-delta / T):
                        colors[idx] = new_c
                        curr_energy += delta

                T *= cooling

            if best_energy == 0:
                best_n = target_n
                best_colors = best_state
                current_n = target_n
                current_colors = best_state
                solved = True
                break

        if not solved:
            break

    return {'n': best_n, 'colors': best_colors}
