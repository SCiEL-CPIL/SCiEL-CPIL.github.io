import numpy as np
from heapq import heappush, heappop

def plan(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    customers = list(range(1, n_customers + 1))
    remaining = set(customers)
    routes = []

    STATE_CAP = 200000

    def build_route(rem):
        rem = list(rem)
        # state key: (visited_frozenset, last, bat_int)
        # best_time[key] = min time
        best_time = {}
        parent = {}  # key -> (prev_key, added_node)
        pq = []
        start_key = (frozenset(), 0, int(round(battery_max)))
        best_time[start_key] = 0.0
        # heap item: (neg_count, dist, time, visited, last, bat)
        heappush(pq, (0, 0.0, 0.0, frozenset(), 0, battery_max))
        expanded = 0
        best_complete = None  # (count, dist, key)

        while pq and expanded < STATE_CAP:
            negc, d, t, visited, last, bat = heappop(pq)
            count = -negc
            key = (visited, last, int(round(bat)))
            if best_time.get(key, float('inf')) < t - 1e-9:
                continue
            expanded += 1

            # Try complete: return to depot
            d_back = float(D[last][0])
            nb = bat - d_back
            nt = t + d_back / speed
            if nb >= -1e-9 and nt <= horizon + 1e-9:
                cand = (count, d + d_back, key)
                if best_complete is None or cand[0] > best_complete[0] or (cand[0] == best_complete[0] and cand[1] < best_complete[1] - 1e-9):
                    best_complete = cand

            # Extend
            # candidates: remaining customers not in visited, and stations
            for c in rem:
                if c in visited:
                    continue
                if demands[c] + sum(demands[v] for v in visited) > capacity:
                    continue
                dstep = float(D[last][c])
                nb = bat - dstep
                if nb < -1e-9:
                    continue
                t_arr = t + dstep / speed
                t_arr = max(t_arr, float(ready[c]))
                if t_arr > float(due[c]) + 1e-9:
                    continue
                nt = t_arr + float(service[c])
                nvis = visited | {c}
                nkey = (nvis, c, int(round(nb)))
                if nt < best_time.get(nkey, float('inf')) - 1e-9:
                    best_time[nkey] = nt
                    parent[nkey] = (key, c)
                    heappush(pq, (-len(nvis), d + dstep, nt, nvis, c, nb))
            for s in stations:
                dstep = float(D[last][s])
                nb = bat - dstep
                if nb < -1e-9:
                    continue
                t_arr = t + dstep / speed
                recharge_t = (battery_max - nb) * recharge_rate
                nt = t_arr + recharge_t
                # battery after refill
                nb2 = battery_max
                nkey = (visited, s, int(round(nb2)))
                if nt < best_time.get(nkey, float('inf')) - 1e-9:
                    best_time[nkey] = nt
                    parent[nkey] = (key, s)
                    heappush(pq, (-len(visited), d + dstep, nt, visited, s, nb2))

        if best_complete is None:
            return None
        count, dist, key = best_complete
        # reconstruct
        path = []
        cur = key
        while cur is not None:
            # cur = (visited, last, bat)
            path.append(cur[1])
            if cur in parent:
                cur = parent[cur][0]
            else:
                cur = None
        # path is in reverse: [last, ..., first_added, 0]
        path.reverse()
        # path[0] should be 0 (depot)
        route = [0] + path + [0]
        # remove depot internal duplicates? path includes 0 at start.
        # Actually path after reverse: [0, n1, n2, ..., last]
        # route = [0] + [0, n1,...,last] + [0] -> duplicate 0. Fix:
        route = path + [0]  # path already starts with 0
        # The customers served:
        served = set()
        for node in route:
            if node in rem and node not in stations:
                served.add(node)
        return route, served

    while remaining:
        res = build_route(remaining)
        if res is None:
            # fallback: single customer routes
            for c in sorted(remaining, key=lambda v: due[v]):
                routes.append([0, c, 0])
            break
        route, served = res
        routes.append(route)
        remaining -= served
        if not served:
            # avoid infinite loop: force single
            c = min(remaining, key=lambda v: due[v])
            routes.append([0, c, 0])
            remaining.discard(c)

    return routes
