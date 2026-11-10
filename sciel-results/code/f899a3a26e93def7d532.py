import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 320  # safety margin before 330s kill

    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_index = {p: i for i, p in enumerate(all_points)}
    num_points = len(all_points)

    best_set = []
    best_size = 0

    def is_valid_set(selected):
        """Check if a set of points has no isosceles triangle."""
        n = len(selected)
        if n < 3:
            return True
        # Build distance matrix
        dist = [[0]*n for _ in range(n)]
        for i in range(n):
            xi, yi = selected[i]
            for j in range(i+1, n):
                xj, yj = selected[j]
                d = (xi-xj)**2 + (yi-yj)**2
                dist[i][j] = d
                dist[j][i] = d
        # Check each triangle
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if dist[i][j] == dist[i][k] or dist[i][j] == dist[j][k] or dist[i][k] == dist[j][k]:
                        return False
        return True

    def can_insert(px, py, selected, dist_from):
        """Check if (px,py) can be added. Returns (valid, d_list)."""
        d_list = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_list.append(d_sq)
        if len(set(d_list)) != len(d_list):
            return False, d_list
        for i in range(len(selected)):
            d_pa = d_list[i]
            if d_pa in dist_from[i]:
                return False, d_list
        return True, d_list

    def insert_point(p, selected, dist_from):
        px, py = p
        d_list = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_list.append(d_sq)
        selected.append(p)
        dist_from.append(set(d_list))
        for i in range(len(selected) - 1):
            dist_from[i].add(d_list[i])

    def remove_point(idx, selected, dist_from):
        selected.pop(idx)
        dist_from.pop(idx)
        new_dist_from = []
        for i in range(len(selected)):
            s = set()
            for j in range(len(selected)):
                if i == j:
                    continue
                dx = selected[i][0] - selected[j][0]
                dy = selected[i][1] - selected[j][1]
                s.add(dx*dx + dy*dy)
            new_dist_from.append(s)
        dist_from.clear()
        dist_from.extend(new_dist_from)

    def greedy_select(order, time_limit):
        """Greedy: add points in order if valid."""
        selected = []
        dist_from = []
        for p in order:
            if time.time() > time_limit:
                break
            valid, d_list = can_insert(p[0], p[1], selected, dist_from)
            if valid:
                insert_point(p, selected, dist_from)
        return selected, dist_from

    def local_improve(selected, dist_from, time_limit):
        """Local search: try adding points, then try swaps (remove one, add another)."""
        selected = list(selected)
        dist_from = [set(s) for s in dist_from]
        sel_set = set(selected)
        candidates = [p for p in all_points if p not in sel_set]
        random.shuffle(candidates)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding
            new_candidates = []
            for p in candidates:
                if time.time() > time_limit:
                    break
                valid, d_list = can_insert(p[0], p[1], selected, dist_from)
                if valid:
                    insert_point(p, selected, dist_from)
                    sel_set.add(p)
                    improved = True
                else:
                    new_candidates.append(p)
            candidates = new_candidates

            if not improved and time.time() < time_limit:
                # Try swaps: remove one selected, try to add a candidate
                for i in range(len(selected)):
                    if time.time() > time_limit:
                        break
                    removed = selected[i]
                    remove_point(i, selected, dist_from)
                    added_any = False
                    remaining_candidates = []
                    for p in candidates:
                        valid, d_list = can_insert(p[0], p[1], selected, dist_from)
                        if valid:
                            insert_point(p, selected, dist_from)
                            sel_set.discard(removed)
                            sel_set.add(p)
                            candidates = [c for c in candidates if c != p]
                            added_any = True
                            break
                        else:
                            remaining_candidates.append(p)
                    if not added_any:
                        # Restore removed point
                        insert_point(removed, selected, dist_from)
                        sel_set.add(removed)
                        candidates = remaining_candidates
                    else:
                        improved = True
        return selected, dist_from

    def simulated_annealing(selected, dist_from, time_limit):
        """SA: random swaps and additions/removals."""
        selected = list(selected)
        dist_from = [set(s) for s in dist_from]
        sel_set = set(selected)
        candidates = [p for p in all_points if p not in sel_set]
        random.shuffle(candidates)

        T = 10.0
        T_min = 0.1
        cooling = 0.995
        current_size = len(selected)
        best_local = list(selected)
        best_local_size = current_size

        step = 0
        while time.time() < time_limit and T > T_min:
            step += 1
            if step % 1000 == 0 and time.time() > time_limit:
                break

            # Random move
            move_type = random.random()
            if move_type < 0.4:
                # Try to add a random candidate
                if not candidates:
                    T *= cooling
                    continue
                ci = random.randrange(len(candidates))
                p = candidates[ci]
                valid, d_list = can_insert(p[0], p[1], selected, dist_from)
                if valid:
                    insert_point(p, selected, dist_from)
                    candidates.pop(ci)
                    sel_set.add(p)
                    current_size += 1
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = list(selected)
                else:
                    # Reject addition (always reject invalid)
                    pass
            elif move_type < 0.7:
                # Try to remove a random selected point
                if not selected:
                    T *= cooling
                    continue
                ri = random.randrange(len(selected))
                p = selected[ri]
                remove_point(ri, selected, dist_from)
                candidates.append(p)
                sel_set.discard(p)
                current_size -= 1
            else:
                # Try a swap: remove one selected, add one candidate
                if not selected or not candidates:
                    T *= cooling
                    continue
                ri = random.randrange(len(selected))
                ci = random.randrange(len(candidates))
                p_remove = selected[ri]
                p_add = candidates[ci]
                # Remove p_remove
                remove_point(ri, selected, dist_from)
                sel_set.discard(p_remove)
                # Try to add p_add
                valid, d_list = can_insert(p_add[0], p_add[1], selected, dist_from)
                if valid:
                    insert_point(p_add, selected, dist_from)
                    sel_set.add(p_add)
                    # Update candidates
                    candidates[ci] = p_remove
                else:
                    # Restore
                    insert_point(p_remove, selected, dist_from)
                    sel_set.add(p_remove)

            if current_size > best_local_size:
                best_local_size = current_size
                best_local = list(selected)

            T *= cooling

        # Rebuild dist_from for best_local
        new_dist_from = []
        for i in range(len(best_local)):
            s = set()
            for j in range(len(best_local)):
                if i == j:
                    continue
                dx = best_local[i][0] - best_local[j][0]
                dy = best_local[i][1] - best_local[j][1]
                s.add(dx*dx + dy*dy)
            new_dist_from.append(s)
        return best_local, new_dist_from

    # Main loop: multiple restarts with SA + local improvement
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 5:
            break

        # Random shuffle for diversity
        shuffled = list(all_points)
        random.shuffle(shuffled)

        # Greedy initialization
        greedy_limit = min(deadline - 15, start_time + 5)
        selected, dist_from = greedy_select(shuffled, greedy_limit)

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)

        # Local improvement
        if time.time() < deadline - 15:
            local_limit = min(deadline - 15, time.time() + 30)
            new_selected, new_dist = local_improve(selected, dist_from, local_limit)
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = list(new_selected)

        # Simulated annealing
        if time.time() < deadline - 15:
            sa_limit = min(deadline - 10, time.time() + 60)
            sa_selected, sa_dist = simulated_annealing(new_selected, new_dist, sa_limit)
            if len(sa_selected) > best_size:
                best_size = len(sa_selected)
                best_set = list(sa_selected)

        # If time remains, do another round of local improvement on best
        if time.time() < deadline - 10 and best_size > 0:
            # Rebuild dist_from for best_set
            bd_from = []
            for i in range(len(best_set)):
                s = set()
                for j in range(len(best_set)):
                    if i == j:
                        continue
                    dx = best_set[i][0] - best_set[j][0]
                    dy = best_set[i][1] - best_set[j][1]
                    s.add(dx*dx + dy*dy)
                bd_from.append(s)
            local_limit = min(deadline - 10, time.time() + 20)
            li_selected, li_dist = local_improve(best_set, bd_from, local_limit)
            if len(li_selected) > best_size:
                best_size = len(li_selected)
                best_set = list(li_selected)

    return best_set
