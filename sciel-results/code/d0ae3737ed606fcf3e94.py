import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    def evaluate(result):
        if np.any(result < 0) or np.any(result >= n_facilities):
            return float('inf')
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)

        order = np.argsort(-demands)

        for i in order:
            d = demands[i]
            feasible = load + d <= capacities

            if not np.any(feasible):
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue

            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            marginal_costs[~feasible] = np.inf

            if randomize:
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(5, n_facilities)]
                f = top_k[rng.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)

            result[i] = f
            load[f] += d
            is_open[f] = True

        return result, load, is_open

    def local_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                f_old = result[i]
                d = demands[i]

                best_delta = 1e-9
                best_f_new = -1

                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    if load[f_new] + d > capacities[f_new]:
                        continue

                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]

                    delta_open = 0.0
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0 and not is_open[f_new]:
                        delta_open += open_cost[f_new]

                    total_delta = delta_assign + delta_open

                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new

                if best_f_new != -1:
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                    load[best_f_new] += d
                    if load[best_f_new] == d:
                        is_open[best_f_new] = True
                    result[i] = best_f_new
                    improved = True

                    if time.time() > deadline:
                        return result, load, is_open

        return result, load, is_open

    def swap_search(result, load, is_open):
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                for j in range(i + 1, n_customers):
                    f_i = result[i]
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    d_i = demands[i]
                    d_j = demands[j]

                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue

                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])

                    delta_open = 0.0
                    if load[f_i] - d_i + d_j == 0 and load[f_i] - d_i != 0:
                        delta_open -= open_cost[f_i]
                    if load[f_j] - d_j + d_i == 0 and load[f_j] - d_j != 0:
                        delta_open -= open_cost[f_j]
                    if load[f_i] - d_i + d_j != 0 and load[f_i] - d_i == 0:
                        delta_open += open_cost[f_i]
                    if load[f_j] - d_j + d_i != 0 and load[f_j] - d_j == 0:
                        delta_open += open_cost[f_j]

                    total_delta = delta_assign + delta_open

                    if total_delta < -1e-9:
                        load[f_i] -= d_i
                        load[f_j] -= d_j
                        if load[f_i] == 0:
                            is_open[f_i] = False
                        if load[f_j] == 0:
                            is_open[f_j] = False

                        load[f_i] += d_j
                        load[f_j] += d_i
                        if load[f_i] > 0 and not is_open[f_i]:
                            is_open[f_i] = True
                        if load[f_j] > 0 and not is_open[f_j]:
                            is_open[f_j] = True

                        result[i] = f_j
                        result[j] = f_i
                        improved = True

                        if time.time() > deadline:
                            return result, load, is_open

        return result, load, is_open

    best_result = None
    best_score = float('inf')

    # Deterministic greedy + local search + swap search
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    res, load, is_open = swap_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()

    # Randomized restarts
    for k in range(60):
        if time.time() > deadline - 15:
            break

        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        res, load, is_open = swap_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()

    return best_result
