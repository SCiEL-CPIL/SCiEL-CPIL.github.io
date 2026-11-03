    def local_search(p, current_cost):
        """Best-improvement 2-opt local search with restart after each move."""
        p = p.copy()
        while True:
            if time.time() - t_start > deadline:
                break
            moved = False
            for i in range(n):
                if time.time() - t_start > deadline:
                    break
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                best_delta = 0
                best_j = -1
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    delta = 2 * (sum_all - term_i - term_j)
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                if best_j != -1:
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    current_cost += best_delta
                    moved = True
                    break  # restart scan from i=0
            if not moved:
                break
        return p, current_cost
