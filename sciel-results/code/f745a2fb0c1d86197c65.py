            s = 0.0
            for t in range(rank):
                s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
            # s should be 1
            diff = abs(s - 1.0)
