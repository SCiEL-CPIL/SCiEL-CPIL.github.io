def solve(a, b):
    n = len(a)
    deadline = time.time() + 110
    is_sym = np.array_equal(b, b.T)
    
    # Define fast swap delta
    def _swap_delta_fast(a, b, p, i, j, is_sym):
        pi = p[i]
        pj = p[j]
        if is_sym:
            diff = b[:, pj] - b[:, pi]
            diff_p = diff[p]
            ai = a[:, i] + a[i, :]
            aj = a[:, j] + a[j, :]
            delta = ai * diff_p + aj * (-diff_p)
            delta[i] = 0
            delta[j] = 0
            direct = (a[i, j] + a[j, i]) * (b[pj, pi] - b[pi, pj]) # b[pj, pi] == b[pi, pj] so 0?
            # If symmetric, b[pj, pi] == b[pi, pj], so direct is 0.
            # Wait, the cost term is a[i,j] b[p[i], p[j]].
            # Swap i,j.
            # Old: a[i,j] b[pi, pj] + a[j,i] b[pj, pi].
            # New: a[i,j] b[pj, pi] + a[j,i] b[pi, pj].
            # Diff: a[i,j] (b[pj, pi] - b[pi, pj]) + a[j,i] (b[pi, pj] - b[pj, pi]).
            # If b symmetric, b[pj, pi] == b[pi, pj], so diff is 0.
            return int(np.sum(delta))
        else:
             # Original logic
             bpj = b[:, pj][p]
             bpi = b[:, pi][p]
             brj = b[pj, :][p]
             bri = b[pi, :][p]
             delta = (a[:, i] * (bpj - bpi) +
                      a[:, j] * (bpi - bpj) +
                      a[i, :] * (brj - bri) +
                      a[j, :] * (bri - brj))
             delta[i] = 0
             delta[j] = 0
             direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
             return int(np.sum(delta) + direct)

    # ... rest of solve
