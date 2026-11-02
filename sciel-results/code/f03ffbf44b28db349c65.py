                u = in_indices[rng.integers(len(in_indices))]
                v = out_indices[rng.integers(len(out_indices))]
                if u == v: continue
                if count_conflicts[v] < count_conflicts[u]:
                    # Perform swap
