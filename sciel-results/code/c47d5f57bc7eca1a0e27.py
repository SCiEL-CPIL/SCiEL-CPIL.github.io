            clique.append(v)
            remaining_mask[v] = False
            remaining_mask &= valid[v]
