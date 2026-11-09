        for c in range(color_count - 1, -1, -1):
            if current_size + c <= best_size:
                break
            for v in buckets[c]:
                current_clique.append(v)
                new_mask = mask & adj[v]
                expand(new_mask, current_clique, current_size + 1)
                current_clique.pop()
                mask &= ~(1 << v)
