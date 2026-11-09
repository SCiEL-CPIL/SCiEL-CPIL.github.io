        for subset in pair_subsets[(u,v)]:
            if is_q3(subset, graph_bits):
                # Revert
                graph_bits[u] &= ~(1 << v)
                graph_bits[v] &= ~(1 << u)
                return False
