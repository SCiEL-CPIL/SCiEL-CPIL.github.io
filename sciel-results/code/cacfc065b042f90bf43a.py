    def get_walks_dir(max_len):
        walks = []
        for start in range(10):
            # Stack items: (curr, prev, l, coeff_list)
            # coeff_list is a list of 15 integers
            initial_coeff = [0] * num_edges
            stack = [(start, -1, 0, initial_coeff)]
            while stack:
                curr, prev, l, coeff = stack.pop()
                if l == max_len:
                    if curr == start:
                        # Convert to tuple of non-zeros for storage
                        nz = [(i, c) for i, c in enumerate(coeff) if c != 0]
                        if nz: # Should always be non-empty for closed walks
                            walks.append(nz)
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_index[(curr, nxt)]
                    if (curr, nxt) == edges_p[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    # Create a new coeff list (copy)
                    new_coeff = coeff[:]
                    new_coeff[e_idx] += sign
                    stack.append((nxt, curr, l + 1, new_coeff))
        return walks
