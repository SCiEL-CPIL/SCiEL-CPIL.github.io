            while curr_node != T:
                next_node = -1
                for e in adj[curr_node]:
                    v, cap, c, rev, orig_cap = e
                    if orig_cap == 1 and cap == 0:
                        next_node = v
                        break
                if next_node == -1:
                    break
                curr_node = next_node
                if curr_node == T:
                    break
                vertex = curr_node // 2
                if curr_node % 2 == 0: # v_in
                    path.append(vertex)
