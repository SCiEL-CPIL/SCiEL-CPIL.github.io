    # Replace the entire block from "    # Branch and bound for maximum independent set" to the end of the restart loop
    # And fix the typo in greedy_init
    
    def greedy_min_degree(conflict, N_S, deadline):
        """
        Greedy independent set using minimum degree heuristic on the induced subgraph.
        """
        x = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        # We need to efficiently pick the node with min degree in the available set
        # For N=330, recalculating degrees is O(N^2) per step, total O(N^3) ~ 330^3 = 36M ops.
        # This is fast enough in Python if optimized or done in NumPy?
        # NumPy sum of rows is fast.
        
        for _ in range(N_S):
            if time.time() > deadline - 5:
                break
            if not np.any(available):
                break
            
            # Calculate degrees in the available subgraph
            # deg[i] = sum(conflict[i, available])
            degs = np.sum(conflict[:, available], axis=1)
            degs[~available] = np.inf
            
            # Pick node with min degree
            v = int(np.argmin(degs))
            
            if not available[v]:
                break
                
            x[v] = True
            # Remove v and its neighbors from available
            available[v] = False
            neighbors = np.where(conflict[v])[0]
            available[neighbors] = False
            
        return x

    def try_add_and_swap(x, conflict, N_S, deadline, max_swaps=1000):
        """
        Try to improve x by adding nodes or swapping 1 for 2.
        """
        x = x.copy()
        in_set = set(np.where(x)[0].tolist())
        
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            
            # 1. Try to add any available node
            blocked = np.zeros(N_S, dtype=bool)
            for v in in_set:
                blocked |= conflict[v]
            candidates = np.where(~x & ~blocked)[0]
            if len(candidates) > 0:
                # Pick one (random or first)
                v = candidates[0]
                x[v] = True
                in_set.add(v)
                continue
            
            # 2. Try swap: remove v, add u, w
            # This is expensive to check all pairs.
            # Heuristic: Pick a random v in set, remove it, see if we can add 2.
            if len(in_set) == 0:
                break
            v_remove = list(in_set)[np.random.randint(len(in_set))]
            
            # Temporarily remove
            in_set.remove(v_remove)
            x[v_remove] = False
            
            blocked = np.zeros(N_S, dtype=bool)
            for u in in_set:
                blocked |= conflict[u]
            
            candidates = np.where(~x & ~blocked)[0]
            if len(candidates) >= 2:
                # Try to add two
                # Check if candidates[0] and candidates[1] are compatible
                c1, c2 = candidates[0], candidates[1]
                if not (conflict[c1, c2]):
                    x[c1] = True
                    x[c2] = True
                    in_set.add(c1)
                    in_set.add(c2)
                    # Net gain +1
                else:
                    # Try next pair? Or just revert
                    # For simplicity, revert
                    pass
            else:
                # Revert
                x[v_remove] = True
                in_set.add(v_remove)
                
        return x
    