            # Pruning: Connectivity
            comps = _count_components(current_adj, n)
            if comps - 1 > remaining:
                continue # Actually, this should be a check before the loop or inside?
                # If we can't connect, we can't reach a valid state.
                # But we are in a loop over candidates. We should check this at the start of the recursion.
