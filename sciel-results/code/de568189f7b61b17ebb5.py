            adj.add(pair)
            valid = True
            for subset in candidate_subsets:
                if is_q3(subset, adj):
                    valid = False
                    break
            if not valid:
                adj.discard(pair)
