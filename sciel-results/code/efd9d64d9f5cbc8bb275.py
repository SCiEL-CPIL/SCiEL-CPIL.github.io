    def count_bad_subgraphs(adj_masks, n, k=6):
        count = 0
        for S in itertools.combinations(range(n), k):
            mask = 0
            for v in S:
                mask |= (1 << v)
            first_deg = -1
            regular = True
            for v in S:
                d = (adj_masks[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    regular = False
                    break
            if regular:
                count += 1
        return count
