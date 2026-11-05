    def get_matchings(part_size):
        if part_size % 2 != 0:
            return []
        if part_size == 0:
            return [()]
        if part_size == 2:
            return [((0,1),)]
        if part_size == 4:
            return [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]
        if part_size == 6:
            matchings = []
            for p1 in range(1, 6):
                remaining = [i for i in range(6) if i != 0 and i != p1]
                for m in get_matchings(4):
                    new_m = [((0, p1),)]
                    for (a, b) in m:
                        new_m.append((remaining[a], remaining[b]))
                    matchings.append(tuple(sorted(new_m)))
            return matchings
        return []
