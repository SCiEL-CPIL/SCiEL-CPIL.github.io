    changed = True
    while changed:
        changed = False
        for v in list(solution):
            # check redundancy
            if redundant:
                solution.remove(v)
                changed = True
