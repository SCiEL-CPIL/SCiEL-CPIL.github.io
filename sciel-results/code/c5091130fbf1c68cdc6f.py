    def add_fast(x, used_sums, current_set):
        new_sums = used_sums | (1 << (2*x))
        for y in current_set:
            new_sums |= (1 << (x+y))
        return new_sums
    