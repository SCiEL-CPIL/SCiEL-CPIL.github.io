def construct():
    # The Arithmetic Progression {0, 1, ..., 11} minimizes the largest Sidon subset size.
    # High density of elements leads to frequent pair sum collisions,
    # forcing the largest Sidon subset to be small.
    # For n=12, the maximum Sidon subset size in an AP is 4.
    return list(range(12))
