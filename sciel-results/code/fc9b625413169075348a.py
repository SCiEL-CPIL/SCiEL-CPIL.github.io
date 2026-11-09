smooths_7 = generate_smooth_numbers(N, [2, 3, 5, 7])
for a in smooths_7:
    b = N - a
    if b > 0 and b in smooths_7:
        return [a, b]
