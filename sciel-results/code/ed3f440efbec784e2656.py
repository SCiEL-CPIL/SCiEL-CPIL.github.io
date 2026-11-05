def int_to_base3(num, length):
    res = [0] * length
    for i in range(length - 1, -1, -1):
        res[i] = num % 3
        num //= 3
    return res
