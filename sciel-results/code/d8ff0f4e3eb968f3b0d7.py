    def decode_code(code, num_weights):
        coeffs = [0] * num_weights
        c = int(code)
        for i in range(num_weights - 1, -1, -1):
            digit = c % 3
            c //= 3
            # 0 -> -1, 1 -> 0, 2 -> 1
            coeffs[i] = digit - 1
        return coeffs
