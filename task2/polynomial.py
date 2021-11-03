def calculate_polynomial(inp, multipliers, description):
    res = 0
    for row in range(0, len(description)):
        coef = 1
        for col in range(len(description[row]) - 1, -1, -1):
            if description[row][col] == 0:
                continue
            coef *= inp[int(description[row][col] - 1)]
        res += coef * multipliers[row]
    return res
