import numpy as np

def simplex(c, A, b):
    # 初始化
    num_vars = len(c)
    num_constraints = len(A)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[:-1, :num_vars] = A
    tableau[:-1, -1] = b
    tableau[-1, :num_vars] = -c
    tableau[:-1, num_vars:num_vars+num_constraints] = np.eye(num_constraints)

    # 主循环
    iteration = 0
    while True:
        print(f'Iteration {iteration}:')
        print(tableau)
        if np.all(tableau[-1, :-1] >= 0):
            break
        pivot_column = np.argmin(tableau[-1, :-1])
        if np.all(tableau[:-1, pivot_column] <= 0):
            raise ValueError('Problem is unbounded.')
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_column]
        valid_ratios = ratios >= 0
        if not np.any(valid_ratios):
            raise ValueError('Problem is infeasible.')
        pivot_row = np.argmin(ratios[valid_ratios])
        pivot_element = tableau[pivot_row, pivot_column]
        tableau[pivot_row] /= pivot_element
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[pivot_row] * tableau[i, pivot_column]
        iteration += 1

    # 提取解决方案
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:, i]
        if np.sum(col == 0) == len(col) - 1 and np.sum(col == 1) == 1:
            solution[i] = tableau[col == 1, -1]

    return solution

# 测试
c = np.array([-3, -2], dtype=float)
A = np.array([[2, 1], [1, 3]], dtype=float)
b = np.array([18, 15], dtype=float)

try:
    solution = simplex(c, A, b)
    print('Solution:', solution)
except ValueError as e:
    print(e)
