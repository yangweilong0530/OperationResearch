import numpy as np  # 导入numpy库，用于处理矩阵和数组
import pandas as pd


def simplex(c, A, b):  # 定义单纯形法函数，输入参数为目标函数系数c，约束条件系数A和约束条件值b
    num_vars = len(c)  # 获取变量的数量
    num_constraints = len(A)  # 获取约束条件的数量
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))  # 初始化单纯形表
    tableau[:-1, :num_vars] = A  # 将约束条件系数填入单纯形表
    tableau[:-1, -1] = b  # 将约束条件值填入单纯形表
    tableau[-1, :num_vars] = -c  # 将目标函数系数填入单纯形表
    tableau[:-1, num_vars:num_vars+num_constraints] = np.eye(num_constraints)  # 在单纯形表中添加单位矩阵，表示初始基变量

    iteration = 0  # 初始化迭代次数
    while True:  # 开始迭代
        print(f'Iteration {iteration}:')  # 打印当前的迭代次数
        df = pd.DataFrame(tableau)
        df.columns = [f'x{i + 1}' for i in range(num_vars)] + [f's{i + 1}' for i in range(num_constraints)] + ['b']
        df.index = [f's{i + 1}' for i in range(num_constraints)] + ['z']
        print(df)
        if np.all(tableau[-1, :-1] >= 0):  # 如果目标函数的系数都大于等于0，说明找到了最优解，跳出循环
            break
        pivot_column = np.argmin(tableau[-1, :-1])  # 找到目标函数系数最小的列，作为主元列
        if np.all(tableau[:-1, pivot_column] <= 0):  # 如果主元列的所有元素都小于等于0，说明问题是无界的，抛出错误
            raise ValueError('Problem is unbounded.')
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_column]  # 计算每一行最后一列（b值）与主元列的比值
        valid_ratios = ratios >= 0  # 只有比值大于等于0的行才是有效的
        if not np.any(valid_ratios):  # 如果没有有效的行，说明问题是无解的，抛出错误
            raise ValueError('Problem is infeasible.')
        pivot_row = np.argmin(ratios[valid_ratios])  # 在有效的行中找到比值最小的行，作为主元行
        pivot_element = tableau[pivot_row, pivot_column]  # 获取主元
        tableau[pivot_row] /= pivot_element  # 将主元行除以主元，使得主元变为1
        for i in range(len(tableau)):  # 对于单纯形表中的每一行
            if i != pivot_row:  # 如果不是主元行
                tableau[i] -= tableau[pivot_row] * tableau[i, pivot_column]  # 则将该行减去主元行乘以该行主元列的值，使得主元列除了主元外其他元素都为0
        iteration += 1

    solution = np.zeros(num_vars)  # 初始化解向量
    for i in range(num_vars):  # 对于每一个变量
        col = tableau[:, i]
        if np.sum(col == 0) == len(col) - 1 and np.sum(col == 1) == 1:
            solution[i] = tableau[col == 1, -1]

    return solution

# 测试代码
c = np.array([-3, -2], dtype=float)
A = np.array([[2, 1], [1, 3]], dtype=float)
b = np.array([18, 15], dtype=float)

try:
    solution = simplex(c, A, b)
    print('Solution:', solution)
except ValueError as e:
    print(e)
