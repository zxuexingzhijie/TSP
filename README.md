# TSP
遗传算法解决TSP问题

### 代码注意事项

1. 每次的图像结果和最短路径结果都会保存在本地,`output_Python`即为保存路径,如果要自己跑一遍,请按需求自行修改.该代码位于开头import部分.

```Python
output_path = "E:/结果/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
```

2. 这些参数量设置的比较大,所以跑起来的时间比较久(大概40分钟),参数设置位于120行和134行

```Python
num_individuals = 5000      #每一代的个体
num_generations = 50000     #要产生多少代    
mutation_rate = 0.1         #变异率
improvement_threshold = 0.01  # 显著改进的阈值
max_no_improve_generations = 5000  # 连续无显著改进的最大代数
generations_without_improvement = 0  # 计数无显著改进的代数
```

3. 确保自己有`SimHei`字体,否则matplotlib绘图时可能会显示乱码.
4. 因为代码大部分是ai写的,我感觉可读性很强,注释就没太需要了.
5. 代码整体就是遗传算法的实现,运行10次，每次运行都初始化种群，然后通过选择、交叉、变异生成新的种群。记录每次运行的最佳路径和最短距离。如果在一定代数内没有显著改进，则提前终止算法。(`具体参数参照第二个注意事项`)
