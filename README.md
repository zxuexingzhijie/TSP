# TSP
遗传算法解决TSP问题



### 总体感悟

1. 我感觉遗传算法相当不靠谱,而且算法实现复杂,比较吃运气,不光会出现收敛结果不一致的情况,时间消耗也很大(看运气),效果也一般,下次不会用这个破玩意了.

2. 因为我对java语言较为熟悉,此算法的框架大部分是ai实现的,自己再添加功能并修改,我感觉自己手撕不太现实,但其实大部分代码花点时间就能看的八九不离十,毕竟只用了一个算法,我也考虑再揉进去个模拟退火算法,效果会好点,但我能力有限,就作罢了.



### 代码注意事项

1. 每次的图像结果和最短路径结果都会保存在本地,`output_Python`即为保存路径,如果要自己跑一遍,请按需求自行修改.该代码位于开头import部分.

```Python
output_path = "E:/结果/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
```

2. 这些参数量比较大,所以跑起来的时间比较久(大概1个半h),代码位置位于120行和134行

```Python
num_individuals = 5000      #每一代的个体
num_generations = 50000     #要产生多少代    
mutation_rate = 0.1         #变异率
improvement_threshold = 0.01  # 显著改进的阈值
max_no_improve_generations = 5000  # 连续无显著改进的最大代数
generations_without_improvement = 0  # 计数无显著改进的代数
```

3. 确保自己有`SimHei`字体,否则matplotlib绘图时可能会显示乱码.
5. 代码整体就是遗传算法的实现,运行10次，每次运行都初始化种群，然后通过选择、交叉、变异生成新的种群。记录每次运行的最佳路径和最短距离。如果在一定代数内没有显著改进，则提前终止算法。(`具体参数参照第二个注意事项`)

6.部分图例如下
![image](https://github.com/user-attachments/assets/eebc99c2-628d-438c-925c-7717edbde05e)
![image](https://github.com/user-attachments/assets/e65f2a6d-bea0-47c2-a687-63f03290f3da)
![image](https://github.com/user-attachments/assets/abcb3720-b4d1-4767-bef8-859383fc1d14)
![image](https://github.com/user-attachments/assets/2e823b38-bf38-45f0-bb40-aa4b50244568)



