# adult_income
## 文件结构
- data_processing.py
对原始数据进行处理, 将各项属性映射到0~N, 并存为.csv
- svm.py
主函数, 使用adult.data.csv的数据训练0支撑向量机, 对test数据集进行预测
- grid_search.py
参数调优函数, 寻找rbf核函数的最优C, gamma参数
- grid_search_local.py, grid_search_contour.py
同上进行参数调优, 并绘制等值线图寻找趋势 