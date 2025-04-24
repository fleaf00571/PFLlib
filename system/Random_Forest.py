import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# 1. 加载数据
data_file = 'FedEntSmooth_BO.csv'
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file)

try:
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
    else:
        print(f"File {data_path} not found, creating sample data for demonstration")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 100
        n_params = 5
        
        # Create sample parameters and results
        sample_data = {}
        for i in range(n_params):
            sample_data[f'Param_{i+1}'] = np.random.rand(n_samples)
        
        # Create target values with some correlation to params
        values = 0.5 + 2 * sample_data['Param_1'] - 1.5 * sample_data['Param_2'] 
        values += 0.5 * sample_data['Param_3'] + np.random.normal(0, 0.2, n_samples)
        
        sample_data['Value'] = values
        sample_data['State'] = ['COMPLETE'] * n_samples
        
        data = pd.DataFrame(sample_data)
        
    # Filter successful trials
    successful_trials = data[data['State'] == 'COMPLETE']
    print(f"Found {len(successful_trials)} successful trials")

    # 2. 准备特征和目标值
    param_columns = [col for col in successful_trials.columns if col.startswith('Param')]
    print(f"Feature columns: {param_columns}")
    X = successful_trials[param_columns]
    y = successful_trials['Value']

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 4. 定义随机森林的超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # 5. 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )

    # 6. 执行搜索
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)

    # 7. 输出最优参数和得分
    print("最佳参数：", grid_search.best_params_)
    print("交叉验证得分（R2）：", grid_search.best_score_)
    
    # 8. 在测试集上评估
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"测试集得分（R2）：{test_score:.4f}")
    
    # 9. 特征重要性分析
    feature_importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': param_columns,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\n特征重要性：")
    print(importance_df)
    
except Exception as e:
    print(f"Error in Random Forest analysis: {e}")
