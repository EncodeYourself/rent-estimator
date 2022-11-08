# rent-estimator
Description:

A simple yet unique project for my portfolio. The goal is to create a model capable of assessing the rent level for apartments in Saint-Petersburg and estimate a rent range for the Object of Interest. 

The method:

To parse avito.ru, create a dataset from the extracted items fit it into a simple linear model.
After a few attempts with different linear models / settings, the residual analysis showed a clear pattern, so the data cannot be separated linearly. 
Had to abandon the linear model idea and used XGBoost. 

The result:

The result, to put it frankly, is quite… suboptimal. The dataset is quite noisy due to the subjectivity of the prices, maybe better features could help. Anyway, it is enough to access my skills. 
