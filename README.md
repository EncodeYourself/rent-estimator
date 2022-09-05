# rent-estimator
Description:

A simple yet unique project for my portfolio. The goal is to create a model capable of assessing the rent level for apartments in Saint-Petersburg and get a rent range for the Object of Interest. 

The method:

To parse avito.ru, create a dataset from the extracted items fit it into a simple linear model.
After a few attempts with different linear models / settings, the residual analysis showed a clear pattern, so the data cannot be separated linearly. 
Had to abandon the linear model idea and used XGBoost. 

The result:

The result, to put it frankly, is quite… suboptimal. Was it my lack of experience, noisiness of the dataset or oversimplified features? I think all of them. Yet I guess the project is still good enough for you to assess my capabilities.  
