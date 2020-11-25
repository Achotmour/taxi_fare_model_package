# imports
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import haversine_vectorized
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """set pipeline"""
        # create distance pipeline
        dist_pipe = make_pipeline(DistanceTransformer(), StandardScaler())
        # create distance pipeline
        time_pipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))
        
        # create preprocessing pipeline
        prepro_pipe = ColumnTransformer([('distance', dist_pipe, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']),('time', time_pipe, ['pickup_datetime'])]) 
        # display preprocessing pipeline
        prepro_pipe
        # Add the model of your choice to the pipeline
        from sklearn.linear_model import SGDRegressor
        self.pipeline = Pipeline([('preprocessing', prepro_pipe), ('sgd', SGDRegressor())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        return round(rmse, 2)


if __name__ == "__main__":
    df = clean_data(get_data())
    X = df.drop('fare_amount', axis =1)
    y = df.fare_amount
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3) 
    trainer = Trainer(Xtrain, ytrain)
    trainer.run()
    print(trainer.evaluate(Xtest, ytest))


