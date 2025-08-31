import numpy as np
import pandas as pd

# Machine Learning
import keras
import ml_edu.experiment
import ml_edu.results

# Visualization
import plotly.express as px
chicago_taxi_dataset = pd.read_csv('https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv')
training_df = chicago_taxi_dataset.loc[:,('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

def create_model(settings: ml_edu.experiment.ExperimentSettings, metrics: list[keras.metrics.Metric]) -> keras.Model:
    """
    Create and compile simple linear regression model.
    """
    
    # Describe the topography of the model.
    # The topography of a simple lienar regression model is a single node in sigle layer.
    inputs = {name: keras.Input(shape=(1,), name = name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model topography into code that model can efficiently execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate), loss='mean_squared_error', metrics=metrics)

    print(model.summary())

    return model

def train_model(experiment_name:str, model:keras.Model, dataset: pd.DataFrame, lable_name:str, settings: ml_edu.experiment.ExperimentSettings) ->ml_edu.experiment.Experiment:
    """
    Train the model by fedding the data
    """

    # Feed the model the features and the lable
    # The model will train on specified number of epochs
    features = {name: dataset[name].values for name in settings.input_features}
    lables = dataset[lable_name].values
    history = model.fit(x = features, y = lables, batch_size=settings.batch_size,epochs=settings.number_epochs)

    return ml_edu.experiment.Experiment(name=experiment_name, settings=settings,model=model, epochs=history.epoch, metrics_history=pd.DataFrame(history.history))


settings_1 = ml_edu.experiment.ExperimentSettings(learning_rate=0.001, number_epochs=20, batch_size=50, input_features=['TRIP_MILES'])
metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_1 = create_model(settings_1, metrics)
# experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

# ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
# ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')

