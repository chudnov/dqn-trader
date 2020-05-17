from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=24,
        activation='relu', loss='mse', learning_rate = 0.01):
  """ A multi-layer perceptron """
  model = Sequential()
  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  for i in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer * (2*i + 2), activation=activation))
  model.add(Dense(n_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam(lr=learning_rate))
  print(model.summary())
  return model
