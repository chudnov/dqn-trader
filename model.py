from keras.models import Sequential, Model
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K

def mlp(n_obs, n_action, n_hidden_layer=2, n_neuron_per_layer=24,
        activation='relu', loss='mse', learning_rate = 0.01, dqn_type = 0):
  """ A multi-layer perceptron """
  model = Sequential()
  model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
  for i in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer * (2*i + 2), activation=activation))
  model.add(Dense(n_action, activation='linear'))

  if(dqn_type == 2):
      #Duelling DQN addition
      layer = model.layers[-2]
      nb_action = model.output._keras_shape[-1]
      y = Dense(nb_action + 1, activation='linear')(layer.output)
      outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
				 output_shape=(nb_action,))(y)  #  Using the max dueling type
      model = Model(inputs=model.input, outputs=outputlayer)

  model.compile(loss=loss, optimizer=Adam(lr=learning_rate))
  return model
