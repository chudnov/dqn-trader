from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, LSTM
from keras.optimizers import Adam
from keras import backend as K

def mlp(n_obs, n_action, activation, loss, learning_rate, dqn_type):
  """ A multi-layer perceptron """
  model = Sequential()
  model.add(Dense(24, input_dim=n_obs, activation=activation))
  model.add(Dense(48, activation=activation))
  model.add(Dense(96, activation=activation))
  #model.add(LSTM(96, activation=activation))
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

def mlp2(window_size, n_market_features, n_private_vars, n_action, activation='relu', loss='mse', learning_rate = 0.01, dqn_type = 0):
  """ A multi-layer perceptron """
  indicators = Input(shape=(32,))
  private_vars = Input(shape=(128,))
  # the first branch operates on the first input
  x = Dense(8, activation="relu")(inputA)
  x = Dense(4, activation="relu")(x)
  x = Model(inputs=inputA, outputs=x)
  # the second branch opreates on the second input
  y = Dense(64, activation="relu")(inputB)
  y = Dense(32, activation="relu")(y)
  y = Dense(4, activation="relu")(y)
  y = Model(inputs=inputB, outputs=y)
  # combine the output of the two branches
  combined = concatenate([x.output, y.output])
  # apply a FC layer and then a regression prediction on the
  # combined outputs
  z = Dense(2, activation="relu")(combined)
  z = Dense(1, activation="linear")(z)
  # our model will accept the inputs of the two branches and
  # then output a single value
  model = Model(inputs=[x.input, y.input], outputs=z)

  model.compile(loss=loss, optimizer=Adam(lr=learning_rate))
  return model
