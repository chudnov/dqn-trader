from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, LSTM, Dropout
from keras.optimizers import Adam
from keras import backend as K

def mlp(obs_shape, n_action, activation, loss, learning_rate, dqn_type):
  model = Sequential()
  model.add(LSTM(128, input_shape=obs_shape, return_sequences=True, activation=activation))
  model.add(Dropout(0.2))
  model.add(LSTM(128, activation=activation))
  model.add(Dropout(0.2))
  model.add(Dense(n_action, activation='linear'))
  model.compile(loss=loss, optimizer=Adam(lr=learning_rate)) 
  return model

'''
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
'''


