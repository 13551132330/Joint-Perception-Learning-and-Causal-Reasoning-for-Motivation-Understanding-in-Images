import tensorflow as tf
import vgg16

SCNUM_CLASSES =100

ACNUM_CLASSES =100

MCNUM_CLASSES =256

INPUT_SIZE =4800

IMAGE_SIZE =224

FEATURE_SIZE = 4096

CHANNELS = 3

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def get_feature(_F, _dropout,  batch_size,   _weights,   _biases):
  # Output: class prediction
  feature = tf.matmul(_F, _weights['w1']) + _biases['b1']
  feature = tf.nn.relu(feature, name='fc1') 
  feature = tf.nn.dropout(feature, _dropout)
  feature = tf.matmul(feature, _weights['out']) + _biases['out'] #NUM_CLASSES 
  feature = tf.nn.relu(feature, name='fc2') # Relu activation
  feature = tf.nn.dropout(feature, _dropout)
  return feature

def get_predict(_F, _dropout,  batch_size,   _weights,   _biases):
  
  feature = tf.matmul(_F, _weights['w1']) + _biases['b1']
  feature = tf.nn.relu(feature, name='fc1') 
  feature = tf.nn.dropout(feature, _dropout)
  feature = tf.matmul(feature, _weights['out']) + _biases['out'] #NUM_CLASSES 
  return feature

def sc_model(_X, _dropout,  batch_size,   _weights,   _biases):
  #get Us
  sc = tf.matmul(_X, _weights['w1']) + _biases['b1']
  sc = tf.nn.relu(sc, name='fc1')
  sc = tf.nn.dropout(sc, _dropout)
  sc = tf.matmul(sc, _weights['w2']) + _biases['b2']
  sc = tf.nn.relu(sc, name='fc2')
  sc = tf.nn.dropout(sc, _dropout)
  sc = tf.matmul(sc, _weights['out']) + _biases['out'] #NUM_CLASSES 
  # sigmoid (Us)
  sc = tf.nn.sigmoid(sc, name='sigmoid') # sigmoid
  
  return sc

def ac_model(_Z, sc ,  _dropout,  batch_size,   _weights,   _biases):
  #get Ua
  ac = tf.matmul(_Z, _weights['w1']) + _biases['b1']
  ac = tf.nn.relu(ac, name='fc1')
  ac = tf.nn.dropout(ac, _dropout)
  ac = tf.matmul(ac, _weights['w2']) + _biases['b2']
  ac = tf.nn.relu(ac, name='fc2')
  ac = tf.nn.dropout(ac, _dropout)
  ac = tf.matmul(ac, _weights['out']) + _biases['out'] #NUM_CLASSES 
  ac = tf.nn.relu(ac, name='out') # Relu activation
  
  # W_Ua * Ua
  ac = tf.transpose(ac, perm=[1,0]) 
  ac = tf.matmul(_weights['W_Ua'], ac)
  ac = tf.transpose(ac, perm=[1,0])
  # W_alpha * Ys
  
  if sc.shape==[batch_size]:
      sc = tf.one_hot(sc,100,on_value=1,off_value=None,axis=1)
      sc = tf.cast(sc, tf.float32)
  
  sc = tf.transpose(sc, perm=[1,0]) 
  sc = tf.matmul(_weights['W_alpha'], sc)
  sc = tf.transpose(sc, perm=[1,0])

  # sigmoid (W_alpha * Ys + W_Ua * Ua)
  ac = tf.add_n([ac, sc])
  ac = tf.nn.sigmoid(ac, name='sigmoid') # sigmoid
  
  return ac

def mc_model(_Y, sc , ac, _dropout,  batch_size,   _weights,   _biases ):
  #get Ua  
  mc = tf.matmul(_Y, _weights['w1']) + _biases['b1']
  mc = tf.nn.relu(mc, name='fc1')
  mc = tf.nn.dropout(mc, _dropout)
  mc = tf.matmul(mc, _weights['w2']) + _biases['b2']
  mc = tf.nn.relu(mc, name='fc2')
  mc = tf.nn.dropout(mc, _dropout)
  mc = tf.matmul(mc, _weights['out']) + _biases['out'] #NUM_CLASSES 
  mc = tf.nn.relu(mc, name='out') # Relu activation

  # W_Um * Um
  mc = tf.transpose(mc, perm=[1,0]) 
  mc = tf.matmul(_weights['W_Um'], mc)
  mc = tf.transpose(mc, perm=[1,0])

  # W_beta * Ya + W_gama * Ys 
  if sc.shape==[batch_size]:
      sc = tf.one_hot(sc,100,on_value=1,off_value=None,axis=1)
      sc = tf.cast(sc, tf.float32)
      ac = tf.one_hot(ac,100,on_value=1,off_value=None,axis=1)
      ac = tf.cast(ac, tf.float32)
  ac = tf.transpose(ac, perm=[1,0]) 
  sc = tf.transpose(sc, perm=[1,0]) 
  ac_sc = tf.add_n([tf.matmul(_weights['W_beta'], ac),tf.matmul(_weights['W_gama'], sc)])
  ac_sc = tf.transpose(ac_sc, perm=[1,0])
    
  # sigmoid (W_beta * Ya + W_gama * Ys + W_Um * Um)
  mc = tf.add_n([mc,ac_sc])
  mc = tf.nn.sigmoid(mc, name='sigmoid') # sigmoid

  return mc
def sc_model2(_X, _dropout,  batch_size,   _weights,   _biases  ,globle_step):
  #get Us
  if _dropout==0.5:
    tst=False
    tst=tf.cast(tst, tf.bool)
  else:
    tst=True
    tst=tf.cast(tst, tf.bool)
  sc = tf.matmul(_X, _weights['w2'])
  sc, update_ema1 = batchnorm(sc, tst, globle_step, _biases['b2'])
  sc = tf.nn.relu(sc, name='fc2')
  sc = tf.nn.dropout(sc, _dropout)
  sc = tf.matmul(sc, _weights['out']) + _biases['out'] #NUM_CLASSES 
  # sigmoid (Us)
  sc = tf.nn.sigmoid(sc, name='sigmoid') # sigmoid
  
  return sc,update_ema1

def ac_model2(_Z, sc ,  _dropout,  batch_size,   _weights,   _biases ,globle_step):
  #get Ua
  if _dropout==0.5:
    tst=False
    tst=tf.cast(tst, tf.bool)
  else:
    tst=True
    tst=tf.cast(tst, tf.bool)
  ac = tf.matmul(_Z, _weights['w2'])
  ac, update_ema1 = batchnorm(ac, tst, globle_step, _biases['b2'])
  ac = tf.nn.relu(ac, name='fc2')
  ac = tf.nn.dropout(ac, _dropout)
  ac = tf.matmul(ac, _weights['out'])#NUM_CLASSES 
  ac, update_ema2 = batchnorm(ac, tst, globle_step, _biases['out'])
  ac = tf.nn.relu(ac, name='out') # Relu activation
  # W_Ua * Ua
  ac = tf.transpose(ac, perm=[1,0]) 
  ac = tf.matmul(_weights['W_Ua'], ac)
  ac = tf.transpose(ac, perm=[1,0])
  # W_alpha * Ys
  if sc.shape==[batch_size]:
      sc = tf.one_hot(sc,100,on_value=1,off_value=None,axis=1)
      sc = tf.cast(sc, tf.float32)
  sc = tf.transpose(sc, perm=[1,0]) 
  sc = tf.matmul(_weights['W_alpha'], sc)
  sc = tf.transpose(sc, perm=[1,0])

  # sigmoid (W_alpha * Ys + W_Ua * Ua)
  ac = tf.add_n([ac, sc])
  ac = tf.nn.sigmoid(ac, name='sigmoid') # sigmoid
  
  return ac,update_ema1,update_ema2

def mc_model2(_Y, sc , ac, _dropout,  batch_size,   _weights,   _biases ,globle_step):
  #get Ua
  if _dropout==0.5:
    tst=False
    tst=tf.cast(tst, tf.bool)
  else:
    tst=True
    tst=tf.cast(tst, tf.bool)
  mc = tf.matmul(_Y, _weights['w2'])
  mc, update_ema1 = batchnorm(mc, tst, globle_step, _biases['b2'])
  mc = tf.nn.relu(mc, name='fc2')
  mc = tf.nn.dropout(mc, _dropout)
  mc = tf.matmul(mc, _weights['out'])  #NUM_CLASSES 
  mc, update_ema2 = batchnorm(mc, tst, globle_step, _biases['out'])
  mc = tf.nn.relu(mc, name='out') # Relu activation

  # W_Um * Um
  mc = tf.transpose(mc, perm=[1,0]) 
  mc = tf.matmul(_weights['W_Um'], mc)
  mc = tf.transpose(mc, perm=[1,0])

  # W_beta * Ya + W_gama * Ys 
  if sc.shape==[batch_size]:
      sc = tf.one_hot(sc,100,on_value=1,off_value=None,axis=1)
      sc = tf.cast(sc, tf.float32)
  if ac.shape==[batch_size]:
      ac = tf.one_hot(ac,100,on_value=1,off_value=None,axis=1)
      ac = tf.cast(ac, tf.float32)
  ac = tf.transpose(ac, perm=[1,0]) 
  sc = tf.transpose(sc, perm=[1,0]) 
  ac_sc = tf.add_n([tf.matmul(_weights['W_beta'], ac),tf.matmul(_weights['W_gama'], sc)])
  ac_sc = tf.transpose(ac_sc, perm=[1,0])
    
  # sigmoid (W_beta * Ya + W_gama * Ys + W_Um * Um)
  mc = tf.add_n([mc,ac_sc])
  mc = tf.nn.sigmoid(mc, name='sigmoid') # sigmoid

  return mc,update_ema1,update_ema2
