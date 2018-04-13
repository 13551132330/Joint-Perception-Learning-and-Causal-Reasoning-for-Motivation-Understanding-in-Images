import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
from get_data_step2 import input_data
import tensorflow as tf
import model
import math
import vgg16
import numpy as np

flags = tf.app.flags
flags.DEFINE_integer('max_steps',421, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size',6 , 'Batch size.')
FLAGS = flags.FLAGS
gpu_num = 1
pre_model_save_dir = './models/baseline_mc_models'
def placeholder_inputs(batch_size):
	#bulit placeholder_inputs
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         model.IMAGE_SIZE,
                                                         model.IMAGE_SIZE,
                                                         model.CHANNELS))
	mc_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	keep_pro = tf.placeholder(tf.float32)

	return images_placeholder,mc_labels_placeholder,keep_pro


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
    cross_entropy_mean = tf.reduce_mean(
                      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit)
                      )
    tf.summary.scalar(
                      name_scope + '_cross_entropy',
                      cross_entropy_mean
                      )
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay_loss 
    tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
    return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def topk_acc(logit, labels , k):
    list=tf.nn.in_top_k(logit,labels,k)
    in_top1 = tf.to_float(list)
    num_correct = tf.reduce_sum(in_top1)
    return  num_correct/ 6

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var	

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  print ('loading and init vgg16.........')
  vgg=vgg16.Vgg16()
  with tf.Graph().as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder,mc_labels_placeholder,keep_pro =              placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    tower_grads3 = []
    mc_logits = []

    learning_rate=tf.train.exponential_decay(1e-4,global_step,decay_steps=FLAGS.max_steps/50,decay_rate=0.99,staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    opt_mc = tf.train.AdamOptimizer(learning_rate)
    with tf.variable_scope('var_name') as var_scope:
      
      sc_fea_weights = {
              'w1': _variable_with_weight_decay('sc_w1', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('sc_feawout', [2048, 100], 0.005)
              }
      sc_fea_biases = {
              'b1': _variable_with_weight_decay('sc_b1', [2048], 0.000),
              'out': _variable_with_weight_decay('sc_feabout', [100], 0.000),
              }
      ac_fea_weights = {
              'w1': _variable_with_weight_decay('ac_w1', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('ac_feawout', [2048, 100], 0.005)
              }
      ac_fea_biases = {
              'b1': _variable_with_weight_decay('ac_b1', [2048], 0.000),
              'out': _variable_with_weight_decay('ac_feabout', [100], 0.000),
              }
      mc_fea_weights = {
              'w1': _variable_with_weight_decay('mc_w1', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('mc_feawout', [2048, 256], 0.005)
              }
      mc_fea_biases = {
              'b1': _variable_with_weight_decay('mc_b1', [2048], 0.000),
              'out': _variable_with_weight_decay('mc_feabout', [256], 0.000),
              }
    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):

        varlist1 = [ mc_fea_weights.values(),mc_fea_biases.values() ]
        
        vgg.build(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:])
        train_features=vgg.relu7
        
        mc_logit = model.get_predict(
                        train_features,
                        keep_pro,
                        FLAGS.batch_size,
                        mc_fea_weights,
                        mc_fea_biases
                        )
        
        
        loss_name_scope = ('gpud_%d_loss' % gpu_index)
        '''
        regularizer = tf.contrib.layers.l1_regularizer(0.1)
        with tf.variable_scope('var', initializer=tf.random_normal_initializer(), 
        regularizer=regularizer):
            weight = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())
        with tf.variable_scope('var2', initializer=tf.random_normal_initializer(), 
        regularizer=regularizer):
            weight2 = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        '''

        mc_loss = tower_loss(
                        loss_name_scope+'_scene',
                        mc_logit,
                        mc_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                        )
        grads1 = opt_mc.compute_gradients(mc_loss, varlist1)
        tower_grads1.append(grads1)
        mc_logits.append(mc_logit)

    mc_logits = tf.concat(mc_logits,0)
    predictions = tf.nn.top_k(tf.nn.softmax(mc_logits),5)
    #mc_accuracy = tower_acc(mc_logits, mc_labels_placeholder)
    mc_accuracy = topk_acc(tf.nn.softmax(mc_logits), mc_labels_placeholder,5)
    
    grads1 = average_gradients(tower_grads1)
    
    apply_gradient_mc = opt_mc.apply_gradients(grads1, global_step=global_step)

    '''
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    '''
    train_mc = tf.group(apply_gradient_mc)
    
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(mc_fea_weights.values() + mc_fea_biases.values())
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)

    # Create summary writter
    merged = tf.summary.merge_all()
    
  ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)  
  if ckpt and ckpt.model_checkpoint_path:  
    print "loading checkpoint,waiting......"
    saver.restore(sess, ckpt.model_checkpoint_path)
    print "load complete!"
    
  next_strat_pos=0
  predict_labels=[]
  for step in xrange(FLAGS.max_steps):
      
    start_time = time.time()
    print('TEST Data Eval:')
    val_actions,val_images,val_ac_labels,val_sc_labels,val_mc_labels, next_strat_pos, _= input_data(
                        filename='./list/test.list',
                        start_pos=next_strat_pos,
                        batch_size=FLAGS.batch_size * gpu_num,
                        shuffle=False)
    predict,mc_acc,mc_loss_value = sess.run(
                       [predictions,mc_accuracy,mc_loss],
                        feed_dict={
                                  images_placeholder: val_images,
                                  mc_labels_placeholder: val_mc_labels,
                                  keep_pro : 1
                                        })
    print ("mc_accuracy: " + "{:.5f}".format(mc_acc))
    print 'mc_loss= %.2f'% np.mean(mc_loss_value)
    for i in range(FLAGS.batch_size):
        predict_labels.append(predict[1][i])

    duration = time.time() - start_time
    print('Batchnum %d: %.3f sec' % (step+1, duration))
    #print predict_labels
    #print val_mc_labels

  print("get_predict_label_done!")
  return predict_labels

def main(_):
  #for i in range(model.MCNUM_CLASSES):
  #  if not os.path.exist('./test/%d.txt'%i):
  lines = open('./list/test.list','r')
  lines = list(lines)
  datasets = open('./dataset.txt','r')
  datasets = list(datasets)
  cluster_256=np.load('./clusters_256.npz')
  cluster_100=np.load('./clusters_100.npz')
  label_list=run_training()


  sum=0

  class_list=[0]*256
  true_list=[0]*256
  for i in range(len(label_list)):
    line = lines[i].strip('\n').split('\t')
    dirname=line[0]
    line_num = line[2]
    dataset=datasets[int(line_num)].strip('\n').split('\t')
    action=dataset[1]
    scene=dataset[3]
    motivation=dataset[2]
    tmp_ac_label = cluster_100['ac'][int(line_num)]
    tmp_sc_label = cluster_100['sc'][int(line_num)]
    tmp_mc_label = cluster_256['mc'][int(line_num)]
    class_list[int(tmp_sc_label)]+=1
    if tmp_mc_label in label_list[i]:
        true_list[int(tmp_mc_label)]+=1
  for i in range(256):
    if class_list[i]!=0:
        sum+= float(true_list[i])/class_list[i]
  print sum/255
  '''
  sum=0
  for i in range(len(label_list)):
    line = lines[i].strip('\n').split('\t')
    dirname=line[0]
    line_num = line[2]
    dataset=datasets[int(line_num)].strip('\n').split('\t')
    action=dataset[1]
    scene=dataset[3]
    motivation=dataset[2]
    tmp_ac_label = cluster_100['ac'][int(line_num)]
    tmp_sc_label = cluster_100['sc'][int(line_num)]
    tmp_mc_label = cluster_256['mc'][int(line_num)]
    sum += list(label_list[i]).index(int(tmp_mc_label))
  print (float(sum)/(421*6))
  '''
  '''
  for i in range(len(label_list)):
    line = lines[i].strip('\n').split('\t')
    dirname=line[0]
    line_num = line[2]
    dataset=datasets[int(line_num)].strip('\n').split('\t')
    action=dataset[1]
    scene=dataset[3]
    motivation=dataset[2]
    tmp_ac_label = cluster_100['ac'][int(line_num)]
    tmp_sc_label = cluster_100['sc'][int(line_num)]
    tmp_mc_label = cluster_256['mc'][int(line_num)]
    f=open('./test/motivation/%d.txt'%int(tmp_mc_label),'a+')
    f.write(str(int(line_num)+1)+'\t'+dirname+'\t'+motivation+'\t'+'groud_truth:'+str(tmp_mc_label)+'\t'+'predict:'+str(label_list[i])+'\n')
   '''

if __name__ == '__main__':
  tf.app.run()
