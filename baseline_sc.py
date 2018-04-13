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
flags.DEFINE_integer('max_steps',20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size',16 , 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
gpu_num = 1
def placeholder_inputs(batch_size):
	#bulit placeholder_inputs
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         model.IMAGE_SIZE,
                                                         model.IMAGE_SIZE,
                                                         model.CHANNELS))
	sc_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	keep_pro = tf.placeholder(tf.float32)

	return images_placeholder,sc_labels_placeholder,keep_pro


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

def tower_loss(name_scope, logit, labels,classic):
    labels=tf.one_hot(labels,classic,on_value=1,off_value=None,axis=1)
    cross_entropy_mean = tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logit)
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
    return  num_correct/ 16

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
    images_placeholder,sc_labels_placeholder,keep_pro =              placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    tower_grads3 = []
    sc_logits = []

    learning_rate=tf.train.exponential_decay(1e-5,global_step,decay_steps=FLAGS.max_steps/50,decay_rate=0.99,staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    opt_sc = tf.train.AdamOptimizer(learning_rate)
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

        varlist1 = [ sc_fea_weights.values(),sc_fea_biases.values() ]
        
        vgg.build(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:])
        train_features=vgg.relu7
        
        sc_logit = model.get_predict(
                        train_features,
                        keep_pro,
                        FLAGS.batch_size,
                        sc_fea_weights,
                        sc_fea_biases
                        )
        
        loss_name_scope = ('gpud_%d_loss' % gpu_index)

        sc_loss = tower_loss(
                        loss_name_scope,
                        sc_logit,
                        sc_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        100
                        )
        grads1 = opt_sc.compute_gradients(sc_loss, varlist1)
        tower_grads1.append(grads1)
        sc_logits.append(sc_logit)

    sc_logits = tf.concat(sc_logits,0)
    sc_accuracy = topk_acc(sc_logits, sc_labels_placeholder ,5)
    #sc_accuracy = tower_acc(sc_logits, sc_labels_placeholder)
    tf.summary.scalar('sc_accuracy', sc_accuracy)
    
    grads1 = average_gradients(tower_grads1)
    
    apply_gradient_sc = opt_sc.apply_gradients(grads1, global_step=global_step)
    
    train_sc = tf.group(apply_gradient_sc)
    
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(sc_fea_weights.values() + sc_fea_biases.values())
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)

    # Create summary writter
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/baseline_sc_visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/baseline_sc_visual_logs/test', sess.graph)
    for step in xrange(FLAGS.max_steps+1):
      
      start_time = time.time()
      trian_actions,train_images,train_ac_labels,train_sc_labels,train_mc_labels, _, _= input_data(
                      filename='./list/train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      start_pos=-1,
                      shuffle=True
                      )

      sess.run(train_sc, feed_dict={
                      images_placeholder: train_images,
                      sc_labels_placeholder: train_sc_labels,
                      keep_pro : 0.5
                      })
        
      duration = time.time() - start_time
      print('Batchnum %d: %.3f sec' % (step, duration))


      if (step) %50 == 0 or (step + 1) == FLAGS.max_steps:
        
        print('Step %d/%d: %.3f sec' % (step,FLAGS.max_steps, duration))
        print('Training Data Eval:')
        summary,sc_acc,sc_loss_value = sess.run(
                        [merged,sc_accuracy,sc_loss],
                        feed_dict={   
                              images_placeholder: train_images,
                              sc_labels_placeholder: train_sc_labels,
                              keep_pro : 1
                            })

        print ("sc_accuracy: " + "{:.5f}".format(sc_acc))
        print 'sc_loss= %.2f'% np.mean(sc_loss_value)
        train_writer.add_summary(summary, step)
        
      if (step) %100 == 0 or (step + 1) == FLAGS.max_steps:
        print('Validation Data Eval:')
        val_actions,val_images,val_ac_labels,val_sc_labels,val_mc_labels, _, _= input_data(
                        filename='./list/test.list',
                        start_pos=-1,
                        batch_size=FLAGS.batch_size * gpu_num,
                        shuffle=True)
        summary,sc_acc,sc_loss_value = sess.run(
                        [merged,sc_accuracy,sc_loss],
                        feed_dict={
                                  images_placeholder: val_images,
                                  sc_labels_placeholder: val_sc_labels,
                                  keep_pro : 1
                                        })
        print ("sc_accuracy: " + "{:.5f}".format(sc_acc))
        print 'sc_loss= %.2f'% np.mean(sc_loss_value)
        test_writer.add_summary(summary, step)
      # Save the model checkpoint periodically.
      if step > 1 and step % 5000 == 0:
        checkpoint_path = os.path.join('./models/baseline_sc_models', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step) 


  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
