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
pre_model_save_dir = './models/step1_models'
gpu_num = 1
def placeholder_inputs(batch_size):
	#bulit placeholder_inputs
	actions_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            
                                                         model.INPUT_SIZE,
                                                            ))
	scenes_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            
                                                         model.INPUT_SIZE,
                                                            ))
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         model.IMAGE_SIZE,
                                                         model.IMAGE_SIZE,
                                                         model.CHANNELS))
	ac_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	sc_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	mc_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	keep_pro = tf.placeholder(tf.float32)

	return actions_placeholder,scenes_placeholder,images_placeholder,ac_labels_placeholder,sc_labels_placeholder,mc_labels_placeholder,keep_pro


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
    actions_placeholder,scenes_placeholder,images_placeholder,ac_labels_placeholder,sc_labels_placeholder,mc_labels_placeholder,keep_pro =              placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    tower_grads3 = []
    sc_logits = []
    ac_logits = []
    mc_logits = []

    learning_rate_sc=tf.train.exponential_decay(1e-4,global_step,decay_steps=FLAGS.max_steps/50,decay_rate=0.98,staircase=True)
    learning_rate_ac=tf.train.exponential_decay(1e-4,global_step,decay_steps=FLAGS.max_steps/50,decay_rate=0.98,staircase=True)
    learning_rate_mc=tf.train.exponential_decay(1e-5,global_step,decay_steps=FLAGS.max_steps/50,decay_rate=0.98,staircase=True)
    tf.summary.scalar('learning_rate', learning_rate_sc)
    #tf.summary.scalar('learning_rate2', learning_rate2)
    opt_sc = tf.train.AdamOptimizer(learning_rate_sc)
    opt_ac = tf.train.AdamOptimizer(learning_rate_ac)
    opt_mc = tf.train.AdamOptimizer(1e-5)
    with tf.variable_scope('var_name') as var_scope:
      sc_weights = {
              'w1': _variable_with_weight_decay('sc_w1', [4800, 4096], 0.005),
              'w2': _variable_with_weight_decay('sc_w2', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('sc_wout', [2048, model.SCNUM_CLASSES], 0.005)
              }
      sc_biases = {
              'b1': _variable_with_weight_decay('sc_b1', [4096], 0.000),
              'b2': _variable_with_weight_decay('sc_b2', [2048], 0.000),
              'out': _variable_with_weight_decay('sc_bout', [model.SCNUM_CLASSES], 0.000),
              }
      ac_weights = {
              'w1': _variable_with_weight_decay('ac_w1', [4800, 4096], 0.005),
              'w2': _variable_with_weight_decay('ac_w2', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('ac_wout', [2048, model.ACNUM_CLASSES], 0.005),
              'W_alpha': _variable_with_weight_decay('alpha_learn', [100,100], 0.005),
              'W_Ua': _variable_with_weight_decay('Ua', [100,100], 0.005),
              }
      ac_biases = {
              'b1': _variable_with_weight_decay('ac_b1', [4096], 0.000),
              'b2': _variable_with_weight_decay('ac_b2', [2048], 0.000),
              'out': _variable_with_weight_decay('ac_bout', [model.ACNUM_CLASSES], 0.000),
              }
      mc_weights = {
              'w1': _variable_with_weight_decay('mc_w1', [4800, 4096], 0.005),
              'w2': _variable_with_weight_decay('mc_w2', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('mc_wout', [2048, model.MCNUM_CLASSES], 0.005),
              'W_beta': _variable_with_weight_decay('beta_learn', [256,100], 0.005),
              'W_gama': _variable_with_weight_decay('gama_learn', [256,100], 0.005),
              'W_Um': _variable_with_weight_decay('Um', [256,256], 0.005),
              }
      mc_biases = {
              'b1': _variable_with_weight_decay('mc_b1', [4096], 0.000),
              'b2': _variable_with_weight_decay('mc_b2', [2048], 0.000),
              'out': _variable_with_weight_decay('mc_bout', [model.MCNUM_CLASSES], 0.000),
              }
        
    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):

        varlist1 = list(set(sc_weights.values()+sc_biases.values()))
        varlist2 = list(set(ac_weights.values()+ac_biases.values())-set([ac_weights['W_alpha']])-set([ac_weights['W_Ua']]))
        varlist3 = list(set(mc_weights.values()+mc_biases.values())-set([mc_weights['W_beta']])-set([mc_weights['W_gama']])-set([mc_weights['W_Um']]))
        alpha = ac_weights['W_alpha']
        beta = mc_weights['W_beta']
        gama = mc_weights['W_gama']
        
        vgg.build(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:])
        train_features=vgg.fc6
        sc_logit = model.sc_model(
                        scenes_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:],
                        keep_pro,
                        FLAGS.batch_size,
                        sc_weights,
                        sc_biases
                        )
        
        ac_logit = model.ac_model(
                        actions_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:],
                        sc_logit,
                        keep_pro,
                        FLAGS.batch_size,
                        ac_weights,
                        ac_biases
                        )
        mc_logit,update_ema1,update_ema2 = model.mc_model2(
                        train_features,
                        sc_logit,
                        ac_logit,
                        keep_pro,
                        FLAGS.batch_size,
                        mc_weights,
                        mc_biases,
                        global_step
                        )
        
        mc_update_ema = tf.group(update_ema1, update_ema2)
        loss_name_scope = ('gpud_%d_loss' % gpu_index)

        
        sc_loss = tower_loss(
                        loss_name_scope+'_scene',
                        sc_logit,
                        sc_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        model.SCNUM_CLASSES
                        )
        ac_loss = tower_loss(
                        loss_name_scope+'_action',
                        ac_logit,
                        ac_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        model.ACNUM_CLASSES
                        )
        mc_loss = tower_loss(
                        loss_name_scope+'_motivation',
                        mc_logit,
                        mc_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        model.MCNUM_CLASSES
                        )
        grads1 = opt_sc.compute_gradients(sc_loss, varlist1)
        grads2 = opt_ac.compute_gradients(ac_loss, varlist2)
        grads3 = opt_mc.compute_gradients(mc_loss, varlist3)
        
        tower_grads1.append(grads1)
        tower_grads2.append(grads2)
        tower_grads3.append(grads3)
        
        sc_logits.append(sc_logit)
        ac_logits.append(ac_logit)
        mc_logits.append(mc_logit)

    sc_logits = tf.concat(sc_logits,0)
    sc_predictions = tf.nn.top_k(sc_logits,5)
    
    ac_logits = tf.concat(ac_logits,0)
    ac_predictions = tf.nn.top_k(ac_logits,5)
    
    mc_logits = tf.concat(mc_logits,0)
    mc_predictions = tf.nn.top_k(mc_logits,5)

    #sc_accuracy = tower_acc(sc_logits, sc_labels_placeholder)
    sc_accuracy = topk_acc(sc_logits, sc_labels_placeholder ,5)
    tf.summary.scalar('sc_accuracy', sc_accuracy)
    
    #ac_accuracy = tower_acc(ac_logits, ac_labels_placeholder)
    ac_accuracy = topk_acc(ac_logits, ac_labels_placeholder ,5)
    tf.summary.scalar('ac_accuracy', ac_accuracy)
    
    #mc_accuracy = tower_acc(mc_logits, mc_labels_placeholder)
    mc_accuracy = topk_acc(mc_logits, mc_labels_placeholder ,5)
    tf.summary.scalar('mc_accuracy', mc_accuracy)
    
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    grads3 = average_gradients(tower_grads3)
    
    apply_gradient_sc = opt_sc.apply_gradients(grads1, global_step=global_step)    
    apply_gradient_ac = opt_ac.apply_gradients(grads2, global_step=global_step)
    apply_gradient_mc = opt_mc.apply_gradients(grads3, global_step=global_step)

    train_sc = tf.group(apply_gradient_sc)
    train_ac = tf.group(apply_gradient_ac)
    train_mc = tf.group(apply_gradient_mc)
    
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(sc_weights.values() + sc_biases.values()+ac_weights.values() + ac_biases.values()+mc_weights.values() + mc_biases.values())
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/step2_visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/step2_visual_logs/test', sess.graph)
  ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)  
  if ckpt and ckpt.model_checkpoint_path:  
    print "loading checkpoint,waiting......"
    saver.restore(sess, ckpt.model_checkpoint_path)
    print "load complete!"
  test_list=[]
  for step in xrange(FLAGS.max_steps+1):
      start_time = time.time()
      train_actions,train_scenes,train_images,train_ac_labels,train_sc_labels,train_mc_labels, _, _= input_data(
                      filename='./list/train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      start_pos=-1,
                      shuffle=True
                      )
      '''
      sess.run(train_sc, feed_dict={
                      actions_placeholder:train_actions,
                      images_placeholder: train_images,
                      ac_labels_placeholder: train_ac_labels,
                      sc_labels_placeholder: train_sc_labels,
                      mc_labels_placeholder: train_mc_labels,
                      keep_pro : 0.5
                      })
      sess.run(train_ac, feed_dict={
                      actions_placeholder:train_actions,        
                      images_placeholder: train_images,
                      ac_labels_placeholder: train_ac_labels,
                      sc_labels_placeholder: train_sc_labels,
                      mc_labels_placeholder: train_mc_labels,
                      keep_pro : 0.5
                      })
      '''
      sess.run([train_mc,mc_update_ema], feed_dict={
                      actions_placeholder:train_actions,   
                      scenes_placeholder:train_scenes,
                      images_placeholder: train_images,
                      ac_labels_placeholder: train_ac_labels,
                      sc_labels_placeholder: train_sc_labels,
                      mc_labels_placeholder: train_mc_labels,
                      keep_pro : 0.5
                      })
      duration = time.time() - start_time
      print('Batchnum %d: %.3f sec' % (step, duration))


      if (step) %10 == 0 or (step + 1) == FLAGS.max_steps:
        
        print('Step %d/%d: %.3f sec' % (step,FLAGS.max_steps, duration))
        print('Training Data Eval:')
        summary,sc_acc,ac_acc,mc_acc,sc_loss_value,ac_loss_value,mc_loss_value= sess.run(
                        [merged,sc_accuracy,ac_accuracy,mc_accuracy,sc_loss,ac_loss,mc_loss],
                        feed_dict={  
                                  actions_placeholder:train_actions,
                                  scenes_placeholder:train_scenes, 
                                  images_placeholder: train_images,
                                  ac_labels_placeholder: train_ac_labels,
                                  sc_labels_placeholder: train_sc_labels,
                                  mc_labels_placeholder: train_mc_labels,
                                  keep_pro : 1
                            })
        print ("sc_accuracy: " + "{:.5f}".format(sc_acc))
        print 'sc_loss= %.2f'% np.mean(sc_loss_value)
        print ("ac_accuracy: " + "{:.5f}".format(ac_acc))
        print 'ac_loss= %.2f'% np.mean(ac_loss_value)
        print ("mc_accuracy: " + "{:.5f}".format(mc_acc))
        print 'mc_loss= %.2f'% np.mean(mc_loss_value)
        train_writer.add_summary(summary, step)
        
      if (step) %20 == 0 or (step + 1) == FLAGS.max_steps:
        print('Validation Data Eval:')
        val_actions,val_scenes,val_images,val_ac_labels,val_sc_labels,val_mc_labels, _, _= input_data(
                        filename='./list/test.list',
                        start_pos=-1,
                        batch_size=FLAGS.batch_size * gpu_num,
                        shuffle=True)

        summary,sc_acc,ac_acc,mc_acc,sc_loss_value,ac_loss_value,mc_loss_value = sess.run(
                        [merged,sc_accuracy,ac_accuracy,mc_accuracy,sc_loss,ac_loss,mc_loss],
                        feed_dict={ 
                                  actions_placeholder:val_actions, 
                                  scenes_placeholder:val_scenes, 
                                  images_placeholder: val_images,
                                  ac_labels_placeholder: val_ac_labels,
                                  sc_labels_placeholder: val_sc_labels,
                                  mc_labels_placeholder: val_mc_labels,
                                  keep_pro : 1
                                        })
        print ("sc_accuracy: " + "{:.5f}".format(sc_acc))
        print 'sc_loss= %.2f'% np.mean(sc_loss_value)
        print ("ac_accuracy: " + "{:.5f}".format(ac_acc))
        print 'ac_loss= %.2f'% np.mean(ac_loss_value)
        print ("mc_accuracy: " + "{:.5f}".format(mc_acc))
        print 'mc_loss= %.2f'% np.mean(mc_loss_value)
        test_writer.add_summary(summary, step)
        test_list.append()
        # Save the model checkpoint periodically.
      if step > 1 and step % 2000 == 0:
        checkpoint_path = os.path.join('./models/step2_models', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step) 


  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
