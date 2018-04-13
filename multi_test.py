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
pre_model_save_dir = './models/baseline_multi_models'
def placeholder_inputs(batch_size):
	#bulit placeholder_inputs
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         model.IMAGE_SIZE,
                                                         model.IMAGE_SIZE,
                                                         model.CHANNELS))
	sc_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	ac_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	mc_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
	keep_pro = tf.placeholder(tf.float32)

	return images_placeholder,sc_labels_placeholder,ac_labels_placeholder,mc_labels_placeholder,keep_pro


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

def tower_loss(name_scope, logit, sc_labels,ac_labels,mc_labels):
    sc_labels = tf.one_hot(sc_labels,100,on_value=1,off_value=None,axis=1)
    ac_labels = tf.one_hot(ac_labels,100,on_value=1,off_value=None,axis=1)
    mc_labels = tf.one_hot(mc_labels,256,on_value=1,off_value=None,axis=1)
    labels = tf.concat([sc_labels,ac_labels,mc_labels],1)
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
    total_loss = cross_entropy_mean  
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
    images_placeholder,sc_labels_placeholder,ac_labels_placeholder,mc_labels_placeholder,keep_pro =              placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    tower_grads3 = []
    multi_logits = []

    learning_rate=tf.train.exponential_decay(1e-4,global_step,decay_steps=FLAGS.max_steps/50,decay_rate=0.99,staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    opt_multi = tf.train.AdamOptimizer(learning_rate)
    with tf.variable_scope('var_name') as var_scope:
      
      multi_fea_weights = {
              'w1': _variable_with_weight_decay('multi_w1', [4096, 2048], 0.005),
              'out': _variable_with_weight_decay('multi_feawout', [2048, 456], 0.005)
              }
      multi_fea_biases = {
              'b1': _variable_with_weight_decay('multi_b1', [2048], 0.000),
              'out': _variable_with_weight_decay('multi_feabout', [456], 0.000),
              }      
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

        varlist1 = [ multi_fea_weights.values(),multi_fea_biases.values() ]
        
        vgg.build(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:])
        train_features=vgg.fc7
        
        multi_logit = model.get_predict(
                        train_features,
                        keep_pro,
                        FLAGS.batch_size,
                        multi_fea_weights,
                        multi_fea_biases
                        )
        
        loss_name_scope = ('gpud_%d_loss' % gpu_index)

        multi_loss = tower_loss(
                        'multi',
                        multi_logit,
                        sc_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        ac_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        mc_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                        )
        grads1 = opt_multi.compute_gradients(multi_loss, varlist1)
        tower_grads1.append(grads1)
        multi_logits.append(multi_logit)

    multi_logits = tf.concat(multi_logits,0)
    sc_logits = tf.slice(multi_logits,[0,0],[6,100])
    sc_predictions = tf.nn.top_k(tf.nn.softmax(sc_logits),5)
    sc_accuracy = topk_acc(sc_logits, sc_labels_placeholder ,5)
    #sc_accuracy = tower_acc(sc_logits, sc_labels_placeholder)
    tf.summary.scalar('sc_accuracy', sc_accuracy)
    ac_logits = tf.slice(multi_logits,[0,100],[6,100])
    ac_predictions = tf.nn.top_k(tf.nn.softmax(ac_logits),5)
    ac_accuracy = topk_acc(ac_logits, ac_labels_placeholder ,5)
    #ac_accuracy = tower_acc(ac_logits, ac_labels_placeholder)
    tf.summary.scalar('ac_accuracy', ac_accuracy)
    mc_logits = tf.slice(multi_logits,[0,200],[6,256])
    mc_predictions = tf.nn.top_k(tf.nn.softmax(mc_logits),5)
    mc_accuracy = topk_acc(mc_logits, mc_labels_placeholder ,5)
    #mc_accuracy = tower_acc(mc_logits, mc_labels_placeholder)
    tf.summary.scalar('mc_accuracy', mc_accuracy)
    
    grads1 = average_gradients(tower_grads1)
    
    apply_gradient_multi = opt_multi.apply_gradients(grads1, global_step=global_step)
    
    train_multi = tf.group(apply_gradient_multi)
    
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(multi_fea_weights.values() + multi_fea_biases.values())
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)

  ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)  
  if ckpt and ckpt.model_checkpoint_path:  
    print "loading checkpoint,waiting......"
    saver.restore(sess, ckpt.model_checkpoint_path)
    print "load complete!"
  next_strat_pos=0
  sc_predict_labels=[]
  ac_predict_labels=[]
  mc_predict_labels=[]
  for step in xrange(FLAGS.max_steps):
    
    start_time = time.time()
    print('TEST Data Eval:')
    val_actions,val_images,val_ac_labels,val_sc_labels,val_mc_labels,next_strat_pos, _= input_data(
                        filename='./list/test.list',
                        start_pos=next_strat_pos,
                        batch_size=FLAGS.batch_size * gpu_num,
                        shuffle=False)

    sc_predict,ac_predict,mc_predict,sc_acc,ac_acc,mc_acc = sess.run(
                        [sc_predictions,ac_predictions,mc_predictions,sc_accuracy,ac_accuracy,mc_accuracy],
                        feed_dict={
                                  images_placeholder:val_images,
                                  ac_labels_placeholder: val_ac_labels,
                                  sc_labels_placeholder: val_sc_labels,
                                  mc_labels_placeholder: val_mc_labels,
                                  keep_pro : 1
                                        })
    #print (ac_predict)
    for i in range(FLAGS.batch_size):
        sc_predict_labels.append(sc_predict[1][i])
        ac_predict_labels.append(ac_predict[1][i])
        mc_predict_labels.append(mc_predict[1][i])

    duration = time.time() - start_time
    print('Batchnum %d: %.3f sec' % (step+1, duration))
    #print predict_labels
    #print val_mc_labels

  print("get_predict_label_done!")
  return sc_predict_labels,ac_predict_labels,mc_predict_labels

def main(_):
  #for i in range(model.MCNUM_CLASSES):
  #  if not os.path.exist('./test/%d.txt'%i):
  lines = open('./list/test.list','r')
  lines = list(lines)
  datasets = open('./dataset.txt','r')
  datasets = list(datasets)
  cluster_256=np.load('./clusters_256.npz')
  cluster_100=np.load('./clusters_100.npz')
  sc_label_list,ac_label_list,mc_label_list=run_training()
  sc_sum=0
  ac_sum=0
  mc_sum=0
  sc_class_list=[0]*100
  sc_true_list=[0]*100
  ac_class_list=[0]*100
  ac_true_list=[0]*100
  mc_class_list=[0]*256
  mc_true_list=[0]*256
  for i in range(len(sc_label_list)):
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
    sc_class_list[int(tmp_sc_label)]+=1
    if tmp_sc_label in sc_label_list[i]:
        sc_true_list[int(tmp_sc_label)]+=1
  for i in range(100):
    if sc_class_list[i]!=0:
        sc_sum+= float(sc_true_list[i])/sc_class_list[i]
  print sc_sum/100

  for i in range(len(ac_label_list)):
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
    ac_class_list[int(tmp_ac_label)]+=1
    if tmp_ac_label in ac_label_list[i]:
        ac_true_list[int(tmp_ac_label)]+=1
  for i in range(100):
    if ac_class_list[i]!=0:
        ac_sum+= float(ac_true_list[i])/ac_class_list[i]
  print ac_sum/100

  for i in range(len(mc_label_list)):
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
    mc_class_list[int(tmp_mc_label)]+=1
    if tmp_mc_label in mc_label_list[i]:
        mc_true_list[int(tmp_mc_label)]+=1
  for i in range(256):
    if mc_class_list[i]!=0:
        mc_sum+= float(mc_true_list[i])/mc_class_list[i]
  print mc_sum/255



  '''
  for i in range(len(sc_label_list)):
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
    sc_sum += list(sc_label_list[i]).index(int(tmp_sc_label))
  print (float(sc_sum)/(421*6))

  for i in range(len(ac_label_list)):
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
    ac_sum += list(ac_label_list[i]).index(int(tmp_ac_label))
  print (float(ac_sum)/(421*6))
  for i in range(len(mc_label_list)):
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
    mc_sum += list(mc_label_list[i]).index(int(tmp_mc_label))
  print (float(mc_sum)/(421*6))
    '''
    
    


  '''
  for i in range(len(sc_label_list)):
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
    f=open('./test/multi/predict.txt','a+')
    f.write(dirname+'\t'+action+'\t'+motivation+'\t'+scene+'\t'+'ac_groud_truth:'+str(tmp_ac_label)+'\t'+'ac_predict:'+str(ac_label_list[i])+'mc_groud_truth:'+str(tmp_mc_label)+'\t'+'mc_predict:'+str(mc_label_list[i])+'sc_groud_truth:'+str(tmp_sc_label)+'\t'+'sc_predict:'+str(sc_label_list[i])+'\n')
  for i in range(len(ac_label_list)):
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
    f=open('./test/multi/action/%d.txt'%int(tmp_ac_label),'a+')
    f.write(str(int(line_num)+1)+'\t'+dirname+'\t'+action+'\t'+'groud_truth:'+str(tmp_ac_label)+'\t'+'predict:'+str(ac_label_list[i])+'\n')
  for i in range(len(sc_label_list)):
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
    f=open('./test/multi/scene/%d.txt'%int(tmp_sc_label),'a+')
    f.write(str(int(line_num)+1)+'\t'+dirname+'\t'+scene+'\t'+'groud_truth:'+str(tmp_sc_label)+'\t'+'predict:'+str(sc_label_list[i])+'\n')
  for i in range(len(mc_label_list)):
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
    f=open('./test/multi/motivation/%d.txt'%int(tmp_mc_label),'a+')
    f.write(str(int(line_num)+1)+'\t'+dirname+'\t'+motivation+'\t'+'groud_truth:'+str(tmp_mc_label)+'\t'+'predict:'+str(mc_label_list[i])+'\n')
   '''
if __name__ == '__main__':
  tf.app.run()