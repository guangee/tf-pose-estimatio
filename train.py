import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_pose.pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from tf_pose.pose_augment import set_network_input_wh, set_network_scale
from tf_pose.common import get_sample_images
from tf_pose.networks import get_network

logger = logging.getLogger('train')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='cmu', help='model name')
    parser.add_argument('--datapath', type=str, default='G:/tensorflow/post-estimation/datasets/annotations')
    parser.add_argument('--imgpath', type=str, default='G:/tensorflow/post-estimation/datasets/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--lr', type=str, default='0.001')
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument('--input-width', type=int, default=432)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--quant-delay', type=int, default=-1)
    args = parser.parse_args()

    modelpath = logpath = './models/train/'

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    # 设置图片的宽高，做数据增强时用到
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['cmu', 'vgg'] or 'mobilenet' in args.model:
        # 因为CMU使用了VGG19网络，做了3次步长为2的max_pool操作，即缩小了8倍
        scale = 8

    # 设置scale，做数据增强时用到
    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        # 定义占位符
        # 输入图像 shape=(16, 368, 432, 3)
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
        # 向量图 shape=(16, 46, 54, 38)
        vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 38), name='vectmap')
        # 热图 shape=(16, 46, 54, 19)
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 19), name='heatmap')

        # prepare data
        # 初始化数据
        # args.datapath：annotations
        # batchsize：batchsize
        # imgpath：dataset
        # 解析 person_keypoints_train2017.json 文件，将训练数据存到队列
        df = get_dataflow_batch(args.datapath, True, args.batchsize, img_path=args.imgpath)
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
        # 从队列中移出元素
        q_inp, q_heat, q_vect = enqueuer.dequeue()

    # 解析 person_keypoints_val2017.json 文件
    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize, img_path=args.imgpath)
    df_valid.reset_state()
    validation_cache = []

    # 获取示例图片数据
    val_image = get_sample_images(args.input_width, args.input_height)
    logger.debug('tensorboard val image: %d' % len(val_image))
    logger.debug(q_inp)
    logger.debug(q_heat)
    logger.debug(q_vect)

    # define model for multi-gpu
    # 如果有多块GPU，将队列划分为多块，以分给每块GPU
    q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(q_vect, args.gpus)

    output_vectmap = []
    output_heatmap = []
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    outputs = []

    # 将任务分配到多块GPU上
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                # 根据传入的model参数获取net，已经训练好了的模型路径，最后一层网络名
                net, pretrain_path, last_layer = get_network(args.model, q_inp_split[gpu_id])

                # 如果传入参数checkpoint，则pretrain_path直接用checkpoint的路径而不是默认的路径
                if args.checkpoint:
                    pretrain_path = args.checkpoint

                # 获取最后一层的输出 L 和 S
                vect, heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)
                # 获取最后输出结果
                outputs.append(net.get_output())

                # 获取 stage2 后的每一层的输出 L 和 S
                l1s, l2s = net.loss_l1_l2()
                # 求每一层的L2范数 loss
                for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                    loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id], name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                    loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id], name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                    losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                # 最后一层的L2 范数 loss
                last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU")):
        # define loss
        # 计算每张图片的L1和L2总损失
        total_loss = tf.reduce_sum(losses) / args.batchsize
        # 计算每张图片的L1总损失
        total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
        # 计算每张图片的L2总损失
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
        # 计算每个batch 的L1和L2总损失
        total_loss_ll = tf.reduce_sum([total_loss_ll_paf, total_loss_ll_heat])

        # define optimizer
        # 设置学习率
        # 每个epoch执行的步数
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
            #                                            decay_steps=10000, decay_rate=0.33, staircase=True)
            # 学习率余弦衰减
            learning_rate = tf.train.cosine_decay(starter_learning_rate, global_step, args.max_epoch * step_per_epoch, alpha=0.0)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    # 量化模式，我们没用到，不用管
    if args.quant_delay >= 0:
        logger.info('train using quantized mode, delay=%d' % args.quant_delay)
        g = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=args.quant_delay)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.8, use_locking=True, use_nesterov=True)
    # 关于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用。
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # tf.control_dependencies，该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("queue_size", enqueuer.size())
    tf.summary.scalar("lr", learning_rate)
    merged_summary_op = tf.summary.merge_all()

    # 定义验证集和示例的占位符
    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t, valid_loss_ll_t])

    # 用于保存模型
    saver = tf.train.Saver(max_to_keep=1000)
    # 创建会话
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())
        # 加载模型
        if args.checkpoint and os.path.isdir(args.checkpoint):
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        elif pretrain_path:
            logger.info('Restore pretrained weights... %s' % pretrain_path)
            if '.npy' in pretrain_path:
                # 如果是npy的格式
                net.load(pretrain_path, sess, False)
            else:
                try:
                    loader = tf.train.Saver(net.restorable_variables(only_backbone=False))
                    loader.restore(sess, pretrain_path)
                except:
                    logger.info('Restore only weights in backbone layers.')
                    loader = tf.train.Saver(net.restorable_variables())
                    loader.restore(sess, pretrain_path)
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(os.path.join(logpath, args.tag), sess.graph)
        # 启动队列
        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        last_log_epoch1 = last_log_epoch2 = -1
        while True:
            # 开始训练
            _, gs_num = sess.run([train_op, global_step])
            # 当前epoch
            curr_epoch = float(gs_num) / step_per_epoch

            # 训练到指定次数了，退出
            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= 500:
                # 训练500步输出一次损失
                train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary = sess.run([total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, learning_rate, merged_summary_op])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat))
                last_gs_num = gs_num

                if last_log_epoch1 < curr_epoch:
                    file_writer.add_summary(summary, curr_epoch)
                    last_log_epoch1 = curr_epoch

            if gs_num - last_gs_num2 >= 2000:
                # 训练2000次保存一次
                # save weights
                saver.save(sess, os.path.join(modelpath, args.tag, 'model_latest'), global_step=global_step)

                average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                        validation_cache.append((images_test, heatmaps, vectmaps))
                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                # 输出测试准确率
                for images_test, heatmaps, vectmaps in validation_cache:
                    lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, output_vectmap, output_heatmap],
                        feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps}
                    )
                    average_loss += lss * len(images_test)
                    average_loss_ll += lss_ll * len(images_test)
                    average_loss_ll_paf += lss_ll_paf * len(images_test)
                    average_loss_ll_heat += lss_ll_heat * len(images_test)
                    total_cnt += len(images_test)

                logger.info('validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f' % (total_cnt, args.tag, average_loss / total_cnt, average_loss_ll / total_cnt, average_loss_ll_paf / total_cnt, average_loss_ll_heat / total_cnt))
                last_gs_num2 = gs_num

                sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                outputMat = sess.run(
                    outputs,
                    feed_dict={q_inp: np.array((sample_image + val_image) * max(1, (args.batchsize // 16)))}
                )
                pafMat, heatMat = outputMat[:, :, :, 19:], outputMat[:, :, :, :19]

                sample_results = []
                for i in range(len(sample_image)):
                    test_result = CocoPose.display_image(sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    sample_results.append(test_result)

                test_results = []
                for i in range(len(val_image)):
                    test_result = CocoPose.display_image(val_image[i], heatMat[len(sample_image) + i], pafMat[len(sample_image) + i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    test_results.append(test_result)

                # save summary
                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss: average_loss / total_cnt,
                    valid_loss_ll: average_loss_ll / total_cnt,
                    valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                    sample_valid: test_results,
                    sample_train: sample_results
                })
                if last_log_epoch2 < curr_epoch:
                    file_writer.add_summary(summary, curr_epoch)
                    last_log_epoch2 = curr_epoch

        saver.save(sess, os.path.join(modelpath, args.tag, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
