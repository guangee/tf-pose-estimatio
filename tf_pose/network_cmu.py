from __future__ import absolute_import

from tf_pose import network_base
import tensorflow as tf


class CmuNetwork(network_base.BaseNetwork):
    def setup(self):
        # 用了 VGG19 的前10层，对后基层网络进行了微调
        (self.feed('image')
             .normalize_vgg(name='preprocess')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1_stage1', padding='VALID')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2_stage1', padding='VALID')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .conv(3, 3, 256, 1, 1, name='conv3_4')
             .max_pool(2, 2, 2, 2, name='pool3_stage1', padding='VALID')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')    # 这里上去的都是VGG19的前10层网络

             .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
             .conv(3, 3, 128, 1, 1, name='conv4_4_CPM')          # ***** 得到原始图片的特征图F

##########################################################################################
        # stage 1 ,分别得到 S1 和 L1
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L1')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='conv5_5_CPM_L1'))

        (self.feed('conv4_4_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L2')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='conv5_5_CPM_L2'))
##########################################################################################
        # stage2，将上一个stage得到的 S 和 L，再加上原始图片特征图F当成输入
        # L1 是矢量图 L， L2 是热图（置信图）S
        (self.feed('conv5_5_CPM_L1',
                   'conv5_5_CPM_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage2_L2'))
##########################################################################################
        # stage3，将上一个stage得到的 S 和 L，再加上原始图片特征图F当成输入
        (self.feed('Mconv7_stage2_L1',
                   'Mconv7_stage2_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage3_L1'))

        (self.feed('concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage3_L2'))
##########################################################################################
        # stage4，将上一个stage得到的 S 和 L，再加上原始图片特征图F当成输入
        (self.feed('Mconv7_stage3_L1',
                   'Mconv7_stage3_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage4_L1'))

        (self.feed('concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage4_L2'))
##########################################################################################
        # stage4，将上一个stage得到的 S 和 L，再加上原始图片特征图F当成输入
        (self.feed('Mconv7_stage4_L1',
                   'Mconv7_stage4_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage5_L1'))

        (self.feed('concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage5_L2'))
##########################################################################################
        # stage6，将上一个stage得到的 S 和 L，再加上原始图片特征图F当成输入
        (self.feed('Mconv7_stage5_L1',
                   'Mconv7_stage5_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage6_L1'))

        (self.feed('concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage6_L2'))
##########################################################################################
        # 最后一层，将得到的 S6 和 L6 拼接
        with tf.variable_scope('Openpose'):
            (self.feed('Mconv7_stage6_L2',
                       'Mconv7_stage6_L1')
                 .concat(3, name='concat_stage7'))

    # 获取 stage2 后的每一层的输出 L (矢量图)和 S（热图）
    def loss_l1_l2(self):
         l1s = []
         l2s = []
         for layer_name in self.layers.keys():
              if 'Mconv7' in layer_name and '_L1' in layer_name:
                   l1s.append(self.layers[layer_name])
              if 'Mconv7' in layer_name and '_L2' in layer_name:
                   l2s.append(self.layers[layer_name])

         return l1s, l2s

    # 获取最后一层的输出L 和 S
    def loss_last(self):
         return self.get_output('Mconv7_stage6_L1'), self.get_output('Mconv7_stage6_L2')

    def restorable_variables(self):
         return None
