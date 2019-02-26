# ===========
# residual block implemented in YOLOv3 using keras
# YOLOv3将downsample/bottleneck/skip在同一个block全部处理，因此形式较为奇怪
# ===========

from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D
from keras.layers.merge import add

x = _conv_block(x, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                    {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1}, 
                    {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2}, 
                    {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}], do_skip=True)

x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5}, 
                    {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6}, 
                    {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}, do_skip=True])

def _conv_block(inp, convs, do_skip=True):
    x = inp
    count = 0

    for conv in convs:
        if do_skip and count == len(convs)-2:
            skip_connection = x
        count += 1
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # unlike tensorflow YOLO prefer left and top padding
        x = Conv2D(conv['filter'], 
                   conv['kernel'], 
                   strides=conv['stride'], 
                   padding='valid' if conv['stride'] > 1 else 'same', 
                   name='conv_' + str(conv['layer_idx']), 
                   use_bias=False if conv['bnorm'] else True)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky' + str(conv['layer_idx']))(x)
    
    return add(x, skip_connection) if do_skip else x


# ===========
# residual block v2 implemented in slim/models using tensorflow.slim
# ===========


