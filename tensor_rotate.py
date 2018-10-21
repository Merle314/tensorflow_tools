# 加载库
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def rot90(tensor,k=1,axes=[1,2],name=None):
    '''
    autor:lizh
    tensor: a tensor 4 or more dimensions
    k: integer, Number of times the array is rotated by 90 degrees.
    axes: (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.
    
    -----
    Returns
    -------
    tensor : tf.tensor
             A rotated view of `tensor`.
    See Also: https://www.tensorflow.org/api_docs/python/tf/image/rot90 
    '''
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
        
    tenor_shape = (tensor.get_shape().as_list())
    dim = len(tenor_shape)
    
    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == dim:
        raise ValueError("Axes must be different.")
        
    if (axes[0] >= dim or axes[0] < -dim 
        or axes[1] >= dim or axes[1] < -dim):
        
        raise ValueError("Axes={} out of range for tensor of ndim={}."
            .format(axes, dim))
    k%=4
    if k==0:
        return tensor
    if k==2:
        img180 = tf.reverse(tf.reverse(tensor, axis=[axes[0]]),axis=[axes[1]],name=name)
        return img180
    
    axes_list = np.arange(0, dim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],axes_list[axes[0]]) # 替换
    
    print(axes_list)
    if k==1:
        img90=tf.transpose(tf.reverse(tensor,axis=[axes[1]]), perm=axes_list, name=name)
        return img90
    if k==3:
        img270=tf.reverse( tf.transpose(tensor, perm=axes_list),axis=[axes[1]],name=name)
        return img270

# 手写体数据集 加载
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/lizhen/data/MNIST/", one_hot=True)

sess=tf.Session()
#选取数据 4D
images = mnist.train.images
img_raw = images[0,:] # [0,784]
img=tf.reshape(img_raw,[-1,28,28,1]) # img 现在是tensor
# 绘图
def fig_2D_tensor(tensor):# 绘图
    #plt.matshow(tensor, cmap=plt.get_cmap('gray'))
    plt.matshow(tensor) # 彩色图像
    # plt.colorbar() # 颜色条
    plt.show()
# 显 显示 待旋转的图片
fig_2D_tensor(sess.run(img)[0,:,:,0]) # 提取ndarray