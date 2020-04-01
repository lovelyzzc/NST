from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import functools

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# 如果你没有图片拿来做style，和content，就从下面加载
content_path = tf.keras.utils.get_file('turtle.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/example_images'
                                       '/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg',
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images'
                                     '/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# GRADED FUNCTION：load_img

def load_img(path_to_img):
    """
    加载图像，并将其最大尺寸限制为512像素
    :param path_to_img: image所在的位置
    :return: 返回image
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


# GRADED FUNCTION: imshow

def imshow(image, title=None):
    """
    显示图像
    :param image: 图像数据
    :param title: 图像标题
    :return: 无
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


# 在这里可以先来查看content 和style
# 如果要用你自己的图片，记得修改图片路径
content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# 在模型的选择上，综合考虑计算资源和模型的优缺点，我使用了VGG19，这是一个已经预训练好的图像分类网络
# VGG19可以直接从tf中下载，也可以从本地加载，如果你下载太慢，我把模型上传到了github，你可以从那里下载

# 预处理
x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
x = tf.image.resize(x, (224, 224))

# 加载模型,查看网络结果
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
print()
for layer in vgg.layers:
    print(layer.name)

# 如果你想要换特征可以自行修改下方的layer
# 内容层将提取出我们的 feature maps （特征图）
content_layers = ['block5_conv2']

# 我们感兴趣的风格层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# 建立模型

def vgg_layers(layer_name):
    # 加载模型， 返回输入值
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outpots = [vgg.get_layer(name).output for name in layer_name]

    model = tf.keras.Model([vgg.input], outpots)

    return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)

# 查看每层输出的统计信息
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()


# GRADED FUNCTION: gram_matrix

def gram_matrix(input_tensor):
    """
    gram矩阵
    :param input_tensor: 输入数据
    :return: 输入数据的gram
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / num_locations


# 构建一个返回style 和content的模型

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """

        :param inputs: 值在[0, 1]之间
        :return:
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[: self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            contente_name: value for contente_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {'content': content_dict, 'style': style_dict}


# BP算法
extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# 为了优化它， 使用两个损失的加权组合来获得总损失
style_weight = 1e-2
content_weight = 1e4


# GRADED FUNCTION: style_content_loss

def style_content_loss(outputs):
    """
    计算风格和内容损失
    :param outputs: style_outputs 和 conten_outputs
    :return: 总损失
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss

    return loss


# 使用tf.GradientTape更新图像
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# 测试一下现在的优化器
# start = time.time()
#
# epochs = 10
# step_per_epoch = 100

# step = 0
# for n in range(epochs):
#     for m in range(step_per_epoch):
#         step += 1
#         train_step(image)
#         print('------{}--------这是第{}批，第{}张图片.'.format(str(step), str(n), str(m)))
#         display.clear_output(wait=True)
#         imshow(image.read_value())
#         plt.title('Train_step:{}'.format(str(step)))
#         plt.show()

# end = time.time()
# print('Total time: {:.1f}'.format(end - start))


# 这个实现发现他有一个缺点，会产生大量的高频误差， 所以我们通过正则化图像的高频误差来减少这种高频误差
# 总变分损失
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, : -1, :]
    y_var = image[:, 1:, :, :] - image[:, : -1, :, :]

    return x_var, y_var


# 查看原图和风格变换之后的高频分量
# 高频分量本质上是一个边缘检测器
x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas: Original")

plt.subplot(2, 2, 2)
imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas: Styled")

plt.subplot(2, 2, 4)
imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas: Styled")


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)

    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)


# 重新优化loss
total_variation_weight = 1e8


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# 重新初始化优化变量
image = tf.Variable(content_image)

start = time.time()

epochs = 10
step_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(step_per_epoch):
        step += 1
        train_step(image)
        print('------{}--------这是第{}批，第{}张图片.'.format(str(step), str(n), str(m)))
    display.clear_output(wait=True)
    imshow(image.read_value())
    plt.title('Train step: {}'.format(str(step)))
    plt.show()

end = time.time()
print('Total time: {:.1f}'.format(end - start))

file_name = 'kadinsky-turtle.png'
mpl.image.imsave(file_name, image[0])

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(file_name)
