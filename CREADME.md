# Keras图像分类常用深度学习网络

----

这个库包含Keras下列网络模型结构:

- VGG16
- VGG19
- ResNet50
- Inception v3
- CRNN for music tagging

所有的模型架构兼容 TensorFlow 和 Theano, 在实例化时，将根据Keras配置文件中设置的图像维度排序构建模型，文件位置： `~/.keras/keras.json`. 例如, 如果你已经设定 `image_dim_ordering=tf`, 然后，将根据TensorFlow维度排序约定构建从此存储库加载的任何模型 "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (`weights='imagenet'` argument in model constructor for all image models, `weights='msd'` for the music tagging model). Weights are automatically downloaded if necessary, and cached locally in `~/.keras/models/`.

## 例子

### 图像分类

```python
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
```

### 从图像提取特征

```python
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### 从任意层提取特征

```python
from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - 如果使用了VGG请引用这篇论文。

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - 如果使用了ResNet 模型请引用这篇论文。

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) - 如果使用了Inception 模型请引用这篇论文。

- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

此外，如果使用了keras,记得引用它 [cite Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras) .


## 协议

- 所有的代码遵循MIT协议。
- ResNet50 模型文件源自于 [released by Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- VGG16和VGG19的权重文件源自于 [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).
- 初始v3重量由我们自己训练并在MIT许可下发布。
