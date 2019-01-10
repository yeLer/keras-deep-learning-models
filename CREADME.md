# Kerasͼ����ೣ�����ѧϰ����

----

��������Keras��������ģ�ͽṹ:

- VGG16
- VGG19
- ResNet50
- Inception v3
- CRNN for music tagging

���е�ģ�ͼܹ����� TensorFlow �� Theano, ��ʵ����ʱ��������Keras�����ļ������õ�ͼ��ά�����򹹽�ģ�ͣ��ļ�λ�ã� `~/.keras/keras.json`. ����, ������Ѿ��趨 `image_dim_ordering=tf`, Ȼ�󣬽�����TensorFlowά������Լ�������Ӵ˴洢����ص��κ�ģ�� "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (`weights='imagenet'` argument in model constructor for all image models, `weights='msd'` for the music tagging model). Weights are automatically downloaded if necessary, and cached locally in `~/.keras/models/`.

## ����

### ͼ�����

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

### ��ͼ����ȡ����

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

### ���������ȡ����

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

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - ���ʹ����VGG��������ƪ���ġ�

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ���ʹ����ResNet ģ����������ƪ���ġ�

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) - ���ʹ����Inception ģ����������ƪ���ġ�

- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

���⣬���ʹ����keras,�ǵ������� [cite Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras) .


## Э��

- ���еĴ�����ѭMITЭ�顣
- ResNet50 ģ���ļ�Դ���� [released by Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- VGG16��VGG19��Ȩ���ļ�Դ���� [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).
- ��ʼv3�����������Լ�ѵ������MIT����·�����
