# Deep Neural Networks for Kinship prediction using face photos

### Requirements :

Keras 2

keras_vggface

to install keras_vggface => "pip install git+https://github.com/rcmalli/keras-vggface.git"

### Data :

The data is available at : https://www.kaggle.com/c/recognizing-faces-in-the-wild/data

It needs to be dezipped into the folder input

### Description

Can we predict if two people are blood related given just a photo of their face
?<br> This is what will try to do in this post by using the [Families In the
Wild: A Kinship Recognition
Benchmark](https://web.northeastern.edu/smilelab/fiw/) dataset in the format
shared on Kaggle :
[https://www.kaggle.com/c/recognizing-faces-in-the-wild/data](https://www.kaggle.com/c/recognizing-faces-in-the-wild/data)

We will explore the effectiveness deep neural networks and transfer learning
techniques to train a neural network that will predict if two people are blood
related or not given a picture of their face.

#### Dataset :

We will use the Families in the Wild dataset shared on
[kaggle](https://www.kaggle.com/c/recognizing-faces-in-the-wild/data). It is the
biggest scale data set of its kind where face photos are grouped by person and
then people are grouped by family.

![](https://cdn-images-1.medium.com/max/1600/1*am4xEqhFkLlgiu7JC9CrWw.png)
<span class="figcaption_hack">Image organization in the FIW dataset</span>

Other than the image folders we also have a file that lists all the cases where
two people from a family are blood related, which is not the case for all
members of the family. (Like the pair mother-father, unless we are talking about
Lannisters 😄 )

#### Model :

In order to solve this task we will use a Siamese network that takes a pair of
images and predict 1 if the people in the photos are related and 0 otherwise.

![](https://cdn-images-1.medium.com/max/1600/1*8h3JpmmLLjeDFUHJYGCl0Q.png)
<span class="figcaption_hack">Siamese Network</span>

The image encoder is applied to each input image and encodes each of them into a
fixed length vector. The square of the difference between the two image vectors
are fed into a fully connected layer which then predicts a binary label of
kinship.

![](https://cdn-images-1.medium.com/max/1600/1*YEt68fg0lmm01CWqR6BO1A.png)
<span class="figcaption_hack">Example of input/output</span>

#### Transfer Learning :

We will base our solution on pretrained image encoders using two different
settings :

Pretraining on ImageNet : A data-set of *14 million* manually labeled images
used for classification into categories like dog, cat, airplane, strawberry …

[Pretraining on VGGFACE2](https://github.com/rcmalli/keras-vggface) : A data-set
of *3.3 million* face images and 9000+ identities in a a wide range of different
ethnicities, accents, professions and ages.

Pretraining techniques are useful because they allow us to transfer
representations learned on a source task ( here image classification or
regression ) into a target task which is kinship prediction in this case.<br>
This helps reduce over-fitting and achieve a much faster convergence rate,
especially if the source task and target task are somewhat close.

#### Results :

We will use the accuracy and AUC Score to evaluate the results of each model.

Resnet50 Imagenet test ROC AUC : **0.70**

![](https://cdn-images-1.medium.com/max/1600/1*XzBsM43ttD4px7d-iIx0Yg.png)
<span class="figcaption_hack">Evaluation on the test set through kaggle submission</span>

Resnet50 VGGFACE2 test ROC AUC : **0.81**

![](https://cdn-images-1.medium.com/max/1600/1*k1sZBMiXcwP6zsKUWKgDag.png)
<span class="figcaption_hack">Evaluation on the test set through kaggle submission</span>

![](https://cdn-images-1.medium.com/max/1600/1*KCpJNyyraN1gyF4H1IfhkA.png)
<span class="figcaption_hack">Validation Accuracy Comparison</span>

We can see that even the architecture in the two different settings is the same
the results are much better on the model [pretrained on
VGGFace2](https://github.com/rcmalli/keras-vggface) since its a source task that
is much closer to the target task of kinship prediction compared to Imagenet.

### Conclusion :

We showed that we can train a Siamese network to predict if two people are blood
related or not. By using transfer learning we were able to achieve encouraging
results of **0.81** in AUC ROC, especially given that the task of kinship 👪
prediction is pretty hard even for humans.


