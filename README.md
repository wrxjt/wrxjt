- 👋 Hi, I’m @wrxjt
- 👀 I’m interested in CV
- 🌱 I’m currently learning Deep learning
- 💞️ I’m looking to collaborate on architecture CNN & VIT
- 📫 How to reach me ...

Effiecient densenet
近十年来卷积神经网络发展出各式各样的变体，不断地使网络加宽加深、
但是即便每个网络有各自地特点，在融合网络上却做的不是很好。
这篇论文，我们以densenet为媒介，提出一种新的框架，使得网络融合变得可行。
具体来说，densenet的第l层接受前l-1层的特征图，且每个特征图的通道数都为k。
我们认为对于第l层来说，第l-1层的特征图是最重要的信息流，所以它的通道数应该
远远大于k，从而我们引出一条支路网络，使得第l-1层的特征图被充分学习。而这个
支路网络可以用任意网络填充，为了使得这个densenet变体具有多样性，我们设计了
一个网络块融合了各个网络的特性。另一方面，由于这条支路网络的可替代性和可对齐性，
我们可以将主流基准网络对标放入，并利用上述设计块充当densenet块融入。这种设计相
当于对任意的基准网络添加densenet网络的性能。而如果这个块是自定义块，则可以融合
多个网络的特点，从而达到了网络融合。我们利用改进后的网络在cifar10,cifar100,svhn数
据集上进行测试，取得了惊人的进展。
cnn has dominated the CV space since alexnet burst onto the scene. In recent years, there have been many improved networks, but with the increasing width and depth of the network, bottlenecks gradually appear, making the progress of traditional CNN in the past two years begin to develop towards the direction of NAS network search. Transformer model began to enter the field of computer vision. vit,swintransformer and other models proved that without the use of convolutional neural network, excellent results could be achieved and they tried to unify the country. Multimodal learning moco,clip, mae also shine. However, since the transformer model needs to learn from an unusually large data set, it is difficult to prove whether the effect of the above model is due to the model itself or the large amount of ancient data. The convnet shows that the traditional convolutional neural network may not be worse than vit, and in the past two years, many networks have fused vit with cnn, proving that convolutional neural network is still indelible in computer vision.
In this paper, we try to integrate the excellent cnn network with densenet in recent years, so that the characteristics of most networks can be brought into play with one network, so as to achieve more efficient and stable performance. We propose two improvements based on densenet.
The first kind of network is densenet as the main network and integrates features of other networks. The reason we do this is that densenet accepts the same number of channels in the feature map of the first l-1 layer, which is obviously not the optimal solution. Because for layer l it mainly learns the features of layer l minus 1. To this end, we increase the number of channels of the l-1 layer feature map, and these additional feature maps can be learned by using another network. We do this to make the additional feature map can be learned by other high-performance networks, so as to achieve the characteristics of other networks and densenet combination. So, we designed a network that combines the features of googlenet and resnext. In addition, we believe that for layer l, the closer the layer is to it (l-2, L-3...). The feature maps of are fully learned, so to prevent overfitting, we cut off the reuse of feature maps within two layers apart. This not only indirectly reduces the parameters, but also prevents overfitting.
The second improved network is to put the baseline network into the red line as the main network as shown in the figure, and then put the corresponding blocks into the densenet network. In this way, the baseline network can have the characteristics of densenet and the corresponding blocks without changing the original characteristics, and realize the network fusion. Intuitively, if the features in the two networks are fundamentally different, the results are at least better than they would have been. This network also indirectly verifies that there are real differences in the characteristics of the emerging networks,
