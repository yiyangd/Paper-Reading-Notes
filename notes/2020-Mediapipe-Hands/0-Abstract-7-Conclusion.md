## Paper Reading | MediaPipe Hands: On-device Real-time Hand Tracking (2020)

![https://arxiv.org/pdf/2006.10214](https://files.mdnice.com/user/1474/c54258ac-4199-423d-9fe5-941ffb702990.png)

### Abstract
| **原文** | **翻译** |
| --- | --- |
| We present a real-time on-device hand tracking solution that predicts a hand skeleton of a human from a single RGB camera for AR/VR applications. | 我们提出了一种实时的设备端手部追踪解决方案，可以利用单个RGB摄像头预测人手的骨架结构，适用于AR/VR应用。 |
| Our pipeline consists of two models: 1) a palm detector, that is providing a bounding box of a hand to, 2) a hand landmark model, that is predicting the hand skeleton. | 我们的流程由两个模型组成：1) 手掌检测器，用于提供手部的边界框；2) 手部关键点模型，用于预测手部骨架。 |
| It is implemented via MediaPipe[12], a framework for building cross-platform ML solutions.| 该流程通过MediaPipe[12]实现，这是一个用于构建跨平台机器学习解决方案的框架。 |
|[12] *Mediapipe: A framework for building perception pipelines*. (2019)|![](https://files.mdnice.com/user/1474/0b7905da-2e05-4b2d-a179-60e2eb5990a1.png)|
| The proposed model and pipeline architecture demonstrate real-time inference speed on mobile GPUs with high prediction quality. | 我们提出的模型和流程架构在移动GPU上展现了实时推理速度，并且具有高质量的预测效果。 |
| MediaPipe Hands is open sourced at https://mediapipe.dev. | MediaPipe Hands已在https://mediapipe.dev开源。 |


### 1. Introduction 

| **原文** | **翻译** |
| --- | --- |
| Hand tracking is a vital component to provide a natural way for interaction and communication in AR/VR, and has been an active research topic in the industry [2] [15]. | 手部追踪是为AR/VR提供自然交互和沟通方式的关键组件，并且一直是业界的一个活跃研究课题[2][15]。 |
|[2] *Facebook. Oculus Quest Hand Tracking.* |![](https://files.mdnice.com/user/1474/e88e7536-7323-4b31-8699-c06c2f5230f9.png)|
|[15] *Snapchat. Lens Studio by Snap Inc.*|![](https://files.mdnice.com/user/1474/c8d00c68-2b1e-4e41-a063-b525fd890ed7.png)|
| Vision-based hand pose estimation has been studied for many years. | 基于视觉的手部姿态估计已被研究多年。 |
| A large portion of previous work requires specialized hardware, e.g. depth sensors [13][16][17][3][4]. | 很多之前的研究工作需要专门的硬件设备，如深度传感器[13][16][17][3][4]。 |
|[13] *Efficient model-based 3d tracking of hand articulations using kinect*. (2011) |![](https://files.mdnice.com/user/1474/1335cb29-7378-4684-84ed-8d6b0d270811.png)|
|[16] *Robust articulated-icp for real-time hand tracking*. (2015)| ![](https://files.mdnice.com/user/1474/2248b943-a57a-4783-9856-de5002b5dc8a.png)|
|[17] *Self-supervised 3d hand pose estimation through training by fitting*. (2019)|![](https://files.mdnice.com/user/1474/2174d94b-7ccd-4cef-ae43-2533889ade9a.png)|
|[3][4] *Robust 3d hand pose estimation in single depth images: from single-view cnn to multi-view cnns*. (2016)| ![](https://files.mdnice.com/user/1474/0589d9d5-4ce4-4844-bb11-276380933e3b.png)|
| Other solutions are not lightweight enough to run real-time on commodity mobile devices[5] and thus are limited to platforms equipped with powerful processors. | 其他一些解决方案由于不够轻量化，无法在普通移动设备上实现实时运行，因此仅限于配备强大处理器的平台使用[5]。 |
|[5] *3d hand shape and pose estimation from a single rgb image*. (2019)|![](https://files.mdnice.com/user/1474/4fa1b561-6170-4ed7-be57-b9f4e4516aea.png)|
| In this paper, we propose a novel solution that does not require any additional hardware and performs in real-time on mobile devices. | 本文提出了一种全新的解决方案，该方案无需额外的硬件支持，并且可以在移动设备上实现实时运行。 |
| Our main contributions are: | 我们的主要贡献包括： |
| • An efficient two-stage hand tracking pipeline that can track multiple hands in real-time on mobile devices. | • 一个高效的两阶段手部追踪流程，能够在移动设备上实时追踪多只手。 |
| • A hand pose estimation model that is capable of predicting 2.5D hand pose with only RGB input. | • 一个手部姿态估计模型，仅依靠RGB输入即可预测2.5D手部姿态。 |
| • And open source hand tracking pipeline as a ready-to-go solution on a variety of platforms, including Android, iOS, Web (Tensorflow.js[7]) and desktop PCs. | • 开源的手部追踪流程，作为可直接应用于多种平台的解决方案，包括Android、iOS、Web（Tensorflow.js[7]）和桌面PC。 |
|[7] Google. Tensorflow.js Handpose.|![https://blog.tensorflow.org/2020/03/face-and-hand-tracking-in-browser-with-mediapipe-and-tensorflowjs.html](https://files.mdnice.com/user/1474/63f59b84-40f7-478a-9045-f2d36ce908cd.png)|

![](https://files.mdnice.com/user/1474/04acb648-62d7-45db-8aab-675b4623e7f0.png)

| Figure 1: Rendered hand tracking result. | 图1：手部追踪结果渲染图。 |
| --- | --- |
| (Left): Hand landmarks with relative depth presented in different shades. | （左）：显示相对深度的手部关键点，不同的阴影代表不同的深度。 |
| The lighter and larger the circle, the closer the landmark is towards the camera. | 圆圈越亮、越大，表示关键点越靠近摄像头。 |
| (Right): Real-time multi-hand tracking on Pixel 3. | （右）：在Pixel 3上实现的实时多手追踪。 |



### 2. Architecture
| **原文** | **翻译** |
| --- | --- |
| Our hand tracking solution utilizes an ML pipeline consisting of two models working together: | 我们的手部追踪解决方案采用了一个由两个模型协同工作的机器学习流程： |
| • A palm detector that operates on a full input image and locates palms via an oriented hand bounding box. | • 一个手掌检测器，它在整个输入图像上运行，通过一个定向的手部边界框定位手掌。 |
| • A hand landmark model that operates on the cropped hand bounding box provided by the palm detector and returns high-fidelity 2.5D landmarks. | • 一个手部关键点模型，它在手掌检测器提供的裁剪后的手部边界框上运行，并返回高保真的2.5D关键点。 |
| Providing the accurately cropped palm image to the hand landmark model drastically reduces the need for data augmentation (e.g. rotations, translation and scale) and allows the network to dedicate most of its capacity towards landmark localization accuracy. | 将准确裁剪的手掌图像提供给手部关键点模型，大大减少了数据增强（例如旋转、平移和缩放）的需求，并使网络能够将大部分容量用于提高关键点定位的准确性。 |
| In a real-time tracking scenario, we derive a bounding box from the landmark prediction of the previous frame as input for the current frame, thus avoiding applying the detector on every frame. | 在实时追踪场景中，我们从上一帧的关键点预测中推导出一个边界框，作为当前帧的输入，从而避免了在每一帧都应用检测器的必要。 |
| Instead, the detector is only applied on the first frame or when the hand prediction indicates that the hand is lost. | 检测器仅在第一帧或当手部预测显示手部丢失时才会被应用。 |


#### 2.1. BlazePalm Detector

| **原文** | **翻译** |
| --- | --- |
| To detect initial hand locations, we employ a single-shot detector model optimized for mobile real-time application similar to BlazeFace[1], which is also available in MediaPipe[12]. | 为了检测初始手部位置，我们采用了一种单次检测器模型，该模型经过优化，适用于移动设备的实时应用，类似于BlazeFace[1]，并且也可在MediaPipe[12]中使用。 |
|[1] *Blazeface: Sub-millisecond neural face detection on mobile gpus*. (2019)|![](https://files.mdnice.com/user/1474/ddf936c4-7551-4e88-a8d3-fd8799805712.png) ![](https://files.mdnice.com/user/1474/1511625c-097b-46e6-98f7-87e5d3fd543e.png)|
| Detecting hands is a decidedly complex task: our model has to work across a variety of hand sizes with a large scale span (∼20x) and be able to detect occluded and self-occluded hands. | 手部检测是一项非常复杂的任务：我们的模型必须能够处理各种手部尺寸，具有大范围的比例变化（约20倍），并且能够检测到被遮挡和自遮挡的手部。 |
| Whereas faces have high contrast patterns, e.g., around the eye and mouth region, the lack of such features in hands makes it comparatively difficult to detect them reliably from their visual features alone. | 尽管人脸具有高对比度的特征，例如眼睛和嘴巴区域的特征，但手部缺乏此类特征，这使得仅通过视觉特征可靠地检测手部变得相对困难。 |
| Our solution addresses the above challenges using different strategies. | 我们的解决方案采用不同的策略来应对上述挑战。 |
| First, we train a palm detector instead of a hand detector, since estimating bounding boxes of rigid objects like palms and fists is significantly simpler than detecting hands with articulated fingers. | 首先，我们训练了一个手掌检测器，而不是手部检测器，因为估计手掌和拳头等刚性物体的边界框比检测带有关节手指的手部要简单得多。 |
| In addition, as palms are smaller objects, the non-maximum suppression algorithm works well even for the two-hand self-occlusion cases, like handshakes. | 此外，由于手掌是较小的物体，即使在双手自遮挡的情况下（如握手），非极大值抑制算法也能很好地工作。 |
| Moreover, palms can be modelled using only square bounding boxes [11], ignoring other aspect ratios, and therefore reducing the number of anchors by a factor of 3∼5. | 更重要的是，手掌可以仅使用方形边界框进行建模[11]，忽略其他纵横比，从而将锚点数量减少3至5倍。 |
|[11] *SSD: single shot multibox detector*. (2015)|![](https://files.mdnice.com/user/1474/398e5b1a-54bc-4b87-bc77-05b1786180cf.png) ![](https://files.mdnice.com/user/1474/2d6c9750-8d36-4eb9-add9-c7843ba6d6d8.png)|
| Second, we use an encoder-decoder feature extractor similar to FPN[9] for a larger scene-context awareness even for small objects. | 其次，我们使用了类似于FPN[9]的编码器-解码器特征提取器，即使是小物体也能够获得更大的场景上下文感知。 |
|[9] *Feature Pyramid Networks for object detection*. (2016)|![](https://files.mdnice.com/user/1474/2b60db03-32f5-49c4-bdd1-68b0a4b07ca1.png)|
| Lastly, we minimize the focal loss[10] during training to support a large amount of anchors resulting from the high scale variance. | 最后，我们在训练过程中最小化了焦点损失[10]，以支持由大比例差异导致的大量锚点。 |
|[10] *Focal loss for dense object detection*. (2017) |![](https://files.mdnice.com/user/1474/90544aaa-53f5-47e0-9479-90eabee08fd6.png)|

![Figure 2: High-level Palm detector model architecture.](https://files.mdnice.com/user/1474/853b42cc-b286-4a53-8683-9e3bfa42b06d.png)
|High-level palm detector architecture is shown in Figure 2. |高层次的手掌检测器模型架构如图2所示。|
| --- | --- |
| We present an ablation study of our design elements in Table 1. | 我们在表1中展示了我们的设计元素的消融研究。 |


![Table 1: Ablation study of palm detector design elements of
palm detector.](https://files.mdnice.com/user/1474/e67ac7c9-5231-47a9-8c30-ddd36f60d8eb.png)

#### 2.2. Hand Landmark Model 

| **原文** | **翻译** |
| --- | --- |
| After running palm detection over the whole image, our subsequent hand landmark model performs precise landmark localization of 21 2.5D coordinates inside the detected hand regions via regression. | 在对整个图像进行手掌检测后，我们的手部关键点模型通过回归方法，在检测到的手部区域内精确定位21个2.5D坐标的关键点。 |
| The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions. | 该模型学习了一种一致的内部手部姿态表示，即使是部分可见的手部和自遮挡情况下也具有很强的鲁棒性。 |

![](https://files.mdnice.com/user/1474/bd08d7bf-5a5d-48cd-ac2e-aa3b1de3f936.png)


| Figure 3: Architecture of our hand landmark model. The model has three outputs sharing a feature extractor. Each head is trained by correspondent datasets marked in the same color: | 图 3：手部关键点模型的结构。该模型有三个输出，共享一个特征提取器。每个输出都由标有相同颜色的相应数据集进行训练: |
| --- | --- |
| 1. 21 hand landmarks consisting of x, y, and relative depth. | 1. 21个手部关键点，包括x、y坐标和相对深度。 |
| 2. A hand flag indicating the probability of hand presence in the input image. | 2. 一个手部标志，指示输入图像中是否存在手的概率。 |
| 3. A binary classification of handedness, e.g. left or right hand. | 3. 一种左右手的二元分类，例如左手或右手。 |
| We use the same topology as [14] for the 21 landmarks. | 我们对这21个关键点使用了与[14]相同的拓扑结构。 |
|[14] *Hand keypoint detection in single images using multiview bootstrapping*. (2017)|![](https://files.mdnice.com/user/1474/918ee599-52e1-4dc6-b641-5ceff6811479.png)![](https://files.mdnice.com/user/1474/2c1b60dd-4fd9-4d0d-9470-056256d9e07b.png)|
| The 2D coordinates are learned from both real-world images as well as synthetic datasets as discussed below, with the relative depth w.r.t. the wrist point being learned only from synthetic images. | 2D坐标通过真实世界图像和下文讨论的合成数据集学习，而相对于手腕点的相对深度则仅从合成图像中学习。 |
| To recover from tracking failure, we developed another output of the model similar to [8] for producing the probability of the event that a reasonably aligned hand is indeed present in the provided crop. | 为了从追踪失败中恢复，我们开发了模型的另一个输出，类似于[8]，用于生成提供的裁剪图像中合理对齐的手部确实存在的概率。 |
|[8] *Real-time facial surface geometry from monocular video on mobile gpus*. (2019) |![](https://files.mdnice.com/user/1474/25bbe172-f7e7-4a75-b26d-ce55d63b38bc.png) ![](https://files.mdnice.com/user/1474/bc5effc1-54b6-4f73-85d3-3e5e47edbba5.png)|
| If the score is lower than a threshold then the detector is triggered to reset tracking. | 如果得分低于阈值，则触发检测器重置追踪。 |
| Handedness is another important attribute for effective interaction using hands in AR/VR. | 左右手分类是手部在AR/VR中有效交互的另一个重要属性。 |
| This is especially useful for some applications where each hand is associated with a unique functionality. | 这对于一些应用程序尤其有用，因为每只手都与独特的功能相关联。 |
| Thus we developed a binary classification head to predict whether the input hand is the left or right hand. | 因此，我们开发了一个二元分类头，用于预测输入的手是左手还是右手。 |
| Our setup targets real-time mobile GPU inference, but we have also designed lighter and heavier versions of the model to address CPU inference on the mobile devices lacking proper GPU support and higher accuracy requirements of accuracy to run on desktop, respectively. | 我们的设置目标是实时的移动GPU推理，但我们还设计了更轻量和更重型的模型版本，分别用于应对缺乏适当GPU支持的移动设备上的CPU推理，以及在桌面设备上运行所需的更高精度要求。 |

### 3. Dataset and Annotation

| **原文** | **翻译** |
| --- | --- |
| To obtain ground truth data, we created the following datasets addressing different aspects of the problem: | 为了获取真实标签数据，我们创建了以下针对问题不同方面的数据集： |
| • In-the-wild dataset: This dataset contains 6K images of large variety, e.g. geographical diversity, various lighting conditions and hand appearance. The limitation of this dataset is that it doesn’t contain complex articulation of hands. | • 实地数据集：该数据集包含6K张图像，具有很大的多样性，例如地理多样性、不同的光照条件和手部外观。该数据集的局限性在于它不包含复杂的手部关节动作。 |
| • In-house collected gesture dataset: This dataset contains 10K images that cover various angles of all physically possible hand gestures. The limitation of this dataset is that it’s collected from only 30 people with limited variation in background. The in-the-wild and in-house dataset are great complements to each other to improve robustness. | • 内部收集的手势数据集：该数据集包含10K张图像，覆盖了从各个角度捕捉的所有物理上可能的手势。该数据集的局限性在于它仅从30人中收集，背景变化有限。实地数据集和内部数据集在提高鲁棒性方面是很好的互补。 |
| • Synthetic dataset: To even better cover the possible hand poses and provide additional supervision for depth, we render a high-quality synthetic hand model over various backgrounds and map it to the corresponding 3D coordinates. We use a commercial 3D hand model that is rigged with 24 bones and includes 36 blendshapes, which control fingers and palm thickness. The model also provides 5 textures with different skin tones. We created video sequences of transformation between hand poses and sampled 100K images from the videos. We rendered each pose with a random high-dynamic-range lighting environment and three different cameras. See Figure 4 for examples. | • 合成数据集：为了更好地覆盖可能的手部姿势并为深度提供额外监督，我们在各种背景上渲染了高质量的合成手部模型，并将其映射到对应的3D坐标。我们使用了一个商业3D手部模型，该模型具有24根骨骼和36个混合形状，用于控制手指和手掌厚度。该模型还提供了5种不同肤色的纹理。我们创建了手部姿势之间变换的视频序列，并从视频中抽取了100K张图像。我们使用随机的高动态范围光照环境和三台不同的摄像机渲染了每个姿势。示例见图4。 |


![](https://files.mdnice.com/user/1474/000e64cc-85ce-493b-bf23-a1fe696fa9cb.png)
| Figure 4: Examples of our datasets. (Top): Annotated real-world images.(Bottom): Rendered synthetic hand images with ground truth annotation.  | 图4：我们数据集的示例。（上）：标注的真实世界图像。（下）：带有真实标签标注的渲染合成手部图像。  |
| --- | --- |
| For the palm detector, we only use in-the-wild dataset, which is sufficient for localizing hands and offers the highest variety in appearance. | 对于手掌检测器，我们只使用实地数据集，它足以定位手部，并提供了外观上的最大多样性。 |
| However, all datasets are used for training the hand landmark model. | 然而，所有数据集都用于训练手部关键点模型。 |
| We annotate the real-world images with 21 landmarks and use projected ground-truth 3D joints for synthetic images. | 我们在真实世界图像上标注了21个关键点，并在合成图像上使用了投影的真实3D关节点。 |
| For hand presence, we select a subset of real-world images as positive examples and sample on the region excluding annotated hand regions as negative examples. | 对于手部存在性，我们选择了一部分真实世界图像作为正例，并在排除标注手部区域的区域内进行采样，作为负例。 |
| For handedness, we annotate a subset of real-world images with handedness to provide such data. | 对于左右手分类，我们在一部分真实世界图像上标注了左右手属性，以提供此类数据。 |


### 4. Results 结果

![Table 2: Results of our model trained from different
datasets.](https://files.mdnice.com/user/1474/84b7c27f-2381-4c7b-b2fb-f7cca2f47e78.png)

| **原文** | **翻译** |
| --- | --- |
| For the hand landmark model, our experiments show that the combination of real-world and synthetic datasets provides the best results. See Table 2 for details.| 对于手部关键点模型，我们的实验表明，结合真实世界和合成数据集能够提供最佳效果。详细信息请参见表2。  |
| We evaluate only on real-world images. | 我们仅在真实世界图像上进行了评估。 |
| Beyond the quality improvement, training with a large synthetic dataset leads to less jitter visually across frames. | 除了质量提升之外，使用大规模合成数据集进行训练还能减少帧间的视觉抖动。 |
| This observation leads us to believe that our real-world dataset can be enlarged for better generalization. | 这一观察结果使我们相信，扩展我们的真实世界数据集可以获得更好的泛化能力。 |
| Our target is to achieve real-time performance on mobile devices. | 我们的目标是在移动设备上实现实时性能。 |
| We experimented with different model sizes and found that the “Full” model (see Table 3) provides a good trade-off between quality and speed. | 我们尝试了不同的模型规模，发现“完整版”模型（见表3）在质量和速度之间提供了良好的平衡。 |
| Increasing model capacity further introduces only minor improvements in quality but decreases significantly in speed (see Table 3 for details). | 增加模型容量只会带来微小的质量提升，但会显著降低速度（详情见表3）。 |

![](https://files.mdnice.com/user/1474/1939a184-57f0-40f7-a34b-a602971fd804.png)
|Table 3: Hand landmark model performance characteristics. |表 3：手部关键点模型的性能特征。|
| --- | --- |
| We use the TensorFlow Lite GPU backend for on-device inference[6]. | 我们使用TensorFlow Lite GPU后端进行设备端推理[6]。 |
|[6] *Google. Tensorflow lite on GPU*.|![](https://files.mdnice.com/user/1474/9543cf01-ebce-43dc-bbaa-90895de44616.png)|




### 5. Implementation 实现
| **原文** | **翻译** |
| --- | --- |
| With MediaPipe[12], our hand tracking pipeline can be built as a directed graph of modular components, called Calculators. | 在MediaPipe[12]中，我们的手部追踪流程可以构建为一个由模块化组件组成的有向图，这些组件被称为计算器（Calculators）。 |
| Mediapipe comes with an extensible set of Calculators to solve tasks like model inference, media processing, and data transformations across a wide variety of devices and platforms. | MediaPipe提供了一组可扩展的计算器，用于解决模型推理、媒体处理和数据转换等任务，适用于各种设备和平台。 |
| Individual Calculators like cropping, rendering and neural network computations are further optimized to utilize GPU acceleration. | 个别计算器，如裁剪、渲染和神经网络计算，经过进一步优化以利用GPU加速。 |
| For example, we employ TFLite GPU inference on most modern phones. | 例如，我们在大多数现代手机上使用TFLite GPU推理。 |
| Our MediaPipe graph for hand tracking is shown in Figure 5. | 我们的MediaPipe手部追踪图如图5所示。 |

![](https://files.mdnice.com/user/1474/acf2dd0b-3897-489f-baa9-08300f753937.png)
| Figure 5: The hand landmark models output controls when the hand detection model is triggered. This behavior is achieved by MediaPipe's powerful synchronization building blocks, resulting in high performance and optimal throughput of the ML pipeline. | 图5：手部关键点模型的输出控制何时触发手部检测模型。 这种行为是通过MediaPipe强大的同步构件实现的，从而带来了高性能和机器学习流程的最佳吞吐量。 |
| --- | --- |
| The graph consists of two subgraphs, one for hand detection and another for landmarks computation. | 该图由两个子图组成，一个用于手部检测，另一个用于关键点计算。 |
| One key optimization MediaPipe provides is that the palm detector only runs as needed (fairly infrequently), saving significant computation. | MediaPipe提供的一个关键优化是手掌检测器仅在需要时运行（相当不频繁），从而节省了大量计算。 |
| We achieve this by deriving the hand location in the current video frames from the computed hand landmarks in the previous frame, eliminating the need to apply the palm detector on every frame. | 我们通过从上一帧计算的手部关键点推导当前视频帧中的手部位置，避免了在每一帧上都应用手掌检测器的必要性。 |
| For robustness, the hand tracker model also outputs an additional scalar capturing the confidence that a hand is present and reasonably aligned in the input crop. | 为了增强鲁棒性，手部追踪模型还输出一个额外的标量，用于捕捉输入裁剪图像中手部存在且合理对齐的置信度。 |
| Only when the confidence falls below a certain threshold is the hand detection model reapplied to the next frame. | 只有当置信度低于某一阈值时，手部检测模型才会重新应用到下一帧。 |

### 6. Application examples 应用
| **原文** | **翻译** |
| --- | --- |
| Our hand tracking solution can readily be used in many applications such as gesture recognition and AR effects. | 我们的手部追踪解决方案可以轻松应用于许多领域，例如手势识别和增强现实（AR）效果。 |
| On top of the predicted hand skeleton, we employ a simple algorithm to compute gestures, see Figure 6. | 基于预测的手部骨架，我们采用了一个简单的算法来计算手势，见图6。 |


![](https://files.mdnice.com/user/1474/aebc48dd-c9bf-435e-9784-7e613466a55a.png)


| Figure 6: Screenshots of real-time gesture recognition. Semantics of gestures are rendered at top of the images. | 图 6：实时手势识别截图。手势的语义显示在图片顶部。 |
| --- | --- |
| First, the state of each finger, e.g. bent or straight, is determined via the accumulated angles of joints. | 首先，通过关节的累计角度来确定每个手指的状态，例如弯曲或伸直。 |
| Then, we map the set of finger states to a set of predefined gestures. | 然后，我们将手指状态映射到一组预定义的手势中。 |
| This straightforward, yet effective technique allows us to estimate basic static gestures with reasonable quality. | 这种简单但有效的技术使我们能够以合理的质量估计基本的静态手势。 |
| Beyond static gesture recognition, it is also possible to use a sequence of landmarks to predict dynamic gestures. | 除了静态手势识别，还可以使用一系列关键点来预测动态手势。 |
| Another application is to apply AR effects on top of the skeleton. | 另一个应用是在骨架上应用增强现实（AR）效果。 |
| Hand based AR effects currently enjoy high popularity. | 基于手部的增强现实效果目前非常流行。 |
| In Figure 7, we show an example AR rendering of the hand skeleton in neon light style. | 在图7中，我们展示了一个以霓虹灯风格渲染的手部骨架增强现实效果的示例。 |

![Figure 7: Example of real-time AR effects based on our predicted hand skeleton.](https://files.mdnice.com/user/1474/cfe625f8-9cc8-4b9c-9dcd-8f365791fbf8.png)

### 7. Conclusion 结论
| **原文** | **翻译** |
| --- | --- |
| In this paper, we proposed MediaPipe Hands, an end-to-end hand tracking solution that achieves real-time performance on multiple platforms. | 在本文中，我们提出了MediaPipe Hands，这是一种端到端的手部追踪解决方案，能够在多个平台上实现实时性能。 |
| Our pipeline predicts 2.5D landmarks without any specialized hardware and thus, can be easily deployed to commodity devices. | 我们的流程能够预测2.5D关键点，而无需任何专用硬件，因此可以轻松部署到普通设备上。 |
| We open sourced the pipeline to encourage researchers and engineers to build gesture control and creative AR/VR applications with our pipeline. | 我们将该流程开源，以鼓励研究人员和工程师使用我们的流程构建手势控制和创意AR/VR应用程序。 |
