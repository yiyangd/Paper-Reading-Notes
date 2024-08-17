# MediaPipe Hands: On-device Real-time Hand Tracking
## Abstract
We present a **real-time, on-device hand tracking solution** that **predicts a human hand skeleton from a single RGB camera** for AR/VR applications. Our pipeline consists two models: 1) a palm detector that provides a bounding box of a hand and a hand landmark model that predicts the hand skeleton. This solution is implemented using MediaPipe, a framework for building cross-platform ML solutions. The proposed model and pipeline architecture exhibit real-time inference speed on mobile GPUs with high prediction quality. MediaPipe Hands is open-sourced at https://mediapipe.dev.

我们提供了一个**实时的、部署在设备上的手部追踪解决方案**，该方案**从单个RGB摄像头预测人的手骨架**，用于AR/VR应用。我们的流程包括两个模型：一个掌部检测器，它提供手的边界框；和一个手部标记模型，它预测手的骨架。这个解决方案是使用MediaPipe实现的，这是一个用于构建跨平台ML解决方案的框架。所提出的模型和流程架构在移动GPU上展示了实时推理速度，具有高预测质量。MediaPipe Hands在 https://mediapipe.dev 上开源。
## 1. Introduction
**Hand tracking** is a **vital component to provide a natural way for interaction and communication in AR/VR**, and has been an active research topic in the industry [2] [15]. **Vision-based hand pose estimation** has been studied for many years. A large portion of previous work requires specialized hardware, e.g. depth sensors [13][16][17][3][4]. Other solutions are not lightweight enough to run real-time on commodity mobile devices[5] and thus are limited to platforms equipped with powerful processors. In this paper, we propose a novel solution that does not require any additional hardware and performs in real-time on mobile devices. Our main contributions are:  
**手部追踪**是**为AR/VR提供自然交互和沟通的重要组件**，并且一直是该行业的活跃研究课题[2] [15]。**基于视觉的手势姿态估计**已经研究了很多年。之前的大部分工作需要专用的硬件，例如深度传感器[13][16][17][3][4]。其他解决方案不够轻量级，无法在通用移动设备上实时运行[5]，因此仅限于配备了强大处理器的平台。在本文中，我们提出了一种不需要任何附加硬件且可以在移动设备上实时运行的新颖解决方案。我们的主要贡献包括：
- An efficient **two-stage hand tracking pipeline** that can **track multiple hands in real-time on mobile devices**.    
一个高效的**两阶段手部追踪流程**，能够**在移动设备上实时追踪多只手**。
- A **hand pose estimation model** that is capable of **predicting 2.5D hand pose with only RGB input**.  
一个能够**仅使用RGB输入预测2.5D手势姿态**的**手势估计模型**。
- And open source hand tracking pipeline as a ready-to-go solution on a variety of platforms, including Android, iOS, Web (Tensorflow.js[7]) and desktop PC.  
并开源的手部追踪流程，作为适用于多种平台的即用型解决方案，包括Android、iOS、Web (Tensorflow.js[7])和桌面PC。

## 2. Architecture 架构
Our hand tracking solution utilizes an ML pipeline consisting of **two models working together**:  
我们的手部追踪解决方案利用了一个ML流程，包括**两个协同工作的模型**：
- A **palm detector** that operates on a full input image and **locates palms via an oriented hand bounding box**.
一个**掌部检测器**，作用于完整的输入图像，并**通过一个定向的手部边界框来定位掌部**。
- A **hand landmark model** that operates on the cropped hand bounding box provided by the palm detector and **returns high-fidelity 2.5D landmarks**.  
一个**手部标记模型**，作用于由掌部检测器提供的裁剪手部边界框，并**返回高保真度的2.5D标记**。

Providing the **accurately cropped palm image to the hand landmark model** drastically **reduces the need for data augmentation (e.g. rotations, translation and scale)** and allows the network to dedicate most of its capacity towards landmark localization accuracy. In a real-time tracking scenario, we derive a bounding box from the landmark prediction of the previous frame as input for the current frame, thus avoiding applying the detector on every frame. Instead, the detector is only applied on the first frame or when the hand prediction indicates that the hand is lost.

将**精确裁剪的掌部图像提供给手部标记模型**大大**减少了对数据增强（例如旋转、平移和缩放）的需求**，并允许网络将大部分容量专用于标记定位的准确性。在实时追踪场景中，我们从上一帧的标记预测中得出一个边界框，作为当前帧的输入，从而避免在每一帧上应用检测器。相反，检测器仅在第一帧或当手部预测指示手部丢失时应用。

<img width="283" alt="image" src="https://github.com/yiyangd/Paper-Reading-Notes/assets/25696979/5a8ce8a1-7628-4476-9a51-43d13c093c46">

