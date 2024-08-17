## Abstract

| **Abstract** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| Computer sensing of hand and limb motion is an important problem for applications in Human-Computer Interaction (HCI), virtual reality, and athletic performance measurement. | 手和肢体运动的计算机感知是人机交互（HCI）、虚拟现实和运动表现测量应用中的一个重要问题。 | 手部和肢体运动的计算机感知在人机交互（HCI）、虚拟现实以及运动表现测量中的应用中是一个重要问题。 |
| Commercially available sensors are invasive, and require the user to wear gloves or targets. | 商用传感器具有接触性，并要求用户佩戴手套或标靶。 | 市售的传感器具有接触性，并要求用户佩戴手套或标记。 |
| We have developed a noninvasive vision-based hand tracking system, called DigitEyes. | 我们开发了一种名为DigitEyes的非侵入式基于视觉的手部追踪系统。 | 我们开发了一种名为DigitEyes的非侵入式视觉手部跟踪系统。 |
| Employing a kinematic hand model, the DigitEyes system has demonstrated tracking performance at speeds of up to 10 Hz, using line and point features extracted from gray scale images of unadorned, unmarked hands. | 该系统采用运动学手部模型，通过从无修饰、无标记手部的灰度图像中提取的线条和点特征，实现了高达10Hz的跟踪性能。 | DigitEyes系统采用运动学手模型，通过从无修饰、无标记的手部灰度图像中提取的线条和点特征，展示了高达10Hz的跟踪性能。 |
| We describe an application of our sensor to a 3D mouse user-interface problem. | 我们描述了该传感器在3D鼠标用户界面问题中的应用。 | 我们介绍了将该传感器应用于3D鼠标用户界面问题的案例。 |

## Introduction


| **Paragraph 1** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| A "human sensor" capable of tracking a person's spatial motion using techniques from Computer Vision would be a powerful tool for human-computer interfaces. | 一种能够使用计算机视觉技术追踪人类空间运动的“人类传感器”将成为人机界面的有力工具。 | 一种能够利用计算机视觉技术追踪人类空间运动的“人类传感器”将是人机界面的强大工具。 |
| Such a sensor could be located in the user's environment (rather than on their person) and could operate under natural conditions of lighting and dress, providing a degree of convenience and flexibility that is currently unavailable. | 这种传感器可以放置在用户的环境中（而不是佩戴在身上），并且能够在自然的光照和穿着条件下运行，提供目前尚不可得的便利性和灵活性。 | 这种传感器可以安装在用户的环境中（而非佩戴在身上），并能在自然的光照和穿着条件下运行，提供目前尚未实现的便利性和灵活性。 |
| For the purpose of visual sensing, human hands and limbs can be modeled as articulated mechanisms, systems of rigid bodies connected together by joints with one or more degrees of freedom (DOF's). | 在视觉感知的目的下，人类的手和肢体可以被建模为铰接机制，即由具有一个或多个自由度（DOF）的关节连接在一起的刚体系统。 | 为了视觉感知的目的，人的手和肢体可以被建模为铰接机制，即由一个或多个自由度（DOF）的关节连接的刚体系统。 |
| This model can be applied at a fine (visual) scale to describe hand motion, and at a coarser scale to describe the motion of the entire body. | 这种模型可以在细微（视觉）尺度上用于描述手部运动，在较粗尺度上用于描述整个身体的运动。 | 该模型可以在细致（视觉）尺度上用于描述手部运动，在较粗尺度上用于描述整个身体的运动。 |
| Based on this observation, we formulate human sensing as the real-time visual tracking of articulated kinematic chains. | 基于这一观察，我们将人类感知表述为对铰接运动链的实时视觉追踪。 | 基于这一观察，我们将人类感知定义为铰接运动链的实时视觉跟踪。 |


| **Paragraph 2** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| Although many frameworks for human motion analysis are possible, our approach has four main advantages. | 尽管人类运动分析有许多可能的框架，我们的方法有四个主要优点。 | 尽管存在许多人类运动分析的框架，我们的方法具有四个主要优势。 |
| First, by tracking all of the hand's DOF's, we provide the user with maximum flexibility for interface applications. | 首先，通过跟踪手的所有自由度（DOF），我们为用户提供了最大限度的界面应用灵活性。 | 首先，通过跟踪手的所有自由度（DOF），我们为用户提供了界面应用的最大灵活性。 |
| (See [15, 6] for examples of interfaces requiring a whole-hand sensor.) | （参见[15, 6]，了解需要整个手部传感器的界面示例。） | （参见[15, 6]，了解需要全手传感器的界面示例。） |
| In addition, our general modeling approach based on 3D kinematics makes it possible to track any subset of hand or body states with the same basic algorithm. | 此外，我们基于3D运动学的通用建模方法使得能够使用相同的基本算法跟踪手或身体状态的任何子集。 | 此外，我们基于3D运动学的通用建模方法使得能够通过相同的基本算法跟踪手或身体状态的任何子集。 |
| Another benefit of full state tracking is invariance to unused hand motions. | 全状态跟踪的另一个好处是对未使用手部运动的无关性。 | 全状态跟踪的另一个优势是对未使用的手部运动具有不变性。 |
| The motion of a particular finger, for example, can be recognized from its joint angles regardless of the pose of the palm relative to the camera. | 例如，特定手指的运动可以通过其关节角度来识别，而不考虑手掌相对于相机的姿态。 | 例如，特定手指的运动可以通过其关节角度识别，而不受手掌相对于相机的姿态影响。 |
| Finally, by modeling the hand kinematics in 3D we eliminate the need for application- or viewpoint-dependent user modeling. | 最后，通过在3D中建模手部运动学，我们消除了对依赖于应用或视角的用户建模的需求。 | 最后，通过在3D中建模手部运动学，我们消除了对依赖于应用或视角的用户建模的需求。 |


| **Paragraph 3** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| The DigitEyes system treats hand tracking as a model-based sequential estimation problem: given a sequence of images and a hand model, we estimate the 3D hand configuration in each frame. | DigitEyes系统将手部跟踪视为一个基于模型的序列估计问题：给定一系列图像和一个手部模型，我们估计每一帧中的3D手部构型。 | DigitEyes系统将手部跟踪视为基于模型的序列估计问题：给定一系列图像和手部模型，我们估计每一帧中的3D手部配置。 |
| All possible hand configurations are represented by vectors in a state space, which encodes the pose of the palm (six rotation and translation DOF's) and the joint angles of the fingers (four states per finger, five for the thumb). | 所有可能的手部构型都通过状态空间中的向量表示，该空间编码了手掌的姿态（六个旋转和平移自由度）以及手指的关节角度（每根手指四个状态，拇指五个）。 | 所有可能的手部构型都由状态空间中的向量表示，状态空间编码了手掌的姿态（六个旋转和平移自由度）和手指的关节角度（每根手指四个状态，拇指五个）。 |
| Each hand configuration generates a set of image features, 2D lines and points, by projection through the camera model. | 每一个手部构型通过摄像机模型的投影生成一组图像特征，包括2D线条和点。 | 每个手部配置通过摄像机模型的投影生成一组图像特征，即2D线条和点。 |
| A feature measurement process extracts these hand features from grey-scale images by detecting the occluding boundaries of finger links and tips. | 特征测量过程通过检测手指关节和指尖的遮挡边界，从灰度图像中提取这些手部特征。 | 特征测量过程通过检测手指关节和指尖的遮挡边界，从灰度图像中提取这些手部特征。 |
| The state estimate for each image is computed by finding the state vector that best fits the measured features. | 每幅图像的状态估计通过找到最符合测量特征的状态向量来计算。 | 每张图像的状态估计通过找到最符合测量特征的状态向量来计算。 |
| Our basic tracking framework is similar to that of [4, 7, 16]. | 我们的基本跟踪框架与[4, 7, 16]的框架相似。 | 我们的基本跟踪框架与[4, 7, 16]中的框架相似。 |


| **Paragraph 4** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| Articulated mechanisms are more difficult to track than the single rigid objects traditionally addressed in Computer Vision. | 铰接机制比计算机视觉中传统处理的单一刚体更难跟踪。 | 铰接机构比计算机视觉中传统处理的单一刚体更难跟踪。 |
| Three major difficulties are the large size of the state space, nonlinearities in the state-to-feature mapping (called the measurement model), and self-occlusions. | 三个主要困难是状态空间的庞大规模、状态到特征映射的非线性（称为测量模型）以及自遮挡。 | 主要困难包括状态空间的庞大规模、状态到特征映射的非线性（即测量模型），以及自遮挡问题。 |
| Finger articulations add an additional 21 states over the rigid body motion of the palm, significantly increasing the computational cost of estimation. | 手指的关节运动为手掌的刚体运动增加了额外的21个状态，显著增加了估计的计算成本。 | 手指的关节运动为手掌的刚体运动增加了额外的21个状态，显著增加了估计的计算成本。 |
| These additional states are parameterized by joint angles, which introduce nonlinearities and kinematic singularities into the measurement model. | 这些额外的状态由关节角度参数化，给测量模型引入了非线性和运动学奇异性。 | 这些额外的状态通过关节角度进行参数化，从而在测量模型中引入了非线性和运动学奇异性。 |
| Singularities arise when a small change in a given state has no effect on the image features. | 当某一状态的微小变化对图像特征没有影响时，就会出现奇异性。 | 当某一状态的微小变化对图像特征没有影响时，就会出现奇异性。 |
| In addition to these problems, the fingers occlude each other and the palm during motion, making feature measurement difficult. | 除了这些问题外，手指在运动过程中会相互遮挡和遮挡手掌，使得特征测量更加困难。 | 除了这些问题外，手指在运动过程中会相互遮挡，并遮挡手掌，使得特征测量更加困难。 |

| **Paragraph 5** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| The DigitEyes system uses local search and linearization to deal with the large state space and nonlinear measurement model. | DigitEyes系统使用局部搜索和线性化来处理大型状态空间和非线性测量模型。 | DigitEyes系统使用局部搜索和线性化来处理大型状态空间和非线性测量模型。 |
| The key to our local, gradient-based approach to tracking is a high image acquisition rate (10 Hz), which limits the change in the hand state, and therefore image feature location, between frames. | 我们基于梯度的局部跟踪方法的关键在于高图像采集率（10 Hz），这限制了帧间手部状态的变化，从而限制了图像特征位置的变化。 | 我们基于梯度的局部跟踪方法的关键在于高图像采集率（10 Hz），这限制了帧间手部状态的变化，从而限制了图像特征位置的变化。 |
| In the state space, we exploit this locality by linearizing the nonlinear state model around the previous estimate. | 在状态空间中，我们通过围绕先前的估计线性化非线性状态模型来利用这一局部性。 | 在状态空间中，我们通过围绕先前估计对非线性状态模型进行线性化来利用这一局部性。 |
| Techniques from robotics provide for fast computation of the necessary kinematic Jacobian. | 来自机器人学的技术提供了必要的运动学雅可比矩阵的快速计算方法。 | 机器人技术提供了必要的运动学雅可比矩阵的快速计算方法。 |
| Kinematic singularities are dealt with by stabilizing the state estimator. | 通过稳定状态估计器来处理运动学奇异性。 | 运动学奇异性通过稳定状态估计器来处理。 |
| The resulting linear estimation problem is solved for each frame, producing a sequence of state corrections which are integrated over time to yield an estimated state trajectory. | 产生的线性估计问题在每一帧中解决，生成的状态校正序列随时间整合，以产生估计的状态轨迹。 | 产生的线性估计问题在每帧中得到解决，生成的状态校正序列随时间整合，最终得出估计的状态轨迹。 |



| **Paragraph 6** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| As a result of the high image sampling rate, the change in hand features between frames is also small. | 由于高图像采样率，帧间手部特征的变化也很小。 | 由于图像采样率高，帧间手部特征的变化也很小。 |
| For a given image, the state estimate from the previous frame is used to predict feature positions. | 对于给定的图像，使用前一帧的状态估计来预测特征位置。 | 对于给定的图像，前一帧的状态估计用于预测特征位置。 |
| Feature detectors, initialized to these predictions, exploit the symmetry of the finger links to extract lines and points and match them to the hand model. | 特征检测器以这些预测为初始，利用手指关节的对称性提取线条和点，并将它们与手部模型匹配。 | 特征检测器根据这些预测初始化，利用手指关节的对称性提取线条和点，并将其与手部模型匹配。 |
| In the first image of the sequence, the user places his hand in a known starting configuration to initialize tracking. | 在序列的第一张图像中，用户将手放置在已知的起始构型中以初始化跟踪。 | 在序列的第一张图像中，用户将手放在已知的起始配置中以初始化跟踪。 |
| In the current system, each finger link detects its features independently of the others, which limits the sensor to hand motions without occlusions. | 在当前系统中，每个手指关节独立于其他关节检测其特征，这限制了传感器只能用于无遮挡的手部运动。 | 在当前系统中，每个手指关节独立检测其特征，这限制了传感器只能用于无遮挡的手部运动。 |
| We are extending our feature processing approach to remove this limitation. | 我们正在扩展特征处理方法以消除这一限制。 | 我们正在扩展特征处理方法，以消除这一限制。 |


| **Paragraph 7** | **第一版翻译** | **第二版翻译** |
|----------|---------------|---------------|
| In [12], we described the DigitEyes system in detail, and gave results of tracking a 27 DOF hand model from a two camera image sequence under orthographic projection. | 在[12]中，我们详细描述了DigitEyes系统，并给出了在正交投影下，通过两台摄像机的图像序列跟踪一个27自由度手部模型的结果。 | 在[12]中，我们详细描述了DigitEyes系统，并给出了在正交投影下，通过双摄像机图像序列跟踪27自由度手部模型的结果。 |
| This paper describes an extension to perspective projection, and gives a detailed example of a user-interface based on our sensor: a 3D graphical mouse. | 本文描述了对透视投影的扩展，并给出了基于我们传感器的用户界面的详细示例：一个3D图形鼠标。 | 本文描述了对透视投影的扩展，并给出了一个基于我们传感器的用户界面的详细示例：一个3D图形鼠标。 |
| While difficult problems still remain in tracking through occlusions and across complicated backgrounds, these results demonstrate the potential of our approach to vision-based human motion sensing. | 尽管在通过遮挡和复杂背景进行跟踪时仍然存在困难问题，这些结果证明了我们基于视觉的人类运动感知方法的潜力。 | 尽管在遮挡和复杂背景下进行跟踪仍存在困难，但这些结果展示了我们基于视觉的人体运动感知方法的潜力。 |

