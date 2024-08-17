## On-device Real-time Hand Gesture Recognition

For our case, we **define two virtual keypoints to describe hand center, scale, and rotation angle**: a **center keypoint** and an **alignment keypoint**. 
对于我们的情况，我们**定义了两个虚拟关键点来描述手的中心、比例和旋转角度**：一个**中心关键点**和一个**对齐关键点**。
- The **center keypoint** is estimated as **the average of the index, middle, and pinky knuckles**.  
**中心关键点**是估计为**食指、中指和小指的关节的平均值**。
- The **alignment keypoint** location is estimated so it **forms the rotation/scale vector with the center keypoint**.
**对齐关键点**的位置是估计的，因此它**与中心关键点形成旋转/比例向量**。

![image](https://github.com/yiyangd/Paper-Reading-Notes/assets/25696979/1205ff58-c7f7-4e5b-9045-fd52048649e8)

- The **rotation angle** is estimated from **a sum of two vectors**: from the **middle base knuckle to the wrist**, and from the **index to the pinky base knuckle**.   
**旋转角度**是**由两个向量的总和估计的**：从**中指基关节到手腕**，以及**从食指到小指基关节**。
  - As the **component vectors tend to be orthogonal in the majority of cases**; the **resulting sum vector changes smoothly for any hand pose**, and never degrades to zero, as shown on Figure 2.  
由于**组件向量在大多数情况下都趋于正交**; 结果的**总和向量对于任何手势都会平滑变化**，并且从未降低到零，如图2所示。
- This increases overall hand tracking quality for **frontal hand cases**. The **scalar value of the alignment vector** is estimated as **the distance from the center keypoint to the farthest knuckle of the same hand**.  
这增加了**正面手势**的整体手部追踪质量。**对齐向量的标量值**估计为**从中心关键点到同一只手的最远的关节的距离**。
- The new rotation and scale normalization results in a significant quality boost for the whole hand pose estimation pipeline: 71.3 mAP vs 66.5 mAP (for the original MediaPipe Hands [2] pipeline) on our validation dataset with complex ASL hand poses.
新的旋转和比例归一化为整个手势估计流程带来了显著的质量提升：在我们的验证数据集上，带有复杂ASL手势的71.3 mAP vs 66.5 mAP（对于原始的 MediaPipe Hands[2]流程）。
