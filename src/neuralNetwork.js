import {
  Array1D,
  InCPUMemoryShuffledInputProviderBuilder,
  Graph,
  Session,
  SGDOptimizer,
  NDArrayMathGPU,
  CostReduction,
} from 'deeplearn';

// Encapsulates math operations on the CPU and GPU.
// 在CPU和GPU上封装数学运算。
const math = new NDArrayMathGPU();

class ColorAccessibilityModel {
  // Runs training.
  // 进行训练
  session;

  // An optimizer with a certain initial learning rate. Used for training.
  // 具有一定初始学习率的优化器。用于训练
  initialLearningRate = 0.06;
  optimizer;

  // Each training batch will be on this many examples.
  // 每个训练一批将变成他的实例
  batchSize = 300;

  inputTensor;
  targetTensor;
  costTensor;//声明一个成本张量。在这里，它就是均方误差。它使用训练集的目标张量（标签）和来自已训练算法的预测张量进行成本计算，以此来优化算法。
  predictionTensor;

  // Maps tensors to InputProviders.
  // 将张量映射到输入提供者
  feedEntries;

  constructor() {
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  setupSession(trainingSet) {
    //接收训练集作为参数,会话会初始化一个空的图，这个图会在后面反映出你的神经网络结构
    const graph = new Graph(); 


    //以张量的形式定义输入数据点和输出数据点的形状。
    //有三个输入单元（每种颜色通道就是一个输入单元）和两个输出单元（二元分类：白颜色或黑颜色）
    this.inputTensor = graph.placeholder('input RGB value', [3]);//3
    this.targetTensor = graph.placeholder('output RGB value', [2]);//2

    //  createFullyConnectedLayer 神经网络包含了隐藏层，这里就是施展魔法的地方。一般来说，神经网络经过训练会得到自己的计算参数。
    //  以图、变形层、新层索引和单元数量
    let fullyConnectedLayer = this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);


    //创建输出二分分类的层。它有两个输出单元，每个单元对应一个离散值
    this.predictionTensor = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 2);

    //声明一个成本张量。在这里，它就是均方误差。它使用训练集的目标张量（标签）和来自已训练算法的预测张量进行成本计算，以此来优化算法。
    this.costTensor = graph.meanSquaredCost(this.targetTensor, this.predictionTensor);


//最后，使用之前架构好的图来创建会话。接下来就可以进入训练阶段了。
    this.session = new Session(graph, math);

    this.prepareTrainingSet(trainingSet);
  }
//最后，使用之前架构好的图来创建会话。接下来就可以进入训练阶段了。
  prepareTrainingSet(trainingSet) {
    //其次，你可以将输入和输出（标签，也叫作目标）从训练集中抽取出来，把它们映射成神经网络可理解的格式。
    //deeplearn.js使用自己的NDArrays完成这些数学运算。
    //最后它们会变成多维度的矩阵或矢量。另外，输入数组中的颜色值会被规范化，用以提升神经网络的性能。
    math.scope(() => {
      const { rawInputs, rawTargets } = trainingSet;

      const inputArray = rawInputs.map(v => Array1D.new(this.normalizeColor(v)));
      const targetArray = rawTargets.map(v => Array1D.new(v));

    //输入和输出数组被搅乱。deeplearn.js提供的搅乱器（Shuffler）在搅乱数组时会保持它们之间的同步。
    //每次训练迭代都会进行搅乱操作，输入被分批填充进神经网络。
    //我们通过搅乱来改进训练算法，因为这更像是通过避免过拟合来实现泛化。
      const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([ inputArray, targetArray ]);
      const [ inputProvider, targetProvider ] = shuffledInputProviderBuilder.getInputProviders();

      // Maps tensors to InputProviders.
      // 填充项就成为神经网络训练阶段的最终输入。
      this.feedEntries = [
        { tensor: this.inputTensor, data: inputProvider },
        { tensor: this.targetTensor, data: targetProvider },
      ];
    });
  }

  train(step, computeCost) {
    // Every 50 steps, lower the learning rate by 10%.
    let learningRate = this.initialLearningRate * Math.pow(0.90, Math.floor(step / 50));
    this.optimizer.setLearningRate(learningRate);

    // Train one batch.
    let costValue;
    math.scope(() => {
      const cost = this.session.train(
        this.costTensor,
        this.feedEntries,
        this.batchSize,
        this.optimizer,
        computeCost ? CostReduction.MEAN : CostReduction.NONE,
      );

      // Compute the cost (by calling get), which requires transferring data from the GPU.
      if (computeCost) {
        costValue = cost.get();
      }
    });

    return costValue;
  }

  predict(rgb) {
    let classifier = [];

    math.scope(() => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(this.normalizeColor(rgb)),
      }];

      classifier = this.session.eval(this.predictionTensor, mapping).getValues();
    });

    return [ ...classifier ];
  }
  //神经网络包含了隐藏层，这里就是施展魔法的地方。一般来说，神经网络经过训练会得到自己的计算参数。
  //用于创建连接层的方法以图、变形层、新层索引和单元数量作为参数。图的层属性可用于返回一个带有名字的张量。
  createFullyConnectedLayer(
    graph,
    inputLayer,
    layerIndex,
    units,
    activationFunction
  ) {
    return graph.layers.dense(
      `fully_connected_${layerIndex}`,
      inputLayer,
      units,
      activationFunction
        ? activationFunction
        : (x) => graph.relu(x)
    );
  }

  normalizeColor(rgb) {
    return rgb.map(v => v / 255);
  }
}

export default ColorAccessibilityModel;