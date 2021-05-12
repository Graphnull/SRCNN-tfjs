import * as tf from '@tensorflow/tfjs-node'

export let modelL = tf.sequential();
modelL.add(tf.layers.conv2d({ filters: 128, kernelSize: [9, 9], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true, inputShape: [120,160, 1]}));
modelL.add(tf.layers.leakyReLU());
modelL.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: [7, 7],strides:[2,2], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));
modelL.add(tf.layers.leakyReLU());
modelL.add(tf.layers.conv2d({ filters: 32, kernelSize: [5, 5], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));
modelL.add(tf.layers.leakyReLU());
modelL.add(tf.layers.conv2d({ filters: 1, kernelSize: [3, 3], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));

modelL.summary();


export let modelC = tf.sequential();
modelC.add(tf.layers.conv2d({ filters: 128, kernelSize: [9, 9], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true, inputShape: [120,160, 1]}));
modelC.add(tf.layers.leakyReLU());
modelC.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: [7, 7],strides:[2,2], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));
modelC.add(tf.layers.leakyReLU());
modelC.add(tf.layers.conv2d({ filters: 32, kernelSize: [5, 5], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));
modelC.add(tf.layers.leakyReLU());
modelC.add(tf.layers.conv2d({ filters: 1, kernelSize: [3, 3], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'same', useBias: true }));

modelC.summary();