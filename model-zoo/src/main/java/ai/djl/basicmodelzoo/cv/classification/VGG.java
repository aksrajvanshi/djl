/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

/**
 * VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
 * https://arxiv.org/abs/1409.1556 paper.
 */
public final class VGG {

    private VGG() {}

    /**
     * function to return a VGG block.
     *
     * @param convArch 2-D array consisting of number of convolutions of each layer and number of
     *     filters in each block.
     * @return a VGG Block.
     */
    public static Block vggBlock(int[][] convArch) {

        SequentialBlock block = new SequentialBlock();
        VGG vgg = new VGG();
        // The convolutional layer part
        for (int i = 0; i < convArch.length; i++) {
            block.add(vgg.vggConstituentBlock(convArch[i][0], convArch[i][1]));
        }

        // The fully connected layer part
        block.add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(4096).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.5f).build())
                .add(Linear.builder().setUnits(4096).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.5f).build())
                .add(Linear.builder().setUnits(10).build());

        return block;
    }

    /**
     * creates VGG network constituent blocks.
     *
     * @param numConvs Numbers of layers in each feature block.
     * @param numChannels Numbers of filters in each feature block. List length should match the
     *     layers.
     * @return returns a VGG sequential block.
     */
    public SequentialBlock vggConstituentBlock(int numConvs, int numChannels) {

        SequentialBlock tempBlock = new SequentialBlock();
        for (int i = 0; i < numConvs; i++) {
            tempBlock
                    .add(
                            Conv2d.builder()
                                    .setFilters(numChannels)
                                    .setKernelShape(new Shape(3, 3))
                                    .optPadding(new Shape(1, 1))
                                    .build())
                    .add(Activation::relu);
        }
        tempBlock.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        return tempBlock;
    }
}