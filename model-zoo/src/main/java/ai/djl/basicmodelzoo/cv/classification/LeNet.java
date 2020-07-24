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
import ai.djl.nn.pooling.Pool;

/**
 * The model was introduced (and named for) Yann Lecun, for the purpose of recognizing handwritten
 * digits in images [LeNet5](http://yann.lecun.com/exdb/lenet/).
 */
public final class LeNet {

    private LeNet() {}

    /**
     * creates a LeNet network block.
     *
     * @return a LeNet Sequential block
     */
    public static Block leNetBlock() {

        SequentialBlock block =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setKernelShape(new Shape(5, 5))
                                        .optPadding(new Shape(2, 2))
                                        .optBias(false)
                                        .setFilters(6)
                                        .build())
                        .add(Activation::sigmoid)
                        .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                        .add(
                                Conv2d.builder()
                                        .setKernelShape(new Shape(5, 5))
                                        .setFilters(16)
                                        .build())
                        .add(Activation::sigmoid)
                        .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                        // Blocks.batchFlattenBlock() will transform the input of the shape (batch
                        // size, channel,
                        // height, width) into the input of the shape (batch size,
                        // channel * height * width)
                        .add(Blocks.batchFlattenBlock())
                        .add(Linear.builder().setUnits(120).build())
                        .add(Activation::sigmoid)
                        .add(Linear.builder().setUnits(84).build())
                        .add(Activation::sigmoid)
                        .add(Linear.builder().setUnits(10).build());

        return block;
    }
}
