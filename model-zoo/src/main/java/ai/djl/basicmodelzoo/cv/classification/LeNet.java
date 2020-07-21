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

    /**
     * creates a LeNet network block
     *
     * @return
     */
    public static Block leNet() {

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
