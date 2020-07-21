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
 * {@code AlexNet} contains a generic implementation of AlexNet adapted from [torchvision
 * implmentation](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)
 *
 * <p>AlexNet model from the "One weird trick..." https://arxiv.org/abs/1404.5997 paper.
 */
public final class AlexNet {

    /**
     * creates a alexNet network block
     *
     * @return
     */
    public static Block alexNet() {
        return new SequentialBlock()
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(11, 11))
                                .optStride(new Shape(4, 4))
                                .setFilters(96)
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Make the convolution window smaller, set padding to 2 for consistent
                // height and width across the input and output, and increase the
                // number of output channels
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(5, 5))
                                .optPadding(new Shape(2, 2))
                                .setFilters(256)
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Use three successive convolutional layers and a smaller convolution
                // window. Except for the final convolutional layer, the number of
                // output channels is further increased. Pooling layers are not used to
                // reduce the height and width of input after the first two
                // convolutional layers
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .setFilters(384)
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .setFilters(384)
                                .build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .setFilters(256)
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                // Here, the number of outputs of the fully connected layer is several
                // times larger than that in LeNet. Use the dropout layer to mitigate
                // over fitting
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(4096).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.5f).build())
                .add(Linear.builder().setUnits(4096).build())
                .add(Activation::relu)
                .add(Dropout.builder().optRate(0.5f).build())
                // Output layer. The number of
                // classes is 10, instead of 1000 as in the paper
                .add(Linear.builder().setUnits(10).build());
    }
}
