package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;

/**
 * NiN uses convolutional layers with window shapes of 11×11 , 5×5 , and 3×3 , and the corresponding
 * numbers of output channels are the same as in AlexNet. Each NiN block is followed by a maximum
 * pooling layer with a stride of 2 and a window shape of 3×3 .
 *
 * <p>The conventional convolutional layer uses linear filters followed by a nonlinear activation
 * function to scan the input.
 *
 * <p>NiN model from the "Network In Network" http://arxiv.org/abs/1312.4400 paper.
 */
public final class NiN {

    /**
     * The NiN block consists of one convolutional layer followed by two 1×1 convolutional layers
     * that act as per-pixel fully-connected layers with ReLU activations. The convolution width of
     * the first layer is typically set by the user. The subsequent widths are fixed to 1×1 .
     *
     * @return a NiN block.
     */
    public static Block niN() {

        NiN nin = new NiN();
        SequentialBlock block =
                new SequentialBlock()
                        .add(nin.niNBlock(96, new Shape(11, 11), new Shape(4, 4), new Shape(0, 0)))
                        .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                        .add(nin.niNBlock(256, new Shape(5, 5), new Shape(1, 1), new Shape(2, 2)))
                        .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                        .add(nin.niNBlock(384, new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                        .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2)))
                        .add(Dropout.builder().optRate(0.5f).build())
                        // There are 10 label classes
                        .add(nin.niNBlock(10, new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                        // The global average pooling layer automatically sets the window shape
                        // to the height and width of the input
                        .add(Pool.globalAvgPool2dBlock())
                        // Transform the four-dimensional output into two-dimensional output
                        // with a shape of (batch size, 10)
                        .add(Blocks.batchFlattenBlock());

        return block;
    }

    /**
     * @param numChannels the number of channels in a NiN block
     * @param kernelShape kernel Shape in the 1st convolutional layer of a NiN block
     * @param strideShape stride Shape in a NiN block
     * @param paddingShape padding Shape in a NiN block
     * @return
     */
    public SequentialBlock niNBlock(
            int numChannels, Shape kernelShape, Shape strideShape, Shape paddingShape) {

        SequentialBlock tempBlock =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setKernelShape(kernelShape)
                                        .optStride(strideShape)
                                        .optPadding(paddingShape)
                                        .setFilters(numChannels)
                                        .build())
                        .add(Activation::relu)
                        .add(
                                Conv2d.builder()
                                        .setKernelShape(new Shape(1, 1))
                                        .setFilters(numChannels)
                                        .build())
                        .add(Activation::relu)
                        .add(
                                Conv2d.builder()
                                        .setKernelShape(new Shape(1, 1))
                                        .setFilters(numChannels)
                                        .build())
                        .add(Activation::relu);

        return tempBlock;
    }
}
