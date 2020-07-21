package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * GoogLeNet uses a stack of a total of 9 inception blocks and global average pooling to generate
 * its estimates. Maximum pooling between inception blocks reduced the dimensionality. The first
 * part is identical to AlexNet and LeNet, the stack of blocks is inherited from VGG and the global
 * average pooling avoids a stack of fully-connected layers at the end.
 *
 * <p>GoogLeNet paper from Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
 * Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich “Going Deeper with
 * Convolutions” https://arxiv.org/abs/1409.4842
 */
public final class GoogLeNet {

    /**
     * creates a GoogLeNet network block
     *
     * @return
     */
    public static Block GoogLeNet() {

        GoogLeNet googLeNet = new GoogLeNet();
        // creation of block1
        SequentialBlock block1 = new SequentialBlock();
        block1.add(
                        Conv2d.builder()
                                .setKernelShape(new Shape(7, 7))
                                .optPadding(new Shape(3, 3))
                                .optStride(new Shape(2, 2))
                                .setFilters(64)
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        // creation of block2
        SequentialBlock block2 = new SequentialBlock();
        block2.add(Conv2d.builder().setFilters(64).setKernelShape(new Shape(1, 1)).build())
                .add(Activation::relu)
                .add(
                        Conv2d.builder()
                                .setFilters(192)
                                .setKernelShape(new Shape(3, 3))
                                .optPadding(new Shape(1, 1))
                                .build())
                .add(Activation::relu)
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        // creation of block3
        SequentialBlock block3 = new SequentialBlock();
        block3.add(googLeNet.inceptionBlock(64, new int[] {96, 128}, new int[] {16, 32}, 32))
                .add(googLeNet.inceptionBlock(128, new int[] {128, 192}, new int[] {32, 96}, 64))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        // creation of block4
        SequentialBlock block4 = new SequentialBlock();
        block4.add(googLeNet.inceptionBlock(192, new int[] {96, 208}, new int[] {16, 48}, 64))
                .add(googLeNet.inceptionBlock(160, new int[] {112, 224}, new int[] {24, 64}, 64))
                .add(googLeNet.inceptionBlock(128, new int[] {128, 256}, new int[] {24, 64}, 64))
                .add(googLeNet.inceptionBlock(112, new int[] {144, 288}, new int[] {32, 64}, 64))
                .add(googLeNet.inceptionBlock(256, new int[] {160, 320}, new int[] {32, 128}, 128))
                .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));

        // creation of block5
        SequentialBlock block5 = new SequentialBlock();
        block5.add(googLeNet.inceptionBlock(256, new int[] {160, 320}, new int[] {32, 128}, 128))
                .add(googLeNet.inceptionBlock(384, new int[] {192, 384}, new int[] {48, 128}, 128))
                .add(Pool.globalAvgPool2dBlock());

        return new SequentialBlock()
                .addAll(
                        block1,
                        block2,
                        block3,
                        block4,
                        block5,
                        Linear.builder().setUnits(10).build());
    }

    // c1 - c4 are the number of output channels for each layer in the path
    public ParallelBlock inceptionBlock(int c1, int[] c2, int[] c3, int c4) {

        // Path 1 is a single 1 x 1 convolutional layer
        SequentialBlock p1 =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setFilters(c1)
                                        .setKernelShape(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu);

        // Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        // convolutional layer
        SequentialBlock p2 =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setFilters(c2[0])
                                        .setKernelShape(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu)
                        .add(
                                Conv2d.builder()
                                        .setFilters(c2[1])
                                        .setKernelShape(new Shape(3, 3))
                                        .optPadding(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu);

        // Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        // convolutional layer
        SequentialBlock p3 =
                new SequentialBlock()
                        .add(
                                Conv2d.builder()
                                        .setFilters(c3[0])
                                        .setKernelShape(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu)
                        .add(
                                Conv2d.builder()
                                        .setFilters(c3[1])
                                        .setKernelShape(new Shape(5, 5))
                                        .optPadding(new Shape(2, 2))
                                        .build())
                        .add(Activation::relu);

        // Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        // convolutional layer
        SequentialBlock p4 =
                new SequentialBlock()
                        .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(1, 1), new Shape(1, 1)))
                        .add(
                                Conv2d.builder()
                                        .setFilters(c4)
                                        .setKernelShape(new Shape(1, 1))
                                        .build())
                        .add(Activation::relu);

        // Concatenate the outputs on the channel dimension
        return new ParallelBlock(
                list -> {
                    List<NDArray> concatenatedList =
                            list.stream().map(NDList::head).collect(Collectors.toList());

                    return new NDList(NDArrays.concat(new NDList(concatenatedList), 1));
                },
                Arrays.asList(p1, p2, p3, p4));
    }
}
