/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.training.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

/** Base class for all training metrics. */
public abstract class TrainingMetrics implements Cloneable {

    private String name;

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name String, name of the metric
     */
    public TrainingMetrics(String name) {
        this.name = name;
    }

    public TrainingMetrics duplicate() {
        try {
            return (TrainingMetrics) clone();
        } catch (CloneNotSupportedException e) {
            // ignore
            throw new AssertionError("Clone is not supported", e);
        }
    }

    /**
     * Update training metrics based on {@link NDList} of labels and predictions.
     *
     * @param labels {@code NDList} of labels
     * @param predictions {@code NDList} of predictions
     * @return NDArray came from the update, used for losses
     */
    public abstract NDArray update(NDList labels, NDList predictions);

    /** reset metric values. */
    public abstract void reset();

    /**
     * calculate metric values.
     *
     * @return {@link Pair} of metric name and value
     */
    public abstract Pair<String, Float> getMetric();

    public String getName() {
        return name;
    }

    /**
     * Check if the two input {@code NDArray} have the same length or shape.
     *
     * @param labels {@code NDArray} of labels
     * @param predictions {@code NDArray} of predictions
     * @param checkDimOnly whether to check for first dimension only
     */
    protected void checkLabelShapes(NDArray labels, NDArray predictions, boolean checkDimOnly) {
        if (labels.getShape().get(0) != predictions.getShape().get(0)) {
            throw new IllegalArgumentException(
                    "The size of labels("
                            + labels.size()
                            + ") does not match that of predictions("
                            + predictions.size()
                            + ")");
        }
        if (!checkDimOnly) {
            if (labels.getShape() != predictions.getShape()) {
                throw new IllegalArgumentException(
                        "The shape of labels("
                                + labels.getShape()
                                + ") does not match that of predictions("
                                + predictions.getShape()
                                + ")");
            }
        }
    }

    /**
     * Convenient method for checking length of NDArrays.
     *
     * @param labels {@code NDArray} of labels
     * @param predictions {@code NDArray} of predictions
     */
    protected void checkLabelShapes(NDArray labels, NDArray predictions) {
        checkLabelShapes(labels, predictions, true);
    }
}
